//! Catalog-backed Delta failures and their diagnostic parameters.

use std::backtrace::Backtrace;
use std::error::Error as StdError;
use std::fmt;
use std::path::Path;

use super::DeltaErrorCondition;

/// Builds diagnostic parameters in the supplied order, stringifying each value.
macro_rules! delta_parameters {
    ($($name:expr => $value:expr),* $(,)?) => {
        vec![$(DeltaErrorParameter::new($name, $value)),*]
    };
}

/// A catalog-defined Delta failure that terminates the current operation.
///
/// Match on [`Self::condition`] and [`Self::sql_state`], not on diagnostic messages or parameter
/// values. A failure does not invalidate the snapshot or require engine shutdown.
///
/// [`fmt::Display`] renders only the catalog message. The original cause is available through
/// [`StdError::source`], and [`fmt::Debug`] includes the cause and backtrace. Diagnostic values are
/// not redacted; connectors are responsible for redacting telemetry.
#[derive(Debug)]
pub struct DeltaError {
    condition: DeltaErrorCondition,
    parameters: Box<[DeltaErrorParameter]>,
    source: Option<Box<dyn StdError + Send + Sync + 'static>>,
    backtrace: Backtrace,
}

/// A named, stringified diagnostic value for a [`DeltaError`] message template.
///
/// The name is a stable catalog identifier; the value is diagnostic text, not a classification key.
#[derive(Debug, Eq, PartialEq)]
pub struct DeltaErrorParameter {
    name: &'static str,
    value: String,
}

impl DeltaError {
    /// Returns the typed Delta condition; [`DeltaErrorCondition::name`] provides its stable name.
    pub fn condition(&self) -> DeltaErrorCondition {
        self.condition
    }

    /// Returns the condition's five-character SQLSTATE, if the catalog defines one.
    pub fn sql_state(&self) -> Option<&'static str> {
        self.condition.sql_state()
    }

    /// Returns the named diagnostic parameters in catalog template order.
    pub fn parameters(&self) -> &[DeltaErrorParameter] {
        &self.parameters
    }

    /// Renders the catalog message, inserting parameter values verbatim.
    ///
    /// Inserted values are not interpreted as further placeholders. The message excludes the cause
    /// and backtrace and is diagnostic text, not a stable classification key.
    pub fn message(&self) -> String {
        self.to_string()
    }

    /// Returns the backtrace captured at construction, subject to Rust's backtrace configuration.
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }

    /// Returns this error with `source` attached as its diagnostic cause, replacing any prior
    /// cause.
    #[must_use]
    pub fn with_source(
        mut self,
        source: impl Into<Box<dyn StdError + Send + Sync + 'static>>,
    ) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Creates a missing-path failure with `path` as its diagnostic table location.
    pub fn delta_path_does_not_exist(path: impl AsRef<Path>) -> Self {
        DeltaError::new(
            DeltaErrorCondition::DeltaPathDoesNotExist,
            delta_parameters!["path" => path.as_ref().display()],
        )
    }

    /// Creates a missing-history failure for `version`, replayed from `checkpoint_version` at
    /// `log_path`. All three values describe the snapshot reconstruction that failed.
    pub fn delta_log_file_not_found(
        version: crate::Version,
        checkpoint_version: crate::Version,
        log_path: &url::Url,
    ) -> Self {
        Self::new(
            DeltaErrorCondition::DeltaLogFileNotFound,
            delta_parameters![
                "version" => version,
                "checkpointVersion" => checkpoint_version,
                "logPath" => log_path,
            ],
        )
    }

    /// Creates a history-gap failure for the diagnostic `version_list`, bounded by `start_version`
    /// and `end_version`, while loading `version_to_load`.
    pub fn delta_versions_not_contiguous(
        version_list: impl ToString,
        start_version: crate::Version,
        end_version: crate::Version,
        version_to_load: crate::Version,
    ) -> Self {
        Self::new(
            DeltaErrorCondition::DeltaVersionsNotContiguous,
            delta_parameters![
                "versionList" => version_list,
                "startVersion" => start_version,
                "endVersion" => end_version,
                "versionToLoad" => version_to_load,
            ],
        )
    }

    /// Creates an adaptive-metadata failure for the explicitly stored column-mapping `mode`.
    pub fn delta_adaptive_metadata_requires_column_mapping_id_mode(mode: impl ToString) -> Self {
        Self::new(
            DeltaErrorCondition::DeltaAdaptiveMetadataRequiresColumnMappingIdMode,
            delta_parameters![
                "feature" => "adaptiveMetadata-preview",
                "prop" => "delta.columnMapping.mode",
                "mode" => mode,
            ],
        )
    }

    /// Creates a duplicate-partition-column failure with `duplicate_columns` as diagnostic names.
    pub fn delta_duplicate_partition_columns(duplicate_columns: impl ToString) -> Self {
        Self::new(
            DeltaErrorCondition::DeltaDuplicateColumnsFoundPartitionColumns,
            delta_parameters!["duplicateCols" => duplicate_columns],
        )
    }

    /// Creates a missing-partition-column failure for `column_name`, with `schema_map` listing
    /// the logical schema fields available to resolve it.
    pub fn delta_partition_column_not_found(
        column_name: impl ToString,
        schema_map: impl ToString,
    ) -> Self {
        Self::new(
            DeltaErrorCondition::DeltaPartitionColumnNotFound,
            delta_parameters!["columnName" => column_name, "schemaMap" => schema_map],
        )
    }

    /// Creates an error without a diagnostic cause and captures its backtrace.
    ///
    /// `parameters` must contain exactly [`DeltaErrorCondition::parameter_names`], in that order.
    /// Condition-specific factories must maintain this invariant.
    fn new(condition: DeltaErrorCondition, parameters: Vec<DeltaErrorParameter>) -> Self {
        Self {
            condition,
            parameters: parameters.into_boxed_slice(),
            source: None,
            backtrace: Backtrace::capture(),
        }
    }
}

impl fmt::Display for DeltaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        render_template(
            self.condition.message_template(),
            &self.parameters,
            formatter,
        )
    }
}

impl StdError for DeltaError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}

impl DeltaErrorParameter {
    /// Returns the parameter's stable catalog name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the parameter's diagnostic string value without redaction.
    pub fn value(&self) -> &str {
        &self.value
    }

    fn new(name: &'static str, value: impl ToString) -> Self {
        Self {
            name,
            value: value.to_string(),
        }
    }
}

fn render_template(
    template: &str,
    parameters: &[DeltaErrorParameter],
    output: &mut impl fmt::Write,
) -> fmt::Result {
    let mut remaining = template;
    while let Some(start) = remaining.find('<') {
        output.write_str(&remaining[..start])?;
        remaining = &remaining[start + 1..];
        if let Some(end) = remaining.find('>') {
            let name = &remaining[..end];
            if let Some(parameter) = parameters.iter().find(|parameter| parameter.name == name) {
                output.write_str(&parameter.value)?;
                remaining = &remaining[end + 1..];
                continue;
            }
        }
        output.write_char('<')?;
    }
    output.write_str(remaining)
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[test]
    fn delta_parameters_render_mixed_value_types() {
        let error = DeltaError::new(
            DeltaErrorCondition::DeltaLogFileNotFound,
            delta_parameters![
                "version" => 42_u64,
                "checkpointVersion" => 40_u64,
                "logPath" => "s3://bucket/table/_delta_log/",
            ],
        );
        assert_eq!(
            error.message(),
            "Unable to retrieve the delta log files to construct table version 42 starting from \
             checkpoint version 40 at s3://bucket/table/_delta_log/."
        );
    }

    #[test]
    fn with_source_preserves_diagnostics_and_replaces_cause() {
        let error = DeltaError::delta_path_does_not_exist("<path>/table");
        assert!(error.source().is_none());
        let condition = error.condition();
        let sql_state = error.sql_state();
        let message = error.message();

        let error = error.with_source(io::Error::new(io::ErrorKind::NotFound, "missing directory"));
        assert_eq!(error.condition(), condition);
        assert_eq!(error.sql_state(), sql_state);
        assert_eq!(error.message(), message);
        assert_eq!(error.parameters()[0].name(), "path");
        assert_eq!(error.parameters()[0].value(), "<path>/table");
        let source = error.source().unwrap().downcast_ref::<io::Error>().unwrap();
        assert_eq!(source.kind(), io::ErrorKind::NotFound);

        let replacement: Box<dyn StdError + Send + Sync> = Box::new(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "access denied",
        ));
        let error = error.with_source(replacement);
        assert_eq!(error.message(), message);
        let source = error.source().unwrap().downcast_ref::<io::Error>().unwrap();
        assert_eq!(source.kind(), io::ErrorKind::PermissionDenied);
    }
}
