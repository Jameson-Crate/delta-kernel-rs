use std::backtrace::Backtrace;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter};

use crate::Version;

include!(concat!(env!("OUT_DIR"), "/delta_error_conditions.rs"));

type BoxedError = Box<dyn StdError + Send + Sync + 'static>;

/// A structured, user-facing error produced while operating on a Delta table.
///
/// The condition, SQLSTATE, and parameter names are stable identifiers. The rendered message is
/// diagnostic text and may evolve with the pinned Delta error catalog.
#[derive(Debug)]
pub struct DeltaError {
    condition: DeltaErrorCondition,
    parameters: Box<[DeltaErrorParameter]>,
    source: Option<BoxedError>,
    backtrace: Backtrace,
}

impl DeltaError {
    /// Returns the typed Delta error condition.
    pub fn condition(&self) -> DeltaErrorCondition {
        self.condition
    }

    /// Returns the stable string identity of the Delta error condition.
    pub fn condition_name(&self) -> &'static str {
        self.condition.name()
    }

    /// Returns the SQLSTATE associated with the condition, if the catalog defines one.
    pub fn sql_state(&self) -> Option<&'static str> {
        self.condition.sql_state()
    }

    /// Returns the named message parameters in template order.
    pub fn parameters(&self) -> &[DeltaErrorParameter] {
        &self.parameters
    }

    /// Renders the user-facing message from the catalog template and parameters.
    pub fn message(&self) -> String {
        render_template(self.condition.message_template(), &self.parameters)
    }

    /// Returns the backtrace captured when the structured error was created.
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }

    pub(crate) fn invalid_cdc_range(start: Version, end: Version) -> Self {
        Self::new(
            DeltaErrorCondition::DeltaInvalidCdcRange,
            vec![
                DeltaErrorParameter::new("start", start),
                DeltaErrorParameter::new("end", end),
            ],
        )
    }

    pub(crate) fn invalid_partition_column_type(
        name: impl ToString,
        data_type: impl ToString,
    ) -> Self {
        Self::new(
            DeltaErrorCondition::DeltaInvalidPartitionColumnType,
            vec![
                DeltaErrorParameter::new("name", name),
                DeltaErrorParameter::new("dataType", data_type),
            ],
        )
    }

    pub(crate) fn versions_not_contiguous(
        version_list: impl ToString,
        start_version: Version,
        end_version: Version,
        version_to_load: Version,
    ) -> Self {
        Self::new(
            DeltaErrorCondition::DeltaVersionsNotContiguous,
            vec![
                DeltaErrorParameter::new("versionList", version_list),
                DeltaErrorParameter::new("startVersion", start_version),
                DeltaErrorParameter::new("endVersion", end_version),
                DeltaErrorParameter::new("versionToLoad", version_to_load),
            ],
        )
    }

    fn new(condition: DeltaErrorCondition, parameters: Vec<DeltaErrorParameter>) -> Self {
        Self {
            condition,
            parameters: parameters.into_boxed_slice(),
            source: None,
            backtrace: Backtrace::capture(),
        }
    }
}

impl Display for DeltaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message())
    }
}

impl StdError for DeltaError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

/// A named parameter used to render a [`DeltaError`] message template.
#[derive(Debug, Eq, PartialEq)]
pub struct DeltaErrorParameter {
    name: &'static str,
    value: String,
}

impl DeltaErrorParameter {
    /// Returns the parameter's stable catalog name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the parameter's display value.
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

fn render_template(template: &str, parameters: &[DeltaErrorParameter]) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut remaining = template;

    while let Some(open) = remaining.find('<') {
        rendered.push_str(&remaining[..open]);
        let placeholder = &remaining[open + 1..];
        let Some(close) = placeholder.find('>') else {
            rendered.push_str(&remaining[open..]);
            return rendered;
        };
        let name = &placeholder[..close];
        if let Some(parameter) = parameters.iter().find(|parameter| parameter.name == name) {
            rendered.push_str(&parameter.value);
        } else {
            rendered.push_str(&remaining[open..open + close + 2]);
        }
        remaining = &placeholder[close + 1..];
    }
    rendered.push_str(remaining);
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_condition_metadata_matches_catalog() {
        let condition = DeltaErrorCondition::DeltaVersionsNotContiguous;
        assert_eq!(condition.name(), "DELTA_VERSIONS_NOT_CONTIGUOUS");
        assert_eq!(condition.sql_state(), Some("KD00C"));
        assert_eq!(
            condition.parameter_names(),
            &["versionList", "startVersion", "endVersion", "versionToLoad"]
        );
    }

    #[test]
    fn factory_parameters_match_catalog() {
        let errors = [
            DeltaError::invalid_cdc_range(2, 1),
            DeltaError::invalid_partition_column_type("payload", "struct<a: string>"),
            DeltaError::versions_not_contiguous("1, 3", 1, 3, 3),
        ];

        for error in errors {
            let parameter_names = error
                .parameters()
                .iter()
                .map(DeltaErrorParameter::name)
                .collect::<Vec<_>>();
            assert_eq!(parameter_names, error.condition().parameter_names());
        }
    }

    #[test]
    fn typed_error_renders_catalog_template() {
        let error = DeltaError::invalid_partition_column_type("payload", "struct<a: string>");
        assert_eq!(
            error.message(),
            "Using column payload of type struct<a: string> as a partition column is not supported."
        );
        assert_eq!(
            error.condition_name(),
            "DELTA_INVALID_PARTITION_COLUMN_TYPE"
        );
        assert_eq!(error.sql_state(), Some("42996"));
        let parameters = error
            .parameters()
            .iter()
            .map(|parameter| (parameter.name(), parameter.value()))
            .collect::<Vec<_>>();
        assert_eq!(
            parameters,
            vec![("name", "payload"), ("dataType", "struct<a: string>"),]
        );
    }
}
