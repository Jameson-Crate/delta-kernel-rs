use std::backtrace::Backtrace;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter};

use super::Error;
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

    fn unclassified(source: KernelError) -> Self {
        Self::new(DeltaErrorCondition::DeltaKernelUnclassified, Vec::new()).with_source(source)
    }

    fn new(condition: DeltaErrorCondition, parameters: Vec<DeltaErrorParameter>) -> Self {
        Self {
            condition,
            parameters: parameters.into_boxed_slice(),
            source: None,
            backtrace: Backtrace::capture(),
        }
    }

    fn with_source(mut self, source: impl Into<BoxedError>) -> Self {
        self.source = Some(source.into());
        self
    }
}

impl From<KernelError> for DeltaError {
    fn from(source: KernelError) -> Self {
        let error = match &source.kind {
            KernelErrorKind::InvalidCdcRange { start, end } => {
                Self::invalid_cdc_range(*start, *end)
            }
            KernelErrorKind::InvalidPartitionColumnType { name, data_type } => {
                Self::invalid_partition_column_type(name, data_type)
            }
            KernelErrorKind::VersionsNotContiguous {
                version_list,
                start_version,
                end_version,
                version_to_load,
                ..
            } => Self::versions_not_contiguous(
                version_list,
                *start_version,
                *end_version,
                *version_to_load,
            ),
            KernelErrorKind::Legacy(_) => return Self::unclassified(source),
        };
        error.with_source(source)
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

/// A named parameter used to render a Delta error message template.
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

/// An error produced by kernel implementation code before a Delta-facing API classifies it.
///
/// Option 3 uses this error with KernelResult and converts it to DeltaError only at the public
/// boundary. Engine-specific errors remain outside this prototype.
#[derive(Debug)]
pub struct KernelError {
    kind: KernelErrorKind,
}

impl KernelError {
    pub(crate) fn invalid_cdc_range(start: Version, end: Version) -> Self {
        Self {
            kind: KernelErrorKind::InvalidCdcRange { start, end },
        }
    }

    pub(crate) fn invalid_partition_column_type(
        name: impl ToString,
        data_type: impl ToString,
    ) -> Self {
        Self {
            kind: KernelErrorKind::InvalidPartitionColumnType {
                name: name.to_string(),
                data_type: data_type.to_string(),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn versions_not_contiguous(
        version_list: impl ToString,
        start_version: Version,
        end_version: Version,
        version_to_load: Version,
        first_version: Version,
        second_version: Version,
    ) -> Self {
        Self {
            kind: KernelErrorKind::VersionsNotContiguous {
                version_list: version_list.to_string(),
                start_version,
                end_version,
                version_to_load,
                first_version,
                second_version,
            },
        }
    }
}

impl From<Error> for KernelError {
    fn from(source: Error) -> Self {
        Self {
            kind: KernelErrorKind::Legacy(source),
        }
    }
}

impl From<KernelError> for Error {
    fn from(source: KernelError) -> Self {
        match source.kind {
            KernelErrorKind::InvalidCdcRange { .. } => Error::generic(
                "Failed to build LogSegment: start_version cannot be greater than end_version",
            ),
            KernelErrorKind::InvalidPartitionColumnType { name, data_type } => {
                Error::generic(format!(
                    "Partition column '{name}' has non-primitive type '{data_type}'. \
                     Partition columns must have primitive types."
                ))
            }
            KernelErrorKind::VersionsNotContiguous {
                first_version,
                second_version,
                ..
            } => Error::LogTailVersionsNotContiguous {
                first_version,
                second_version,
            },
            KernelErrorKind::Legacy(source) => source,
        }
    }
}

impl Display for KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            KernelErrorKind::InvalidCdcRange { start, end } => {
                write!(formatter, "invalid CDF range from {start} to {end}")
            }
            KernelErrorKind::InvalidPartitionColumnType { name, data_type } => write!(
                formatter,
                "partition column {name} has unsupported type {data_type}"
            ),
            KernelErrorKind::VersionsNotContiguous {
                first_version,
                second_version,
                ..
            } => write!(
                formatter,
                "log tail versions {first_version} and {second_version} are not contiguous"
            ),
            KernelErrorKind::Legacy(source) => Display::fmt(source, formatter),
        }
    }
}

impl StdError for KernelError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match &self.kind {
            KernelErrorKind::Legacy(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum KernelErrorKind {
    InvalidCdcRange {
        start: Version,
        end: Version,
    },
    InvalidPartitionColumnType {
        name: String,
        data_type: String,
    },
    VersionsNotContiguous {
        version_list: String,
        start_version: Version,
        end_version: Version,
        version_to_load: Version,
        first_version: Version,
        second_version: Version,
    },
    Legacy(Error),
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

    #[test]
    fn classified_kernel_error_becomes_delta_error() {
        let result: super::super::v3::KernelResult<()> = Err(KernelError::invalid_cdc_range(4, 2));
        let error =
            super::super::v3::into_delta_result(result).expect_err("invalid CDF range must fail");

        assert_eq!(error.condition(), DeltaErrorCondition::DeltaInvalidCdcRange);
        assert_eq!(error.parameters()[0].value(), "4");
        assert_eq!(error.parameters()[1].value(), "2");
        assert!(StdError::source(&error).is_some());
    }

    #[test]
    fn classified_kernel_errors_preserve_legacy_forms() {
        let cdf_error = Error::from(KernelError::invalid_cdc_range(4, 2));
        assert_eq!(
            cdf_error.to_string(),
            "Generic delta kernel error: Failed to build LogSegment: start_version cannot be \
             greater than end_version"
        );

        let partition_error = Error::from(KernelError::invalid_partition_column_type(
            "payload",
            "struct<a: string>",
        ));
        assert!(partition_error
            .to_string()
            .contains("Partition column 'payload' has non-primitive type 'struct<a: string>'"));

        let versions_error =
            Error::from(KernelError::versions_not_contiguous("1, 3", 1, 3, 3, 1, 3));
        assert!(matches!(
            versions_error,
            Error::LogTailVersionsNotContiguous {
                first_version: 1,
                second_version: 3
            }
        ));
    }

    #[test]
    fn unclassified_kernel_error_uses_safe_fallback_and_preserves_source() {
        let error = DeltaError::from(KernelError::from(Error::generic(
            "sensitive implementation detail",
        )));
        assert_eq!(error.condition_name(), "DELTA_KERNEL_UNCLASSIFIED");
        assert_eq!(error.sql_state(), Some("XX000"));
        assert_eq!(
            error.to_string(),
            "An unclassified Delta Kernel error occurred."
        );
        assert_eq!(
            StdError::source(&error).map(ToString::to_string),
            Some("Generic delta kernel error: sensitive implementation detail".to_string())
        );
    }
}
