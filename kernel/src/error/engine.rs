#[cfg(any(test, feature = "default-engine-base"))]
use super::Error;
use crate::schema::DataType;

/// A result returned by an engine-provided operation.
pub type EngineResult<T> = std::result::Result<T, EngineError>;

/// A boxed, `Send` iterator of [`EngineResult<T>`] items.
pub type EngineResultIterator<'a, T> = Box<dyn Iterator<Item = EngineResult<T>> + Send + 'a>;

/// `'static` counterpart to [`EngineResultIterator`].
pub type EngineResultIteratorStatic<T> = EngineResultIterator<'static, T>;

/// An error produced by an engine implementation.
///
/// Kernel wraps this error in [`super::Error::Engine`] when it crosses back into a kernel API.
#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    /// The requested file does not exist.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// A file could not be created because it already exists.
    #[error("File already exists: {0}")]
    FileAlreadyExists(String),

    /// The operation was cancelled.
    #[error("Operation cancelled")]
    Cancelled,

    /// An argument passed to the engine is invalid.
    #[error("Invalid engine argument: {0}")]
    InvalidArgument(String),

    /// Engine data does not satisfy the requested schema or representation.
    #[error("Invalid engine data: {0}")]
    InvalidEngineData(String),

    /// The engine does not support the requested operation.
    #[error("Unsupported engine operation: {0}")]
    Unsupported(String),

    /// An engine value could not be parsed as the requested data type.
    #[error("Failed to parse value '{0}' as '{1}'")]
    ParseError(String, DataType),

    /// An I/O operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// An engine failure described by a message.
    #[error("Engine error: {0}")]
    Generic(String),

    /// An engine failure preserving its original error as the source.
    #[error("External engine error: {0}")]
    External(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
}

#[cfg(any(test, feature = "default-engine-base"))]
pub(crate) fn to_engine_error(error: Error) -> EngineError {
    match error {
        Error::Engine(error) => error,
        Error::Backtraced { source, .. } => to_engine_error(*source),
        Error::Generic(message) => EngineError::Generic(message),
        Error::GenericError { source } => EngineError::External(source),
        Error::FileNotFound(path) => EngineError::FileNotFound(path),
        Error::FileAlreadyExists(path) => EngineError::FileAlreadyExists(path),
        Error::IOError(error) => EngineError::Io(error),
        Error::ParseError(value, data_type) => EngineError::ParseError(value, data_type),
        Error::Unsupported(message) => EngineError::Unsupported(message),
        Error::Cancelled => EngineError::Cancelled,
        Error::EngineDataType(message) => EngineError::InvalidEngineData(message),
        Error::InvalidExpressionEvaluation(message)
        | Error::InvalidTableLocation(message)
        | Error::MissingColumn(message)
        | Error::UnexpectedColumnType(message) => EngineError::InvalidArgument(message),
        error => EngineError::External(Box::new(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_error_is_wrapped_without_losing_its_classification() {
        let error = Error::from(EngineError::FileNotFound("missing".to_string()));
        assert!(matches!(
            error.unbacktraced(),
            Error::Engine(EngineError::FileNotFound(path)) if path == "missing"
        ));
    }

    #[test]
    fn engine_error_classifications_survive_kernel_wrapping() {
        let parse = Error::from(EngineError::ParseError(
            "invalid".to_string(),
            DataType::INTEGER,
        ));
        let io = Error::from(EngineError::Io(std::io::Error::other("transient")));
        let conflict = Error::from(EngineError::FileAlreadyExists("conflict".to_string()));
        let cancelled = Error::from(EngineError::Cancelled);

        assert!(parse.is_parse_error());
        assert!(io.is_io_error());
        assert!(conflict.is_file_already_exists());
        assert!(cancelled.is_cancelled());
    }
}
