//! A typed wrapper around Delta's `CURRENT_DEFAULT` column metadata.
//!
//! See [`ColumnDefault`] for the carrier struct,
//! [`crate::schema::StructField::column_default`] for the per-field accessor,
//! and [`crate::transaction::Transaction::column_defaults`] for the
//! top-level-schema sweep used by writers.

use crate::expressions::{parse_sql_literal, Expression, Scalar};
use crate::schema::DataType;
use crate::{DeltaResult, Engine, Error};

/// A column-level default value parsed from the `CURRENT_DEFAULT` metadata
/// key of a [`crate::schema::StructField`].
///
/// The carrier holds the raw SQL string (via [`Self::sql`]) and the declared
/// column type (via [`Self::data_type`]). On construction the kernel attempts
/// to parse the SQL via [`parse_sql_literal`] and caches the result
/// internally; callers can check whether that parse succeeded via
/// [`Self::is_parseable`] and resolve the default to a [`Scalar`] via
/// [`Self::evaluate`]. When the kernel cannot parse the SQL (typically a
/// function call such as `CURRENT_TIMESTAMP()`), connectors with richer SQL
/// support can fall back to evaluating [`Self::sql`] themselves.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDefault {
    sql: String,
    data_type: DataType,
    /// The default parsed as a kernel [`Expression`], or `None` if the
    /// kernel's built-in SQL literal parser could not parse it.
    parsed: Option<Expression>,
}

impl ColumnDefault {
    /// Build a `ColumnDefault` from a raw SQL string and the target type.
    ///
    /// Attempts to parse `sql` via [`parse_sql_literal`] using `data_type`
    /// as the expected target type. Parse failures are swallowed -- the
    /// parsed form is left empty and no error is propagated -- because not
    /// every stored default is a literal the kernel knows how to interpret.
    /// Use [`Self::is_parseable`] to check whether the parse succeeded
    /// without invoking the engine.
    pub fn new(sql: String, data_type: DataType) -> Self {
        let parsed = parse_sql_literal(&sql, &data_type).ok();
        Self {
            sql,
            data_type,
            parsed,
        }
    }

    /// The raw SQL expression as stored in the column's metadata.
    ///
    /// Connectors with a SQL engine richer than the kernel's literal parser
    /// can evaluate this directly when [`Self::is_parseable`] returns
    /// `false`. Delta column defaults are required to be foldable, so this
    /// expression must be evaluated exactly once at write time and broadcast
    /// to all rows -- never evaluated per row.
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// The declared type of the column whose default this is.
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Returns `true` when the kernel was able to parse [`Self::sql`] into a
    /// form it can evaluate, and `false` otherwise.
    ///
    /// This is a cheap, side-effect-free predicate that lets a connector
    /// decide upfront whether to route evaluation through [`Self::evaluate`]
    /// (kernel-parseable defaults) or through its own SQL engine acting on
    /// [`Self::sql`] (everything else). It does not require an [`Engine`]
    /// reference.
    pub fn is_parseable(&self) -> bool {
        self.parsed.is_some()
    }

    /// Evaluate the parsed default to a [`Scalar`].
    ///
    /// Returns an error when the kernel could not parse [`Self::sql`] (i.e.
    /// [`Self::is_parseable`] returns `false`). In that case the caller is
    /// expected to fall back to its own SQL engine using [`Self::sql`].
    ///
    /// The `engine` parameter is currently unused: today's
    /// [`parse_sql_literal`] only emits [`Expression::Literal`], which is
    /// resolved by AST destructure without invoking the engine. It is part
    /// of the public signature so that when the kernel's SQL parser is
    /// extended to richer constant expressions (e.g. `1 + 2`,
    /// `CAST(0 AS BIGINT)`), evaluation can route through
    /// [`crate::EvaluationHandler`] without a breaking API change.
    pub fn evaluate(&self, _engine: &dyn Engine) -> DeltaResult<Scalar> {
        let expr = self.parsed.as_ref().ok_or_else(|| {
            Error::generic(format!(
                "kernel could not parse column default {:?}; evaluate via connector's SQL engine",
                self.sql
            ))
        })?;
        match expr {
            Expression::Literal(s) => Ok(s.clone()),
            other => Err(Error::generic(format!(
                "kernel cannot evaluate non-literal column default expression: {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sync::SyncEngine;
    use crate::schema::{ColumnMetadataKey, MetadataValue, StructField};

    #[test]
    fn new_with_parseable_string() {
        let d = ColumnDefault::new("'foo'".into(), DataType::STRING);
        assert_eq!(d.sql(), "'foo'");
        assert_eq!(d.data_type(), &DataType::STRING);
        assert!(d.is_parseable());
        assert_eq!(
            d.parsed,
            Some(Expression::literal(Scalar::String("foo".into())))
        );
    }

    #[test]
    fn new_with_parseable_int() {
        let d = ColumnDefault::new("42".into(), DataType::INTEGER);
        assert!(d.is_parseable());
        assert_eq!(d.parsed, Some(Expression::literal(Scalar::Integer(42))));
    }

    #[test]
    fn new_with_null_keyword() {
        let d = ColumnDefault::new("NULL".into(), DataType::INTEGER);
        assert!(d.is_parseable());
        assert_eq!(
            d.parsed,
            Some(Expression::literal(Scalar::Null(DataType::INTEGER)))
        );
    }

    #[test]
    fn new_with_unparseable_sql_keeps_sql_and_drops_parsed() {
        let d = ColumnDefault::new("CURRENT_TIMESTAMP()".into(), DataType::TIMESTAMP);
        assert_eq!(d.sql(), "CURRENT_TIMESTAMP()");
        assert_eq!(d.data_type(), &DataType::TIMESTAMP);
        assert!(!d.is_parseable());
        assert_eq!(d.parsed, None);
    }

    #[test]
    fn struct_field_with_no_metadata_has_no_default() {
        let f = StructField::nullable("a", DataType::INTEGER);
        assert_eq!(f.column_default(), None);
    }

    #[test]
    fn struct_field_with_string_metadata_returns_default() {
        let f = StructField::nullable("a", DataType::INTEGER).add_metadata([(
            ColumnMetadataKey::CurrentDefault.as_ref(),
            MetadataValue::String("42".into()),
        )]);
        let got = f.column_default().expect("default present");
        assert_eq!(got.sql(), "42");
        assert_eq!(got.data_type(), &DataType::INTEGER);
        assert!(got.is_parseable());
        assert_eq!(got.parsed, Some(Expression::literal(Scalar::Integer(42))));
    }

    #[test]
    fn struct_field_with_non_string_metadata_is_ignored() {
        // Defensive: defaults are SQL strings. A number stored here means a
        // malformed schema; we return None rather than guessing.
        let f = StructField::nullable("a", DataType::INTEGER).add_metadata([(
            ColumnMetadataKey::CurrentDefault.as_ref(),
            MetadataValue::Number(42),
        )]);
        assert_eq!(f.column_default(), None);
    }

    #[test]
    fn struct_field_with_unparseable_default_still_returned() {
        // Parse failure does NOT hide the default -- callers see the raw sql
        // and can use their own parser.
        let f = StructField::nullable("a", DataType::TIMESTAMP).add_metadata([(
            ColumnMetadataKey::CurrentDefault.as_ref(),
            MetadataValue::String("CURRENT_TIMESTAMP()".into()),
        )]);
        let got = f.column_default().expect("default present");
        assert_eq!(got.sql, "CURRENT_TIMESTAMP()");
        assert_eq!(got.parsed, None);
    }

    #[test]
    fn evaluate_returns_scalar_for_parseable_literal() {
        let engine = SyncEngine::new();
        let d = ColumnDefault::new("42".into(), DataType::INTEGER);
        assert_eq!(d.evaluate(&engine).unwrap(), Scalar::Integer(42));
    }

    #[test]
    fn evaluate_returns_string_scalar() {
        let engine = SyncEngine::new();
        let d = ColumnDefault::new("'hello'".into(), DataType::STRING);
        assert_eq!(d.evaluate(&engine).unwrap(), Scalar::String("hello".into()));
    }

    #[test]
    fn evaluate_returns_null_scalar() {
        let engine = SyncEngine::new();
        let d = ColumnDefault::new("NULL".into(), DataType::INTEGER);
        assert_eq!(
            d.evaluate(&engine).unwrap(),
            Scalar::Null(DataType::INTEGER)
        );
    }

    #[test]
    fn evaluate_errors_when_sql_unparseable() {
        let engine = SyncEngine::new();
        let d = ColumnDefault::new("CURRENT_TIMESTAMP()".into(), DataType::TIMESTAMP);
        let err = d.evaluate(&engine).unwrap_err().to_string();
        assert!(err.contains("kernel could not parse"), "got: {err}");
        assert!(err.contains("CURRENT_TIMESTAMP()"), "got: {err}");
    }

    #[test]
    fn evaluate_errors_when_parsed_is_non_literal() {
        // Hand-construct a ColumnDefault whose `parsed` is a non-literal
        // expression. The public `new()` cannot produce this today, but the
        // evaluate API must reject it cleanly for forward compatibility.
        let d = ColumnDefault {
            sql: "x".into(),
            data_type: DataType::INTEGER,
            parsed: Some(Expression::column(["x"])),
        };
        let engine = SyncEngine::new();
        let err = d.evaluate(&engine).unwrap_err().to_string();
        assert!(err.contains("non-literal"), "got: {err}");
    }
}
