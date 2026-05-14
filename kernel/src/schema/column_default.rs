//! A typed wrapper around Delta's `CURRENT_DEFAULT` column metadata.
//!
//! See [`ColumnDefault`] for the carrier struct,
//! [`crate::schema::StructField::column_default`] for the per-field accessor,
//! and [`crate::transaction::Transaction::column_defaults`] for the
//! top-level-schema sweep used by writers.

use crate::expressions::{parse_sql_literal, Expression};
use crate::schema::DataType;

/// A column-level default value parsed from the `CURRENT_DEFAULT` metadata
/// key of a [`crate::schema::StructField`].
///
/// `sql` is the raw expression string as stored in the column's metadata,
/// `data_type` is the column's declared type, and `parsed` is the best-effort
/// kernel `Expression` produced by [`parse_sql_literal`]. `parsed` is `None`
/// when the kernel's built-in literal parser cannot interpret the SQL --
/// typically because the default is a function call or other non-literal
/// expression (e.g. `CURRENT_TIMESTAMP()`). Callers that understand richer
/// SQL than the kernel can fall back to `sql`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDefault {
    /// The raw SQL expression as stored in the column's metadata.
    pub sql: String,
    /// The declared type of the column whose default this is.
    pub data_type: DataType,
    /// The default parsed as a kernel [`Expression`], or `None` if the
    /// kernel's built-in SQL literal parser could not parse it.
    pub parsed: Option<Expression>,
}

impl ColumnDefault {
    /// Build a `ColumnDefault` from a raw SQL string and the target type.
    ///
    /// Attempts to parse `sql` via [`parse_sql_literal`] using `data_type`
    /// as the expected target type. Parse failures are swallowed -- `parsed`
    /// is left as `None` and no error is propagated -- because not every
    /// stored default is a literal the kernel knows how to interpret.
    pub fn new(sql: String, data_type: DataType) -> Self {
        let parsed = parse_sql_literal(&sql, &data_type).ok();
        Self {
            sql,
            data_type,
            parsed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::Scalar;
    use crate::schema::{ColumnMetadataKey, MetadataValue, StructField};

    #[test]
    fn new_with_parseable_string() {
        let d = ColumnDefault::new("'foo'".into(), DataType::STRING);
        assert_eq!(d.sql, "'foo'");
        assert_eq!(d.data_type, DataType::STRING);
        assert_eq!(
            d.parsed,
            Some(Expression::literal(Scalar::String("foo".into())))
        );
    }

    #[test]
    fn new_with_parseable_int() {
        let d = ColumnDefault::new("42".into(), DataType::INTEGER);
        assert_eq!(d.parsed, Some(Expression::literal(Scalar::Integer(42))));
    }

    #[test]
    fn new_with_null_keyword() {
        let d = ColumnDefault::new("NULL".into(), DataType::INTEGER);
        assert_eq!(
            d.parsed,
            Some(Expression::literal(Scalar::Null(DataType::INTEGER)))
        );
    }

    #[test]
    fn new_with_unparseable_sql_keeps_sql_and_drops_parsed() {
        let d = ColumnDefault::new("CURRENT_TIMESTAMP()".into(), DataType::TIMESTAMP);
        assert_eq!(d.sql, "CURRENT_TIMESTAMP()");
        assert_eq!(d.data_type, DataType::TIMESTAMP);
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
        assert_eq!(got.sql, "42");
        assert_eq!(got.data_type, DataType::INTEGER);
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
}
