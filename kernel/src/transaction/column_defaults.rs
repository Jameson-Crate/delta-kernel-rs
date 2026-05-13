//! Column default value support for [`Transaction`].
//!
//! Column defaults are part of the Delta protocol's `allowColumnDefaults` writer feature.
//! Per spec, a column's default is stored as a SQL string in the field's metadata under the
//! `CURRENT_DEFAULT` key. Defaults are only valid on top-level (non-nested) fields.
//!
//! Kernel attempts to parse each default eagerly when the [`Transaction`] is constructed,
//! using the engine's [`ParsingHandler`]. If parsing fails (e.g. the SQL is a non-literal
//! expression the default [`LiteralParsingHandler`] cannot handle), the raw SQL is preserved
//! and `parsed_expr` is `None` so the engine can apply its own parser.
//!
//! [`Transaction`]: crate::transaction::Transaction
//! [`ParsingHandler`]: crate::ParsingHandler
//! [`LiteralParsingHandler`]: crate::engine::parse_expression::LiteralParsingHandler

use std::collections::HashMap;

use crate::schema::{DataType, MetadataValue, StructType};
use crate::{Engine, Expression};

/// The `CURRENT_DEFAULT` field-metadata key from the Delta protocol's `allowColumnDefaults`
/// writer feature.
pub(crate) const COLUMN_DEFAULT_METADATA_KEY: &str = "CURRENT_DEFAULT";

/// A parsed column default for a single top-level field.
///
/// Defaults are stored in field metadata as SQL strings under the `CURRENT_DEFAULT` key.
/// Kernel attempts to parse each SQL string into an [`Expression`] when the [`Transaction`]
/// is created. If parsing fails, [`parsed_expr`](Self::parsed_expr) is `None` and the engine
/// can fall back to the raw [`sql`](Self::sql) string.
///
/// [`Transaction`]: crate::transaction::Transaction
#[derive(Debug, Clone)]
pub struct DefaultColumn {
    /// The logical data type of the column the default applies to.
    pub data_type: DataType,
    /// The raw SQL string stored in field metadata under `CURRENT_DEFAULT`.
    pub sql: String,
    /// The parsed expression, or `None` if the SQL could not be parsed by the engine's
    /// [`ParsingHandler`](crate::ParsingHandler).
    pub parsed_expr: Option<Expression>,
}

/// Build a map of column defaults by inspecting the top-level fields of `schema`.
///
/// Only top-level fields are examined; the Delta protocol does not permit `CURRENT_DEFAULT`
/// on nested fields. Fields without the metadata key, or whose value at that key is not a
/// string, are skipped.
pub(crate) fn extract_column_defaults(
    schema: &StructType,
    engine: &dyn Engine,
) -> HashMap<String, DefaultColumn> {
    let parser = engine.parsing_handler();
    schema
        .fields()
        .filter_map(|field| {
            let MetadataValue::String(sql) = field.metadata.get(COLUMN_DEFAULT_METADATA_KEY)?
            else {
                return None;
            };
            let parsed_expr = parser.parse_sql(sql, &field.data_type).ok();
            Some((
                field.name.clone(),
                DefaultColumn {
                    data_type: field.data_type.clone(),
                    sql: sql.clone(),
                    parsed_expr,
                },
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::sync::SyncEngine;
    use crate::expressions::Scalar;
    use crate::schema::{StructField, StructType};

    fn field_with_default(name: &str, ty: DataType, sql: &str) -> StructField {
        StructField::nullable(name, ty).with_metadata([(
            COLUMN_DEFAULT_METADATA_KEY,
            MetadataValue::String(sql.into()),
        )])
    }

    #[test]
    fn no_defaults_returns_empty_map() {
        let schema = StructType::new_unchecked(vec![
            StructField::nullable("a", DataType::INTEGER),
            StructField::nullable("b", DataType::STRING),
        ]);
        let engine = SyncEngine::new();
        let defaults = extract_column_defaults(&schema, &engine);
        assert!(defaults.is_empty());
    }

    #[test]
    fn parses_integer_literal_default() {
        let schema =
            StructType::new_unchecked(vec![field_with_default("n", DataType::INTEGER, "42")]);
        let engine = SyncEngine::new();
        let defaults = extract_column_defaults(&schema, &engine);

        assert_eq!(defaults.len(), 1);
        let d = defaults.get("n").unwrap();
        assert_eq!(d.data_type, DataType::INTEGER);
        assert_eq!(d.sql, "42");
        assert_eq!(
            d.parsed_expr,
            Some(Expression::literal(Scalar::Integer(42)))
        );
    }

    #[test]
    fn parses_string_literal_default() {
        let schema =
            StructType::new_unchecked(vec![field_with_default("s", DataType::STRING, "'hello'")]);
        let engine = SyncEngine::new();
        let defaults = extract_column_defaults(&schema, &engine);

        let d = defaults.get("s").unwrap();
        assert_eq!(d.sql, "'hello'");
        assert_eq!(
            d.parsed_expr,
            Some(Expression::literal(Scalar::String("hello".into())))
        );
    }

    #[test]
    fn non_literal_default_preserves_sql_and_clears_parsed_expr() {
        let schema = StructType::new_unchecked(vec![field_with_default(
            "ts",
            DataType::TIMESTAMP,
            "CURRENT_TIMESTAMP()",
        )]);
        let engine = SyncEngine::new();
        let defaults = extract_column_defaults(&schema, &engine);

        let d = defaults.get("ts").unwrap();
        assert_eq!(d.sql, "CURRENT_TIMESTAMP()");
        assert!(d.parsed_expr.is_none());
    }

    #[test]
    fn mixed_defaulted_and_undefaulted_fields() {
        let schema = StructType::new_unchecked(vec![
            field_with_default("a", DataType::INTEGER, "1"),
            StructField::nullable("b", DataType::STRING),
            field_with_default("c", DataType::STRING, "'x'"),
        ]);
        let engine = SyncEngine::new();
        let defaults = extract_column_defaults(&schema, &engine);

        let mut keys: Vec<&String> = defaults.keys().collect();
        keys.sort();
        assert_eq!(keys, vec!["a", "c"]);
    }

    #[test]
    fn non_string_metadata_value_is_skipped() {
        // CURRENT_DEFAULT must be a string per protocol; a non-string value is ignored.
        let field = StructField::nullable("n", DataType::INTEGER)
            .with_metadata([(COLUMN_DEFAULT_METADATA_KEY, MetadataValue::Number(42))]);
        let schema = StructType::new_unchecked(vec![field]);
        let engine = SyncEngine::new();
        let defaults = extract_column_defaults(&schema, &engine);
        assert!(defaults.is_empty());
    }

    #[test]
    fn nested_field_defaults_are_not_surfaced() {
        // Even if a nested field has CURRENT_DEFAULT metadata, we don't recurse.
        let inner = field_with_default("inner", DataType::INTEGER, "7");
        let nested = StructField::nullable(
            "outer",
            DataType::Struct(Box::new(StructType::new_unchecked(vec![inner]))),
        );
        let schema = StructType::new_unchecked(vec![nested]);
        let engine = SyncEngine::new();
        let defaults = extract_column_defaults(&schema, &engine);
        assert!(defaults.is_empty());
    }

    // The Arc<dyn Engine> path also works.
    #[test]
    fn works_via_dyn_engine_reference() {
        let schema =
            StructType::new_unchecked(vec![field_with_default("n", DataType::INTEGER, "1")]);
        let engine: Arc<dyn Engine> = Arc::new(SyncEngine::new());
        let defaults = extract_column_defaults(&schema, engine.as_ref());
        assert_eq!(defaults.len(), 1);
    }
}
