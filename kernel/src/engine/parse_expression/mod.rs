//! A default [`ParsingHandler`] that parses SQL literals into kernel expressions.
//!
//! [`LiteralParsingHandler`] handles only literal SQL: numbers, booleans, dates, timestamps,
//! decimals, single-quoted strings, and `null`. It is intended as a baseline -- engines that
//! need to parse column references, function calls, or arithmetic should provide their own
//! [`ParsingHandler`] implementation.

use std::sync::Arc;

use crate::expressions::{Expression, Scalar};
use crate::schema::{DataType, PrimitiveType};
use crate::{DeltaResult, Error, ParsingHandler};

/// Default [`ParsingHandler`] implementation that recognizes only literal SQL expressions.
///
/// Accepted inputs (after trimming surrounding whitespace):
/// - `null` (case-insensitive) -- produces a typed null for any `output_type`.
/// - SQL string literals: `'foo'`, with `''` as the embedded-quote escape. Required when
///   `output_type` is [`PrimitiveType::String`].
/// - All other primitive literals are forwarded to [`PrimitiveType::parse_scalar`], which handles
///   numeric, boolean, date, timestamp, decimal, and binary forms.
///
/// Returns [`Error::ParseError`] for non-literal SQL, non-primitive output types, or any
/// input the literal grammar above does not accept.
#[derive(Debug, Default)]
pub struct LiteralParsingHandler;

impl LiteralParsingHandler {
    /// Construct a new [`LiteralParsingHandler`].
    pub fn new() -> Self {
        Self
    }

    /// Convenience constructor returning an `Arc`-wrapped instance.
    pub fn new_arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl ParsingHandler for LiteralParsingHandler {
    fn parse_sql(&self, sql: &str, output_type: &DataType) -> DeltaResult<Expression> {
        let trimmed = sql.trim();

        // SQL `null` is allowed for any output type.
        if trimmed.eq_ignore_ascii_case("null") {
            return Ok(Expression::literal(Scalar::Null(output_type.clone())));
        }

        let DataType::Primitive(primitive) = output_type else {
            return Err(Error::ParseError(sql.to_string(), output_type.clone()));
        };

        // String literals require SQL single-quote syntax. PrimitiveType::parse_scalar would
        // otherwise treat the raw input (including any quotes) as the string value.
        if primitive == &PrimitiveType::String {
            let unquoted = parse_sql_string_literal(trimmed)
                .ok_or_else(|| Error::ParseError(sql.to_string(), output_type.clone()))?;
            return Ok(Expression::literal(Scalar::String(unquoted)));
        }

        let scalar = primitive.parse_scalar(trimmed)?;
        Ok(Expression::literal(scalar))
    }
}

/// Parse a SQL single-quoted string literal: `'foo'`, with `''` as the embedded-quote escape.
/// Returns `None` if `input` is not a valid SQL string literal.
fn parse_sql_string_literal(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'\'' || bytes[bytes.len() - 1] != b'\'' {
        return None;
    }
    let inner = &input[1..input.len() - 1];

    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' {
            // a lone single quote inside the literal must be escaped by doubling
            chars.next_if_eq(&'\'')?;
            out.push('\'');
        } else {
            out.push(c);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::schema::{DataType, DecimalType, PrimitiveType, StructType};

    fn lit(scalar: Scalar) -> Expression {
        Expression::literal(scalar)
    }

    #[rstest]
    // numeric primitives
    #[case::byte("42", DataType::BYTE, lit(Scalar::Byte(42)))]
    #[case::short("42", DataType::SHORT, lit(Scalar::Short(42)))]
    #[case::int("42", DataType::INTEGER, lit(Scalar::Integer(42)))]
    #[case::long("42", DataType::LONG, lit(Scalar::Long(42)))]
    #[case::int_negative("-7", DataType::INTEGER, lit(Scalar::Integer(-7)))]
    #[case::float("3.5", DataType::FLOAT, lit(Scalar::Float(3.5)))]
    #[case::double("3.5", DataType::DOUBLE, lit(Scalar::Double(3.5)))]
    // boolean (case-insensitive)
    #[case::bool_true("true", DataType::BOOLEAN, lit(Scalar::Boolean(true)))]
    #[case::bool_false("FALSE", DataType::BOOLEAN, lit(Scalar::Boolean(false)))]
    // string (SQL-quoted)
    #[case::string_simple("'hello'", DataType::STRING, lit(Scalar::String("hello".into())))]
    #[case::string_empty("''", DataType::STRING, lit(Scalar::String(String::new())))]
    #[case::string_escape("'it''s'", DataType::STRING, lit(Scalar::String("it's".into())))]
    #[case::string_only_escapes(
        "''''",
        DataType::STRING,
        lit(Scalar::String("'".into()))
    )]
    // null (any type)
    #[case::null_int("null", DataType::INTEGER, lit(Scalar::Null(DataType::INTEGER)))]
    #[case::null_string("NULL", DataType::STRING, lit(Scalar::Null(DataType::STRING)))]
    // whitespace tolerance
    #[case::whitespace_int("  42  ", DataType::INTEGER, lit(Scalar::Integer(42)))]
    #[case::whitespace_string(
        "  'hi'  ",
        DataType::STRING,
        lit(Scalar::String("hi".into()))
    )]
    fn parses_literal(#[case] sql: &str, #[case] ty: DataType, #[case] expected: Expression) {
        let h = LiteralParsingHandler::new();
        assert_eq!(h.parse_sql(sql, &ty).unwrap(), expected);
    }

    #[rstest]
    #[case::malformed_int("abc", DataType::INTEGER)]
    #[case::overflow_byte("999", DataType::BYTE)]
    #[case::bare_identifier("col_a", DataType::INTEGER)]
    #[case::expression_addition("1 + 1", DataType::INTEGER)]
    #[case::string_no_quotes("hello", DataType::STRING)]
    #[case::string_unterminated("'hello", DataType::STRING)]
    #[case::string_lone_inner_quote("'a'b'", DataType::STRING)]
    #[case::string_only_one_quote("'", DataType::STRING)]
    #[case::concat_expression("'a' || 'b'", DataType::STRING)]
    fn rejects_invalid(#[case] sql: &str, #[case] ty: DataType) {
        let h = LiteralParsingHandler::new();
        let err = h.parse_sql(sql, &ty).expect_err("expected parse failure");
        assert!(
            matches!(err, Error::ParseError(..)),
            "expected ParseError, got {err:?}"
        );
    }

    #[test]
    fn rejects_non_primitive_output_type() {
        let h = LiteralParsingHandler::new();
        let struct_ty = DataType::Struct(Box::new(StructType::new_unchecked(Vec::new())));
        let err = h
            .parse_sql("'x'", &struct_ty)
            .expect_err("expected parse failure");
        assert!(matches!(err, Error::ParseError(..)));
    }

    #[test]
    fn parses_date_literal() {
        let h = LiteralParsingHandler::new();
        let parsed = h.parse_sql("2024-01-15", &DataType::DATE).unwrap();
        // 2024-01-15 is 19737 days after 1970-01-01.
        assert_eq!(parsed, Expression::literal(Scalar::Date(19737)));
    }

    #[test]
    fn decimal_literal_requires_matching_scale() {
        let h = LiteralParsingHandler::new();
        let dec_type =
            DataType::Primitive(PrimitiveType::Decimal(DecimalType::try_new(10, 2).unwrap()));
        let parsed = h.parse_sql("123.45", &dec_type).unwrap();
        assert!(matches!(parsed, Expression::Literal(Scalar::Decimal(_))));
        // Scale mismatch: declared scale is 2 but the literal has scale 1.
        assert!(h.parse_sql("123.4", &dec_type).is_err());
    }

    #[test]
    fn engine_accessor_returns_handler() {
        use crate::Engine;
        let engine = crate::engine::sync::SyncEngine::new();
        let h = engine.parsing_handler();
        let parsed = h.parse_sql("42", &DataType::INTEGER).unwrap();
        assert_eq!(parsed, Expression::literal(Scalar::Integer(42)));
    }
}
