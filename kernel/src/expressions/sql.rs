//! Parse a SQL literal string into a kernel [`Expression`].
//!
//! Delta stores column defaults, check constraints, and generated column
//! definitions as SQL strings in table metadata. This module turns the
//! literal subset of that grammar into [`Expression::Literal`] values so
//! the kernel can interpret them without depending on a full SQL parser.
//!
//! Non-literal SQL (operators, casts, function calls, column references) is
//! out of scope here and will return an error. A future change will pull in
//! a full parser to support check constraints and generated columns.
//!
//! See [`parse_sql_literal`] for the accepted grammar.

use std::borrow::Cow;

use crate::expressions::{Expression, Scalar};
use crate::schema::{DataType, PrimitiveType};
use crate::{DeltaResult, Error};

/// Parse a SQL literal string into an `Expression::Literal` of the given type.
///
/// The caller supplies the expected `DataType` (e.g. the type of the column
/// whose default is being parsed), and the parser produces a literal of that
/// exact type or returns an error.
///
/// # Accepted grammar
///
/// Leading and trailing whitespace are ignored. Keywords (`NULL`, `TRUE`,
/// `FALSE`, `DATE`, `TIMESTAMP`, `X`) are case-insensitive.
///
/// - `NULL` -- valid for any primitive type
/// - String:    `'foo'`, with `''` interpreted as an embedded single quote
/// - Boolean:   `TRUE` / `FALSE`
/// - Integer / Long / Short / Byte / Float / Double / Decimal: bare numeric literal with optional
///   leading `+` or `-`
/// - Date:      `'2024-01-01'` or `DATE '2024-01-01'`
/// - Timestamp / TimestampNtz: `'2024-01-01 12:00:00[.fff]'` or `TIMESTAMP '...'`. Timestamps also
///   accept ISO 8601 / RFC 3339 form.
/// - Binary:    `X'deadbeef'` (even number of hex digits)
///
/// # Errors
///
/// Returns an error if `data_type` is not a primitive type, if the input
/// does not match a supported literal form, or if the value is out of range
/// for the target type.
pub fn parse_sql_literal(sql: &str, data_type: &DataType) -> DeltaResult<Expression> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err(Error::generic("empty SQL literal"));
    }

    if trimmed.eq_ignore_ascii_case("null") {
        return Ok(Expression::literal(Scalar::Null(data_type.clone())));
    }

    let DataType::Primitive(primitive) = data_type else {
        return Err(Error::generic(format!(
            "SQL literal parsing only supports primitive types, got {data_type:?}"
        )));
    };

    // Binary uses a dedicated X'...' form. String is built directly from the
    // unquoted body. Both bypass parse_scalar, which treats an empty input as
    // SQL NULL (partition-value convention) -- a SQL empty string `''` must
    // round-trip as `Scalar::String("")`, distinct from `NULL`.
    match primitive {
        PrimitiveType::Binary => {
            let bytes = decode_binary_literal(trimmed)?;
            return Ok(Expression::literal(Scalar::Binary(bytes)));
        }
        PrimitiveType::String => {
            let unquoted = unquote_string(trimmed)?;
            return Ok(Expression::literal(Scalar::String(unquoted)));
        }
        _ => {}
    }

    // Strip the SQL syntax envelope per target type, then delegate to the
    // existing PrimitiveType::parse_scalar for the actual value parsing.
    let raw: Cow<'_, str> = match primitive {
        PrimitiveType::Date => Cow::Owned(strip_typed_prefix_and_unquote(trimmed, "DATE")?),
        PrimitiveType::Timestamp | PrimitiveType::TimestampNtz => {
            Cow::Owned(strip_typed_prefix_and_unquote(trimmed, "TIMESTAMP")?)
        }
        // Numeric and Boolean: parse_scalar handles signed numbers and
        // case-insensitive TRUE/FALSE directly. Reject any input that
        // looks like a quoted SQL string to avoid accidentally treating
        // `'42'` as the integer 42.
        _ => {
            if trimmed.starts_with('\'') {
                return Err(Error::generic(format!(
                    "expected a bare {primitive:?} literal, got quoted string: {sql}"
                )));
            }
            Cow::Borrowed(trimmed)
        }
    };

    let scalar = primitive.parse_scalar(&raw)?;
    Ok(Expression::literal(scalar))
}

/// Strip surrounding single quotes from a SQL string literal and un-escape
/// the doubled-quote sequence `''` -> `'`.
fn unquote_string(input: &str) -> DeltaResult<String> {
    let inner = input
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
        .ok_or_else(|| {
            Error::generic(format!("expected a single-quoted SQL string, got: {input}"))
        })?;

    // Reject literals containing an unescaped interior quote. A valid SQL
    // string represents `'` as `''`, so after splitting on `'` every chunk
    // except the last must be empty (meaning the quote was doubled).
    let mut out = String::with_capacity(inner.len());
    let mut chunks = inner.split('\'').peekable();
    while let Some(chunk) = chunks.next() {
        out.push_str(chunk);
        match chunks.peek() {
            None => break,
            Some(&"") => {
                // Doubled quote: consume the empty chunk and emit a single quote.
                chunks.next();
                out.push('\'');
            }
            Some(_) => {
                return Err(Error::generic(format!(
                    "unescaped single quote in SQL string literal: {input}"
                )));
            }
        }
    }
    Ok(out)
}

/// Strip an optional typed-literal keyword prefix (e.g. `DATE` or `TIMESTAMP`)
/// and then unwrap the required `'...'` quoted body.
fn strip_typed_prefix_and_unquote(input: &str, keyword: &str) -> DeltaResult<String> {
    let body = match input.split_once(char::is_whitespace) {
        Some((prefix, rest)) if prefix.eq_ignore_ascii_case(keyword) => rest.trim_start(),
        _ => input,
    };
    unquote_string(body)
}

/// Decode a `X'hex'` SQL binary literal into a byte vector. The leading `X`
/// is case-insensitive; the body must be an even-length sequence of hex
/// digits.
fn decode_binary_literal(input: &str) -> DeltaResult<Vec<u8>> {
    let err = || {
        Error::generic(format!(
            "expected a SQL binary literal like X'..', got: {input}"
        ))
    };
    let hex = input
        .strip_prefix(['x', 'X'])
        .and_then(|rest| rest.strip_prefix('\''))
        .and_then(|rest| rest.strip_suffix('\''))
        .ok_or_else(err)?;
    if !hex.len().is_multiple_of(2) {
        return Err(Error::generic(format!(
            "binary literal must contain an even number of hex digits: {input}"
        )));
    }
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let hi = (pair[0] as char)
                .to_digit(16)
                .ok_or_else(|| Error::generic(format!("invalid hex digit in {input}")))?;
            let lo = (pair[1] as char)
                .to_digit(16)
                .ok_or_else(|| Error::generic(format!("invalid hex digit in {input}")))?;
            Ok((hi << 4 | lo) as u8)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::expressions::Expression;
    use crate::schema::{ArrayType, DataType, StructField};

    fn date_days(year: i32, month: u32, day: u32) -> i32 {
        use chrono::{DateTime, NaiveDate, TimeZone, Utc};
        let nd = NaiveDate::from_ymd_opt(year, month, day)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        Utc.from_utc_datetime(&nd)
            .signed_duration_since(DateTime::UNIX_EPOCH)
            .num_days() as i32
    }

    fn ts_micros(s: &str) -> i64 {
        use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
        let ndt = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f").unwrap();
        Utc.from_utc_datetime(&ndt)
            .signed_duration_since(DateTime::UNIX_EPOCH)
            .num_microseconds()
            .unwrap()
    }

    #[rstest]
    #[case("42", DataType::INTEGER, Scalar::Integer(42))]
    #[case(" -7 ", DataType::INTEGER, Scalar::Integer(-7))]
    #[case("+5", DataType::INTEGER, Scalar::Integer(5))]
    #[case("127", DataType::BYTE, Scalar::Byte(127))]
    #[case("-32768", DataType::SHORT, Scalar::Short(i16::MIN))]
    #[case("9223372036854775807", DataType::LONG, Scalar::Long(i64::MAX))]
    #[case("2.5", DataType::DOUBLE, Scalar::Double(2.5))]
    #[case("0.5", DataType::FLOAT, Scalar::Float(0.5))]
    #[case("TRUE", DataType::BOOLEAN, Scalar::Boolean(true))]
    #[case("false", DataType::BOOLEAN, Scalar::Boolean(false))]
    #[case("'hello'", DataType::STRING, Scalar::String("hello".into()))]
    #[case("''", DataType::STRING, Scalar::String(String::new()))]
    #[case("'it''s'", DataType::STRING, Scalar::String("it's".into()))]
    #[case("'a''b''c'", DataType::STRING, Scalar::String("a'b'c".into()))]
    fn parses_basic_literals(#[case] sql: &str, #[case] ty: DataType, #[case] expected: Scalar) {
        let got = parse_sql_literal(sql, &ty).unwrap();
        assert_eq!(got, Expression::literal(expected));
    }

    #[rstest]
    #[case("'2024-01-01'", date_days(2024, 1, 1))]
    #[case("DATE '2024-01-01'", date_days(2024, 1, 1))]
    #[case("date  '1970-01-02'", date_days(1970, 1, 2))]
    fn parses_date_literals(#[case] sql: &str, #[case] expected_days: i32) {
        let got = parse_sql_literal(sql, &DataType::DATE).unwrap();
        assert_eq!(got, Expression::literal(Scalar::Date(expected_days)));
    }

    #[rstest]
    #[case("'2024-01-01 12:34:56'", "2024-01-01 12:34:56")]
    #[case("TIMESTAMP '2024-01-01 12:34:56.789'", "2024-01-01 12:34:56.789")]
    fn parses_timestamp_literals(#[case] sql: &str, #[case] equivalent: &str) {
        let got = parse_sql_literal(sql, &DataType::TIMESTAMP).unwrap();
        assert_eq!(
            got,
            Expression::literal(Scalar::Timestamp(ts_micros(equivalent)))
        );

        let got_ntz = parse_sql_literal(sql, &DataType::TIMESTAMP_NTZ).unwrap();
        assert_eq!(
            got_ntz,
            Expression::literal(Scalar::TimestampNtz(ts_micros(equivalent)))
        );
    }

    #[rstest]
    #[case("X''", vec![])]
    #[case("X'00'", vec![0x00])]
    #[case("X'DeAdBeEf'", vec![0xde, 0xad, 0xbe, 0xef])]
    #[case("x'01ff'", vec![0x01, 0xff])]
    fn parses_binary_literals(#[case] sql: &str, #[case] expected: Vec<u8>) {
        let got = parse_sql_literal(sql, &DataType::BINARY).unwrap();
        assert_eq!(got, Expression::literal(Scalar::Binary(expected)));
    }

    #[rstest]
    #[case(DataType::INTEGER)]
    #[case(DataType::STRING)]
    #[case(DataType::BOOLEAN)]
    #[case(DataType::DATE)]
    #[case(DataType::BINARY)]
    fn null_is_accepted_for_any_primitive(#[case] ty: DataType) {
        let got = parse_sql_literal("NULL", &ty).unwrap();
        assert_eq!(got, Expression::literal(Scalar::Null(ty.clone())));
        // also case-insensitive
        let got_lower = parse_sql_literal(" null ", &ty).unwrap();
        assert_eq!(got_lower, Expression::literal(Scalar::Null(ty)));
    }

    #[rstest]
    #[case("", DataType::INTEGER)]
    #[case("   ", DataType::INTEGER)]
    #[case("'42'", DataType::INTEGER)] // quoted number for int
    #[case("42", DataType::STRING)] // unquoted number for string
    #[case("foo", DataType::STRING)] // unquoted string
    #[case("'unterminated", DataType::STRING)]
    #[case("'bad'quote'", DataType::STRING)] // unescaped interior quote
    #[case("1 + 1", DataType::INTEGER)] // not a single literal
    #[case("nope", DataType::BOOLEAN)]
    #[case("'TRUE'", DataType::BOOLEAN)]
    #[case("'2024-13-01'", DataType::DATE)] // bad month
    #[case("not-a-date", DataType::DATE)]
    #[case("X'0'", DataType::BINARY)] // odd number of hex digits
    #[case("X'gg'", DataType::BINARY)] // non-hex chars
    #[case("'deadbeef'", DataType::BINARY)] // missing X prefix
    fn rejects_invalid_input(#[case] sql: &str, #[case] ty: DataType) {
        let result = parse_sql_literal(sql, &ty);
        assert!(
            result.is_err(),
            "expected error for {sql:?} as {ty:?}, got {result:?}"
        );
    }

    #[test]
    fn rejects_non_primitive_target() {
        let struct_ty =
            DataType::try_struct_type([StructField::nullable("a", DataType::INTEGER)]).unwrap();
        assert!(parse_sql_literal("'foo'", &struct_ty).is_err());

        let array_ty = DataType::Array(Box::new(ArrayType::new(DataType::INTEGER, true)));
        assert!(parse_sql_literal("'foo'", &array_ty).is_err());
    }

    #[test]
    fn null_is_accepted_for_non_primitive_target() {
        // NULL is special-cased and accepted before the primitive check,
        // matching the protocol's stance that any column can be null.
        let struct_ty =
            DataType::try_struct_type([StructField::nullable("a", DataType::INTEGER)]).unwrap();
        let got = parse_sql_literal("NULL", &struct_ty).unwrap();
        assert_eq!(got, Expression::literal(Scalar::Null(struct_ty)));
    }
}
