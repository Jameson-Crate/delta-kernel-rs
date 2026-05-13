//! End-to-end demonstrations of how a connector consumes
//! [`Transaction::column_defaults`](delta_kernel::transaction::Transaction::column_defaults)
//! during a write.
//!
//! The connector in every test treats the table's schema as opaque. It does not hardcode column
//! names, orders, or which columns have defaults; it discovers everything at runtime from
//! [`Transaction`] / [`WriteContext`] accessors:
//!
//! - [`WriteContext::logical_schema`] -- the target output schema (drives both the column set and
//!   the column *order* in the output struct expression).
//! - [`Transaction::column_defaults`] -- map keyed by logical column name.
//! - [`Transaction::logical_partition_columns`] -- partition column names (for the partitioned
//!   tests).
//!
//! Several tests deliberately submit input columns in a different order than the table schema
//! to show that the [`Expression::struct_from`] approach is order-independent: the kernel
//! evaluator resolves [`Expression::column`] references by name, so input order is irrelevant.
//!
//! Four scenarios are exercised, two unpartitioned and two partitioned:
//!
//! 1. [`test_connector_fills_missing_column_with_default`] -- unpartitioned. Generic schema-walking
//!    helper builds an output struct whose children are either column refs (for fields present in
//!    the input) or parsed defaults (for fields absent from the input).
//! 2. [`test_connector_uses_bitmask_to_substitute_defaults_per_row`] -- unpartitioned. The
//!    connector discovers bitmask-default pairings by iterating [`Transaction::column_defaults`]
//!    and looking for `<name>_use_default` bitmask columns in the input. Per-row blending happens
//!    at the Arrow level (kernel has no `If` / `Case` expression).
//! 3. [`test_connector_fills_defaulted_non_partition_column_in_partitioned_table`] -- mirror of
//!    (1), with a partition column. Same generic helper.
//! 4. [`test_connector_uses_default_as_partition_value`] -- the connector discovers the partition
//!    column(s) via [`Transaction::logical_partition_columns`], looks each one up in
//!    [`Transaction::column_defaults`], and uses the parsed scalar both as the partition value
//!    passed to [`Transaction::partitioned_write_context`] and as the literal placed in the
//!    partition position of the output struct. The default drives the on-disk partition layout.
//!
//! [`Expression::column`]: delta_kernel::expressions::Expression::column
//! [`Expression::struct_from`]: delta_kernel::expressions::Expression::struct_from
//! [`Transaction::partitioned_write_context`]:
//!     delta_kernel::transaction::Transaction::partitioned_write_context
//! [`Transaction::logical_partition_columns`]:
//!     delta_kernel::transaction::Transaction::logical_partition_columns
//! [`WriteContext::logical_schema`]: delta_kernel::transaction::WriteContext::logical_schema

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use delta_kernel::arrow::array::{ArrayRef, BooleanArray, Int32Array, StringArray};
use delta_kernel::arrow::datatypes::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::{TryFromArrow as _, TryIntoArrow as _};
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::expressions::{Expression, Scalar};
use delta_kernel::schema::{DataType, MetadataValue, SchemaRef, StructField, StructType};
use delta_kernel::transaction::{DefaultColumn, Transaction, WriteContext};
use delta_kernel::{DeltaResult, Engine, Error, Snapshot};
use test_utils::{setup_test_tables, test_read};
use url::Url;

const COLUMN_DEFAULT_KEY: &str = "CURRENT_DEFAULT";
/// Bitmask naming convention used by Test B: a boolean column named `<X>_use_default`
/// means "use the default for column `X` on rows where this is true."
const USE_DEFAULT_SUFFIX: &str = "_use_default";

/// Shared scaffolding: open a transaction on `table_url`, let the caller produce the
/// [`WriteContext`] *and* the [`ArrowEngineData`] (since partitioned tests may need to read
/// `column_defaults` from the transaction in order to compute the partition values), then
/// run `write_parquet` + `add_files` + `commit`.
async fn write_via_connector<F>(
    table_url: &Url,
    engine: Arc<DefaultEngine<TokioBackgroundExecutor>>,
    plan_write: F,
) -> DeltaResult<()>
where
    F: FnOnce(&Transaction, &dyn Engine) -> DeltaResult<(WriteContext, ArrowEngineData)>,
{
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_data_change(true);

    let (write_context, output) = plan_write(&txn, engine.as_ref())?;
    let add_metadata = engine.write_parquet(&output, &write_context).await?;
    txn.add_files(add_metadata);

    let _committed = txn.commit(engine.as_ref())?.unwrap_committed();
    Ok(())
}

/// Generic helper that builds the per-column expression list a connector would use to produce
/// a batch matching `logical_schema`. For each target field, in target-schema order:
///
/// - if the input batch supplies that field by name, emit `Expression::column([name])` (the
///   evaluator resolves this by name, so input column order is irrelevant);
/// - otherwise, if the field has a parsed default in `defaults`, emit the parsed expression;
/// - otherwise, return an error (the input is missing a column with no fallback).
///
/// Returns an `Arc<Expression>` of `struct_from(children)` ready to feed to
/// `EvaluationHandler::new_expression_evaluator` whose output type is `logical_schema`.
fn build_struct_from_logical_schema(
    logical_schema: &StructType,
    input_field_names: &HashSet<String>,
    defaults: &HashMap<String, DefaultColumn>,
) -> DeltaResult<Arc<Expression>> {
    let children: Vec<Arc<Expression>> = logical_schema
        .fields()
        .map(|field| -> DeltaResult<Arc<Expression>> {
            if input_field_names.contains(&field.name) {
                Ok(Arc::new(Expression::column([field.name.clone()])))
            } else if let Some(d) = defaults.get(&field.name) {
                let parsed = d.parsed_expr.as_ref().ok_or_else(|| {
                    Error::generic(format!(
                        "column `{}` default `{}` could not be parsed by the engine's \
                         ParsingHandler",
                        field.name, d.sql
                    ))
                })?;
                Ok(Arc::new(parsed.clone()))
            } else {
                Err(Error::generic(format!(
                    "column `{}` is absent from the input batch and has no CURRENT_DEFAULT",
                    field.name
                )))
            }
        })
        .collect::<DeltaResult<_>>()?;
    Ok(Arc::new(Expression::struct_from(children)))
}

/// Collect input field names into a set for fast `contains` lookups inside the helper.
fn input_field_names(batch: &RecordBatch) -> HashSet<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

// =============================================================================
// Test A: Connector fills an entire missing column with its default value.
// =============================================================================

/// Table schema with one defaulted column. The default literal `'unnamed'` is stored under
/// the protocol's `CURRENT_DEFAULT` field-metadata key.
fn schema_id_and_defaulted_name() -> SchemaRef {
    Arc::new(StructType::new_unchecked(vec![
        StructField::nullable("id", DataType::INTEGER),
        StructField::nullable("name", DataType::STRING).with_metadata([(
            COLUMN_DEFAULT_KEY,
            MetadataValue::String("'unnamed'".into()),
        )]),
    ]))
}

#[tokio::test]
async fn test_connector_fills_missing_column_with_default() -> Result<(), Box<dyn std::error::Error>>
{
    let _ = tracing_subscriber::fmt::try_init();

    let table_schema = schema_id_and_defaulted_name();

    // The connector's input has one column. Its name happens to be `id`, but the connector
    // logic below never references that name directly -- it iterates the schema instead.
    let input_arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        ArrowDataType::Int32,
        true,
    )]));
    let input_batch = RecordBatch::try_new(
        input_arrow_schema,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
    )?;

    for (table_url, engine, _store, _name) in
        setup_test_tables(table_schema.clone(), &[], None, "default_fill").await?
    {
        let engine = Arc::new(engine);
        let input_batch = input_batch.clone();
        let table_schema = table_schema.clone();

        write_via_connector(&table_url, engine.clone(), move |txn, engine| {
            let write_ctx = txn.unpartitioned_write_context()?;
            // === Connector logic begins ===
            // Discover everything from the transaction / write context. The connector never
            // names a specific column.
            let logical = write_ctx.logical_schema();
            let defaults = txn.column_defaults();
            let input_names = input_field_names(&input_batch);
            let input_schema = Arc::new(StructType::try_from_arrow(input_batch.schema().as_ref())?);

            let output_expr = build_struct_from_logical_schema(logical, &input_names, defaults)?;
            let evaluator = engine.evaluation_handler().new_expression_evaluator(
                input_schema,
                output_expr,
                logical.as_ref().clone().into(),
            )?;
            let output = evaluator.evaluate(&ArrowEngineData::new(input_batch.clone()))?;
            // === Connector logic ends ===
            Ok((write_ctx, *ArrowEngineData::try_from_engine_data(output)?))
        })
        .await?;

        let expected_arrow_schema: ArrowSchema = table_schema.as_ref().try_into_arrow()?;
        let expected = RecordBatch::try_new(
            Arc::new(expected_arrow_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    "unnamed", "unnamed", "unnamed", "unnamed",
                ])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }

    Ok(())
}

// =============================================================================
// Test B: Connector applies the default to only the rows flagged by a bitmask.
// =============================================================================
//
// The connector treats `column_defaults` as the authority on which columns can be defaulted,
// and uses an `<X>_use_default` naming convention on the input batch to discover bitmask
// columns. For each defaulted column the input also provides a bitmask for, the connector
// substitutes the default scalar on flagged rows -- at the Arrow level, since kernel has no
// `Expression::If`. The output is built in logical-schema order, so input column order is
// irrelevant; bitmask columns are dropped automatically by virtue of not being in
// `logical_schema`.

/// Table schema with an int column carrying a literal-integer default of `99`.
fn schema_id_and_defaulted_val() -> SchemaRef {
    Arc::new(StructType::new_unchecked(vec![
        StructField::nullable("id", DataType::INTEGER),
        StructField::nullable("val", DataType::INTEGER)
            .with_metadata([(COLUMN_DEFAULT_KEY, MetadataValue::String("99".into()))]),
    ]))
}

#[tokio::test]
async fn test_connector_uses_bitmask_to_substitute_defaults_per_row(
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let table_schema = schema_id_and_defaulted_val();

    // Deliberately scramble the input column order to demonstrate that the connector logic
    // below doesn't depend on it -- input is `[use_default-bitmask, val, id]` while the table
    // schema is `[id, val]`. The bitmask is named by convention (`val_use_default`).
    let input_arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("val_use_default", ArrowDataType::Boolean, true),
        ArrowField::new("val", ArrowDataType::Int32, true),
        ArrowField::new("id", ArrowDataType::Int32, true),
    ]));
    let input_batch = RecordBatch::try_new(
        input_arrow_schema,
        vec![
            Arc::new(BooleanArray::from(vec![false, true, false, true])),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
        ],
    )?;

    for (table_url, engine, _store, _name) in
        setup_test_tables(table_schema.clone(), &[], None, "default_bitmask").await?
    {
        let engine = Arc::new(engine);
        let input_batch = input_batch.clone();
        let table_schema = table_schema.clone();

        write_via_connector(&table_url, engine.clone(), move |txn, _engine| {
            let write_ctx = txn.unpartitioned_write_context()?;
            // === Connector logic begins ===
            // 1. For each defaulted column, see whether the input batch supplied a bitmask column
            //    named `<col>_use_default`. If so, build a blended column at the Arrow level and
            //    stash it as the replacement for `<col>`. The connector doesn't know any specific
            //    column name -- it iterates `column_defaults` and the input schema.
            let defaults = txn.column_defaults();
            let mut replaced_columns: HashMap<String, ArrayRef> = HashMap::new();
            for (col_name, default) in defaults {
                let bitmask_name = format!("{col_name}{USE_DEFAULT_SUFFIX}");
                let (Some(input_col), Some(bitmask_col)) = (
                    input_batch.column_by_name(col_name),
                    input_batch.column_by_name(&bitmask_name),
                ) else {
                    continue;
                };
                let parsed = default.parsed_expr.as_ref().ok_or_else(|| {
                    Error::generic(format!("default for `{col_name}` did not parse"))
                })?;
                let Expression::Literal(scalar) = parsed else {
                    return Err(Error::generic(format!(
                        "expected literal default for `{col_name}`, got {parsed:?}"
                    )));
                };
                replaced_columns.insert(
                    col_name.clone(),
                    blend_with_default(input_col, bitmask_col, scalar)?,
                );
            }

            // 2. Build the output RecordBatch in logical-schema order, picking either a replaced
            //    (blended) column or the original input column. Columns not in the logical schema
            //    (e.g. bitmasks) are naturally excluded.
            let logical = write_ctx.logical_schema();
            let output_arrow_schema: ArrowSchema = logical.as_ref().try_into_arrow()?;
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(logical.fields().count());
            for field in logical.fields() {
                let array = if let Some(replaced) = replaced_columns.remove(&field.name) {
                    replaced
                } else if let Some(input_col) = input_batch.column_by_name(&field.name) {
                    input_col.clone()
                } else {
                    return Err(Error::generic(format!(
                        "column `{}` is missing from input and has no bitmask substitution",
                        field.name
                    )));
                };
                columns.push(array);
            }
            let output = RecordBatch::try_new(Arc::new(output_arrow_schema), columns)?;
            // === Connector logic ends ===
            Ok((write_ctx, ArrowEngineData::new(output)))
        })
        .await?;

        let expected_arrow_schema: ArrowSchema = table_schema.as_ref().try_into_arrow()?;
        let expected = RecordBatch::try_new(
            Arc::new(expected_arrow_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
                Arc::new(Int32Array::from(vec![10, 99, 30, 99])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }

    Ok(())
}

/// Blend `input_col` with the default `scalar` on rows where `bitmask_col` is true. Returns a
/// new Arrow array of the same type as `input_col`. This is the per-row substitution kernel
/// can't express today (no `Expression::If` / `Case` variant).
fn blend_with_default(
    input_col: &ArrayRef,
    bitmask_col: &ArrayRef,
    scalar: &Scalar,
) -> DeltaResult<ArrayRef> {
    let bitmask = bitmask_col
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| Error::generic("bitmask column is not boolean"))?;

    match scalar {
        Scalar::Integer(default_value) => {
            let input = input_col
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::generic("input column type mismatch (expected int32)"))?;
            let blended =
                Int32Array::from_iter(bitmask.iter().zip(input.iter()).map(|(use_default, v)| {
                    match use_default {
                        Some(true) => Some(*default_value),
                        _ => v,
                    }
                }));
            Ok(Arc::new(blended))
        }
        Scalar::String(default_value) => {
            let input = input_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::generic("input column type mismatch (expected string)"))?;
            let blended: StringArray = bitmask
                .iter()
                .zip(input.iter())
                .map(|(use_default, v)| match use_default {
                    Some(true) => Some(default_value.as_str()),
                    _ => v,
                })
                .collect();
            Ok(Arc::new(blended))
        }
        other => Err(Error::generic(format!(
            "blend_with_default does not yet handle scalar type {other:?}"
        ))),
    }
}

// =============================================================================
// Test C: Partitioned table; default is on a non-partition column.
// =============================================================================
//
// Same generic schema-walking helper as Test A. The connector supplies the partition value
// explicitly because it has a domain reason to (it's writing a specific partition); column
// defaults play no role in deciding that. Input column order is deliberately scrambled
// relative to the table schema to show the helper handles it.

/// `region STRING (partition), id INT, name STRING DEFAULT 'unnamed'`, partitioned by `region`.
fn schema_partitioned_with_defaulted_data_col() -> SchemaRef {
    Arc::new(StructType::new_unchecked(vec![
        StructField::nullable("region", DataType::STRING),
        StructField::nullable("id", DataType::INTEGER),
        StructField::nullable("name", DataType::STRING).with_metadata([(
            COLUMN_DEFAULT_KEY,
            MetadataValue::String("'unnamed'".into()),
        )]),
    ]))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_connector_fills_defaulted_non_partition_column_in_partitioned_table(
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let table_schema = schema_partitioned_with_defaulted_data_col();

    // Input column order [id, region] vs. table schema [region, id, name]. The helper handles
    // this because the kernel evaluator resolves column refs by name.
    let input_arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, true),
        ArrowField::new("region", ArrowDataType::Utf8, true),
    ]));
    let input_batch = RecordBatch::try_new(
        input_arrow_schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["east", "east", "east"])),
        ],
    )?;

    for (table_url, engine, _store, _name) in setup_test_tables(
        table_schema.clone(),
        &["region"],
        None,
        "default_partitioned",
    )
    .await?
    {
        let engine = Arc::new(engine);
        let input_batch = input_batch.clone();
        let table_schema = table_schema.clone();

        write_via_connector(&table_url, engine.clone(), move |txn, engine| {
            // === Connector logic begins ===
            // 1. The connector decides which partition to write (domain choice). It happens to pull
            //    the partition column name out of the transaction rather than hardcoding it -- it
            //    does need to know that "east" is a valid value, but not the column.
            let part_cols = txn.logical_partition_columns();
            assert_eq!(part_cols.len(), 1, "this test assumes single partition col");
            let part_name = part_cols[0].clone();
            let write_ctx = txn.partitioned_write_context(HashMap::from([(
                part_name,
                Scalar::String("east".into()),
            )]))?;

            // 2. Build the output struct via the generic helper -- no column names hardcoded.
            let logical = write_ctx.logical_schema();
            let defaults = txn.column_defaults();
            let input_names = input_field_names(&input_batch);
            let input_schema = Arc::new(StructType::try_from_arrow(input_batch.schema().as_ref())?);
            let output_expr = build_struct_from_logical_schema(logical, &input_names, defaults)?;

            let evaluator = engine.evaluation_handler().new_expression_evaluator(
                input_schema,
                output_expr,
                logical.as_ref().clone().into(),
            )?;
            let output = evaluator.evaluate(&ArrowEngineData::new(input_batch.clone()))?;
            // === Connector logic ends ===
            Ok((write_ctx, *ArrowEngineData::try_from_engine_data(output)?))
        })
        .await?;

        let expected_arrow_schema: ArrowSchema = table_schema.as_ref().try_into_arrow()?;
        let expected = RecordBatch::try_new(
            Arc::new(expected_arrow_schema),
            vec![
                Arc::new(StringArray::from(vec!["east", "east", "east"])),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["unnamed", "unnamed", "unnamed"])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }

    Ok(())
}

// =============================================================================
// Test D: Partitioned table; default *is* on the partition column.
// =============================================================================
//
// Most novel use case: the connector receives data without a partition value at all and uses
// the table's declared defaults to drive both the on-disk directory layout and the in-batch
// column. The connector discovers which columns are partition columns via
// `txn.logical_partition_columns()`, looks each one up in `column_defaults`, and pulls the
// parsed scalar out. No column name is hardcoded.

/// `region STRING DEFAULT 'unknown' (partition), id INT`, partitioned by `region`.
fn schema_partitioned_with_defaulted_partition_col() -> SchemaRef {
    Arc::new(StructType::new_unchecked(vec![
        StructField::nullable("region", DataType::STRING).with_metadata([(
            COLUMN_DEFAULT_KEY,
            MetadataValue::String("'unknown'".into()),
        )]),
        StructField::nullable("id", DataType::INTEGER),
    ]))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_connector_uses_default_as_partition_value() -> Result<(), Box<dyn std::error::Error>>
{
    let _ = tracing_subscriber::fmt::try_init();

    let table_schema = schema_partitioned_with_defaulted_partition_col();

    // Connector input has whatever it has -- here, only one column. It doesn't know which of
    // the table's columns are partition columns or which have defaults.
    let input_arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        ArrowDataType::Int32,
        true,
    )]));
    let input_batch = RecordBatch::try_new(
        input_arrow_schema,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )?;

    for (table_url, engine, _store, _name) in setup_test_tables(
        table_schema.clone(),
        &["region"],
        None,
        "default_drives_partition",
    )
    .await?
    {
        let engine = Arc::new(engine);
        let input_batch = input_batch.clone();
        let table_schema = table_schema.clone();

        write_via_connector(&table_url, engine.clone(), move |txn, engine| {
            // === Connector logic begins ===
            // 1. Discover partition columns from the transaction. For every partition column that's
            //    missing from the input, pull its parsed default scalar and use it as the partition
            //    value. No column name is hardcoded.
            let defaults = txn.column_defaults();
            let input_names = input_field_names(&input_batch);
            let mut partition_values: HashMap<String, Scalar> = HashMap::new();
            for part_col in txn.logical_partition_columns() {
                if input_names.contains(part_col) {
                    continue; // value comes from the batch itself; nothing to compute
                }
                let default = defaults.get(part_col).ok_or_else(|| {
                    Error::generic(format!(
                        "partition column `{part_col}` is absent from input and has no default"
                    ))
                })?;
                let parsed = default.parsed_expr.as_ref().ok_or_else(|| {
                    Error::generic(format!(
                        "partition default for `{part_col}` did not parse to a literal"
                    ))
                })?;
                let Expression::Literal(scalar) = parsed else {
                    return Err(Error::generic(format!(
                        "expected literal partition default for `{part_col}`, got {parsed:?}"
                    )));
                };
                partition_values.insert(part_col.clone(), scalar.clone());
            }

            // 2. The same scalars determine both the on-disk path and the partition position in the
            //    batch (the generic helper picks the parsed default for absent fields).
            let write_ctx = txn.partitioned_write_context(partition_values)?;
            let logical = write_ctx.logical_schema();
            let input_schema = Arc::new(StructType::try_from_arrow(input_batch.schema().as_ref())?);
            let output_expr = build_struct_from_logical_schema(logical, &input_names, defaults)?;
            let evaluator = engine.evaluation_handler().new_expression_evaluator(
                input_schema,
                output_expr,
                logical.as_ref().clone().into(),
            )?;
            let output = evaluator.evaluate(&ArrowEngineData::new(input_batch.clone()))?;
            // === Connector logic ends ===
            Ok((write_ctx, *ArrowEngineData::try_from_engine_data(output)?))
        })
        .await?;

        let expected_arrow_schema: ArrowSchema = table_schema.as_ref().try_into_arrow()?;
        let expected = RecordBatch::try_new(
            Arc::new(expected_arrow_schema),
            vec![
                Arc::new(StringArray::from(vec!["unknown", "unknown", "unknown"])),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }

    Ok(())
}
