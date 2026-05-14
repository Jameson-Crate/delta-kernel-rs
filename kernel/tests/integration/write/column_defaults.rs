//! Integration tests demonstrating how a connector consumes
//! `Transaction::column_defaults()` (or `StructField::column_default()` on a
//! `Snapshot`) to fill in column default values during a write.
//!
//! The connector logic in these tests is *discovery-driven*: it never names a
//! specific column or default value. Instead it walks the schema and the
//! defaults map returned by the kernel and substitutes whatever it finds. The
//! tests still set up tables with specific columns (the assertions check the
//! end state), but the code that applies defaults treats everything as opaque
//! data discovered at runtime.
//!
//! Five tests are exercised:
//!   1. non-partition column missing from connector input
//!   2. non-partition column with `DEFAULT` sentinel (modeled as `None`)
//!   3. partition column missing from connector input
//!   4. partition column with `DEFAULT` sentinel
//!   5. discovery via `Snapshot::schema()` + `StructField::column_default()`, with data fully
//!      materialized before the transaction even begins

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::arrow::array::{ArrayRef, Int32Array, StringArray};
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::expressions::{ColumnName, Expression, Scalar};
use delta_kernel::schema::{
    ColumnDefault, ColumnMetadataKey, DataType, MetadataValue, SchemaRef, StructField, StructType,
};
use delta_kernel::{Engine, Snapshot};
use test_utils::{load_and_begin_transaction, setup_test_tables, test_read};

// ============================================================================
// Helpers -- none of these mention specific column names or default values.
// ============================================================================

/// Pull the literal `Scalar` out of a `ColumnDefault`. The kernel's literal
/// parser succeeds for every default used by these tests; a real connector
/// would fall back to its own SQL parser (via `col_default.sql`) when
/// `parsed` is `None`.
fn scalar_for(col_default: &ColumnDefault) -> Scalar {
    match col_default.parsed.as_ref() {
        Some(Expression::Literal(scalar)) => scalar.clone(),
        Some(other) => panic!("expected literal default, got {other:?}"),
        None => panic!("kernel could not parse default sql {:?}", col_default.sql),
    }
}

/// For a column whose cells are sentinel-aware (`None` = use the default),
/// substitute the default Scalar for each `None` and build the Arrow array.
/// Panics if a sentinel is encountered without a default.
fn resolve_sentinels(raw: &[Option<Scalar>], default: Option<&Scalar>) -> ArrayRef {
    let resolved: Vec<Scalar> = raw
        .iter()
        .map(|cell| {
            cell.clone()
                .or_else(|| default.cloned())
                .expect("sentinel cell but column has no default")
        })
        .collect();
    // All cells in a single column share a type; dispatch on the first.
    match resolved.first().expect("non-empty column") {
        Scalar::String(_) => Arc::new(StringArray::from(
            resolved
                .into_iter()
                .map(|s| match s {
                    Scalar::String(v) => v,
                    other => panic!("mixed scalar types in column: {other:?}"),
                })
                .collect::<Vec<_>>(),
        )),
        Scalar::Integer(_) => Arc::new(Int32Array::from(
            resolved
                .into_iter()
                .map(|s| match s {
                    Scalar::Integer(v) => v,
                    other => panic!("mixed scalar types in column: {other:?}"),
                })
                .collect::<Vec<_>>(),
        )),
        other => panic!("test helper does not handle {other:?}"),
    }
}

/// Project a connector's partial input batch to the full output schema by
/// running the kernel's `EvaluationHandler` over an `Expression::Struct` whose
/// children are either a column reference (for fields the connector supplied)
/// or a literal (for fields filled from the discovered default). The
/// evaluator broadcasts each literal to the input batch length, so the caller
/// never has to build a constant Arrow array by hand or dispatch on
/// `Scalar` variants.
fn project_with_defaults(
    engine: &dyn Engine,
    input: &ArrowEngineData,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    defaults: &HashMap<String, ColumnDefault>,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let exprs: Vec<Expression> = output_schema
        .fields()
        .map(|field| {
            if input_schema.field(&field.name).is_some() {
                Expression::from(ColumnName::new([field.name.as_str()]))
            } else if let Some(default) = defaults.get(&field.name) {
                Expression::literal(scalar_for(default))
            } else {
                panic!("column '{}' has no input and no default", field.name);
            }
        })
        .collect();

    let evaluator = engine.evaluation_handler().new_expression_evaluator(
        input_schema,
        Arc::new(Expression::struct_from(exprs)),
        output_schema.as_ref().clone().into(),
    )?;
    let projected = evaluator.evaluate(input)?;
    Ok(ArrowEngineData::try_from_engine_data(projected)?
        .record_batch()
        .clone())
}

/// Build the field `name: STRING` with a `CURRENT_DEFAULT` metadata entry.
fn string_field_with_default(name: &str, default_sql: &str) -> StructField {
    StructField::nullable(name, DataType::STRING).add_metadata([(
        ColumnMetadataKey::CurrentDefault.as_ref(),
        MetadataValue::String(default_sql.into()),
    )])
}

// ============================================================================
// Test 1: a non-partition column with a default is missing from the
// connector's input batch. The connector discovers which columns to fill from
// `txn.column_defaults()` -- it does not hardcode the column name or the
// default value.
// ============================================================================
#[tokio::test]
async fn default_filled_when_non_partition_column_missing_from_data(
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let table_schema = Arc::new(StructType::try_new(vec![
        StructField::nullable("id", DataType::INTEGER),
        string_field_with_default("status", "'active'"),
    ])?);

    for (table_url, engine, _store, _name) in
        setup_test_tables(table_schema.clone(), &[], None, "default_missing_col").await?
    {
        let mut txn = load_and_begin_transaction(table_url.clone(), &engine)?
            .with_engine_info("default-test");

        let id_values: Vec<i32> = vec![1, 2, 3];

        // Connector input: just the id column. Wrap as EngineData; its schema
        // is whatever the connector has, NOT the full table schema.
        let input_schema: SchemaRef = Arc::new(StructType::try_new(vec![StructField::nullable(
            "id",
            DataType::INTEGER,
        )])?);
        let input = ArrowEngineData::new(RecordBatch::try_new(
            Arc::new(input_schema.as_ref().try_into_arrow()?),
            vec![Arc::new(Int32Array::from(id_values.clone()))],
        )?);

        // Discovery + projection via the kernel evaluator: walk the output
        // schema, pass through input columns, broadcast literals for the rest.
        let defaults = txn.column_defaults();
        let batch = project_with_defaults(
            &engine,
            &input,
            input_schema,
            table_schema.clone(),
            &defaults,
        )?;

        let engine = Arc::new(engine);
        let write_context = Arc::new(txn.unpartitioned_write_context()?);
        let add = engine
            .write_parquet(&ArrowEngineData::new(batch), write_context.as_ref())
            .await?;
        txn.add_files(add);
        assert!(txn.commit(engine.as_ref())?.is_committed());

        let expected = RecordBatch::try_new(
            Arc::new(table_schema.as_ref().try_into_arrow()?),
            vec![
                Arc::new(Int32Array::from(id_values)),
                Arc::new(StringArray::from(vec!["active", "active", "active"])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }
    Ok(())
}

// ============================================================================
// Test 2: every cell in the connector's input is `Option<Scalar>`; `None`
// means "use the default for this column." The substitution loop iterates
// columns without ever naming one.
// ============================================================================
#[tokio::test]
async fn default_substituted_for_sentinel_in_non_partition_column(
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let table_schema = Arc::new(StructType::try_new(vec![
        StructField::nullable("id", DataType::INTEGER),
        string_field_with_default("status", "'active'"),
    ])?);

    for (table_url, engine, _store, _name) in
        setup_test_tables(table_schema.clone(), &[], None, "default_sentinel_col").await?
    {
        let mut txn = load_and_begin_transaction(table_url.clone(), &engine)?
            .with_engine_info("default-test");

        // Connector input is a column-name -> sentinel-aware vector of cells.
        // Some cells are None, meaning "use the default for this column."
        let raw: HashMap<String, Vec<Option<Scalar>>> = HashMap::from([
            (
                "id".to_string(),
                vec![
                    Some(Scalar::Integer(10)),
                    Some(Scalar::Integer(20)),
                    Some(Scalar::Integer(30)),
                ],
            ),
            (
                "status".to_string(),
                vec![None, Some(Scalar::String("custom".into())), None],
            ),
        ]);

        // Discovery + per-column sentinel substitution. Iterate the schema in
        // declaration order; for each field, build its array from the raw
        // sentinel vector, substituting the discovered default into Nones.
        let defaults = txn.column_defaults();
        let columns: Vec<ArrayRef> = table_schema
            .fields()
            .map(|field| {
                let raw_cells = raw
                    .get(&field.name)
                    .expect("connector input has every column");
                let default_scalar = defaults.get(&field.name).map(scalar_for);
                resolve_sentinels(raw_cells, default_scalar.as_ref())
            })
            .collect();

        let batch =
            RecordBatch::try_new(Arc::new(table_schema.as_ref().try_into_arrow()?), columns)?;

        let engine = Arc::new(engine);
        let write_context = Arc::new(txn.unpartitioned_write_context()?);
        let add = engine
            .write_parquet(&ArrowEngineData::new(batch), write_context.as_ref())
            .await?;
        txn.add_files(add);
        assert!(txn.commit(engine.as_ref())?.is_committed());

        let expected = RecordBatch::try_new(
            Arc::new(table_schema.as_ref().try_into_arrow()?),
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30])),
                Arc::new(StringArray::from(vec!["active", "custom", "active"])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }
    Ok(())
}

// ============================================================================
// Test 3: a partition column has a default but the connector did not specify
// a partition value. The connector discovers the partition columns and the
// defaults, then constructs the partition_values map from the intersection.
// ============================================================================
#[tokio::test]
async fn default_used_as_partition_value_when_partition_column_missing(
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let table_schema = Arc::new(StructType::try_new(vec![
        StructField::nullable("id", DataType::INTEGER),
        string_field_with_default("region", "'us-west'"),
    ])?);
    // Partition columns are not present in the parquet payload schema.
    let data_schema = Arc::new(StructType::try_new(vec![StructField::nullable(
        "id",
        DataType::INTEGER,
    )])?);

    for (table_url, engine, _store, _name) in setup_test_tables(
        table_schema.clone(),
        &["region"],
        None,
        "default_missing_part",
    )
    .await?
    {
        let mut txn = load_and_begin_transaction(table_url.clone(), &engine)?
            .with_engine_info("default-test");

        // Discover both: the partition columns the table requires and the
        // defaults available. Where they overlap and the connector didn't
        // supply a value, fall back to the default.
        let partition_columns: Vec<String> = txn.logical_partition_columns().to_vec();
        let defaults = txn.column_defaults();

        let supplied: HashMap<String, Scalar> = HashMap::new(); // connector supplied nothing
        let partition_values: HashMap<String, Scalar> = partition_columns
            .iter()
            .filter_map(|name| {
                supplied
                    .get(name)
                    .cloned()
                    .or_else(|| defaults.get(name).map(scalar_for))
                    .map(|v| (name.clone(), v))
            })
            .collect();

        let id_values: Vec<i32> = vec![1, 2, 3];
        let batch = RecordBatch::try_new(
            Arc::new(data_schema.as_ref().try_into_arrow()?),
            vec![Arc::new(Int32Array::from(id_values.clone()))],
        )?;

        let engine = Arc::new(engine);
        let write_context = Arc::new(txn.partitioned_write_context(partition_values)?);
        let add = engine
            .write_parquet(&ArrowEngineData::new(batch), write_context.as_ref())
            .await?;
        txn.add_files(add);
        assert!(txn.commit(engine.as_ref())?.is_committed());

        let expected = RecordBatch::try_new(
            Arc::new(table_schema.as_ref().try_into_arrow()?),
            vec![
                Arc::new(Int32Array::from(id_values)),
                Arc::new(StringArray::from(vec!["us-west", "us-west", "us-west"])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }
    Ok(())
}

// ============================================================================
// Test 4: the planner produces an `Option<Scalar>` for each partition column
// per batch; `None` is the sentinel. The resolution loop iterates partition
// columns and discovered defaults without naming either.
// ============================================================================
#[tokio::test]
async fn default_used_as_partition_value_for_sentinel() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let table_schema = Arc::new(StructType::try_new(vec![
        StructField::nullable("id", DataType::INTEGER),
        string_field_with_default("region", "'us-west'"),
    ])?);
    let data_schema = Arc::new(StructType::try_new(vec![StructField::nullable(
        "id",
        DataType::INTEGER,
    )])?);

    for (table_url, engine, _store, _name) in setup_test_tables(
        table_schema.clone(),
        &["region"],
        None,
        "default_sentinel_part",
    )
    .await?
    {
        let mut txn = load_and_begin_transaction(table_url.clone(), &engine)?
            .with_engine_info("default-test");

        let partition_columns: Vec<String> = txn.logical_partition_columns().to_vec();
        let defaults = txn.column_defaults();

        // Planning: per batch, a sentinel-aware partition_values map. None
        // means "fall back to whatever default this column has."
        type PartitionPlan = Vec<(HashMap<String, Option<Scalar>>, Vec<i32>)>;
        let planned: PartitionPlan = vec![
            (
                HashMap::from([("region".to_string(), Some(Scalar::String("eu".into())))]),
                vec![10, 20],
            ),
            (
                HashMap::from([("region".to_string(), None)]),
                vec![30, 40, 50],
            ),
        ];

        let engine = Arc::new(engine);
        for (sentinel_map, ids) in planned {
            let partition_values: HashMap<String, Scalar> = partition_columns
                .iter()
                .map(|name| {
                    let resolved = sentinel_map
                        .get(name)
                        .and_then(|opt| opt.clone())
                        .or_else(|| defaults.get(name).map(scalar_for))
                        .unwrap_or_else(|| {
                            panic!("partition column '{name}' has no value and no default")
                        });
                    (name.clone(), resolved)
                })
                .collect();

            let batch = RecordBatch::try_new(
                Arc::new(data_schema.as_ref().try_into_arrow()?),
                vec![Arc::new(Int32Array::from(ids))],
            )?;
            let write_context = Arc::new(txn.partitioned_write_context(partition_values)?);
            let add = engine
                .write_parquet(&ArrowEngineData::new(batch), write_context.as_ref())
                .await?;
            txn.add_files(add);
        }
        assert!(txn.commit(engine.as_ref())?.is_committed());

        let expected = RecordBatch::try_new(
            Arc::new(table_schema.as_ref().try_into_arrow()?),
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50])),
                Arc::new(StringArray::from(vec![
                    "eu", "eu", "us-west", "us-west", "us-west",
                ])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }
    Ok(())
}

// ============================================================================
// Test 5: discovery happens via `Snapshot::schema()` + `StructField::column_default()`,
// and the data batch is fully materialized BEFORE any transaction is
// started. This mirrors a planner-side architecture where the planning layer
// resolves defaults using only a snapshot reference, then hands a complete
// batch to the writer.
// ============================================================================
#[tokio::test]
async fn defaults_discovered_via_snapshot_schema_and_materialized_before_transaction(
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    // Multiple defaulted columns so the discovery loop has real work to do.
    let table_schema = Arc::new(StructType::try_new(vec![
        StructField::nullable("id", DataType::INTEGER),
        string_field_with_default("status", "'active'"),
        string_field_with_default("region", "'us-west'"),
    ])?);

    for (table_url, engine, _store, _name) in
        setup_test_tables(table_schema.clone(), &[], None, "snapshot_defaults").await?
    {
        // Phase 1: planner-side. Load the snapshot, walk its schema, build a
        // map of (column name -> default) using StructField::column_default()
        // directly, then materialize the complete batch. No transaction yet.
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let snapshot_schema = snapshot.schema();

        let defaults_from_schema: HashMap<String, ColumnDefault> = snapshot_schema
            .fields()
            .filter_map(|f| f.column_default().map(|d| (f.name.clone(), d)))
            .collect();

        let id_values: Vec<i32> = vec![100, 200, 300];
        let input_schema: SchemaRef = Arc::new(StructType::try_new(vec![StructField::nullable(
            "id",
            DataType::INTEGER,
        )])?);
        let input = ArrowEngineData::new(RecordBatch::try_new(
            Arc::new(input_schema.as_ref().try_into_arrow()?),
            vec![Arc::new(Int32Array::from(id_values.clone()))],
        )?);

        let prepared = ArrowEngineData::new(project_with_defaults(
            &engine,
            &input,
            input_schema,
            snapshot_schema.clone(),
            &defaults_from_schema,
        )?);

        // Phase 2: now begin the transaction with the same snapshot, write
        // the pre-materialized batch, commit. No further default lookup.
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?
            .with_engine_info("default-test");
        let engine = Arc::new(engine);
        let write_context = Arc::new(txn.unpartitioned_write_context()?);
        let add = engine
            .write_parquet(&prepared, write_context.as_ref())
            .await?;
        txn.add_files(add);
        assert!(txn.commit(engine.as_ref())?.is_committed());

        let expected = RecordBatch::try_new(
            Arc::new(table_schema.as_ref().try_into_arrow()?),
            vec![
                Arc::new(Int32Array::from(id_values)),
                Arc::new(StringArray::from(vec!["active", "active", "active"])),
                Arc::new(StringArray::from(vec!["us-west", "us-west", "us-west"])),
            ],
        )?;
        test_read(&ArrowEngineData::new(expected), &table_url, engine)?;
    }
    Ok(())
}
