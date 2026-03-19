use std::path::PathBuf;
use std::sync::Arc;

use delta_kernel::actions::deletion_vector::split_vector;
use delta_kernel::arrow::array::{AsArray as _, RecordBatch};
use delta_kernel::arrow::compute::{concat_batches, filter_record_batch};
use delta_kernel::arrow::datatypes::{Field as ArrowField, Int64Type, Schema as ArrowSchema};
use delta_kernel::engine::arrow_conversion::TryFromKernel as _;
use delta_kernel::engine::arrow_data::EngineDataArrowExt as _;
use delta_kernel::engine::default::DefaultEngineBuilder;
use delta_kernel::expressions::{
    column_expr, column_pred, Expression as Expr, ExpressionRef, Predicate as Pred,
};
use delta_kernel::log_segment::LogSegment;
use delta_kernel::parquet::file::properties::{EnabledStatistics, WriterProperties};
use delta_kernel::path::ParsedLogPath;
use delta_kernel::scan::state::{transform_to_logical, ScanFile};
use delta_kernel::scan::Scan;
use delta_kernel::schema::{DataType, MetadataColumnSpec, Schema, StructField, StructType};
use delta_kernel::{Engine, FileMeta, Snapshot};

use itertools::Itertools;
use object_store::{memory::InMemory, path::Path, ObjectStore};
use test_utils::{
    actions_to_string, add_commit, assert_batches_eq, generate_batch, generate_simple_batch,
    into_record_batch, load_test_data, make_top_level_fields_nullable, read_scan,
    record_batch_to_bytes, record_batch_to_bytes_with_props, IntoArray, TestAction, METADATA,
};
use url::Url;

mod common;

const PARQUET_FILE1: &str = "part-00000-a72b1fb3-f2df-41fe-a8f0-e65b746382dd-c000.snappy.parquet";
const PARQUET_FILE2: &str = "part-00001-c506e79a-0bf8-4e2b-a42b-9731b2e490ae-c000.snappy.parquet";
const PARQUET_FILE3: &str = "part-00002-c506e79a-0bf8-4e2b-a42b-9731b2e490ff-c000.snappy.parquet";

#[tokio::test]
async fn single_commit_two_add_files() -> Result<(), Box<dyn std::error::Error>> {
    let batch = generate_simple_batch()?;
    let storage = Arc::new(InMemory::new());
    let parquet_bytes = record_batch_to_bytes(&batch);
    let file_size = parquet_bytes.len() as u64;
    add_commit(
        storage.as_ref(),
        0,
        actions_to_string(vec![
            TestAction::Metadata,
            TestAction::AddWithSize(PARQUET_FILE1.to_string(), file_size),
            TestAction::AddWithSize(PARQUET_FILE2.to_string(), file_size),
        ]),
    )
    .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE1),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE2),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;

    let location = Url::parse("memory:///")?;
    let engine = Arc::new(DefaultEngineBuilder::new(storage.clone()).build());

    let expected = make_top_level_fields_nullable(&batch);
    let expected_data = vec![expected.clone(), expected];

    let snapshot = Snapshot::builder_for(location).build(engine.as_ref())?;
    let scan = snapshot.scan_builder().build()?;

    let mut files = 0;
    let stream = scan.execute(engine)?.zip(expected_data);

    for (data, expected) in stream {
        files += 1;
        assert_eq!(into_record_batch(data?), expected);
    }
    assert_eq!(2, files, "Expected to have scanned two files");
    Ok(())
}

#[tokio::test]
async fn two_commits() -> Result<(), Box<dyn std::error::Error>> {
    let batch = generate_simple_batch()?;
    let storage = Arc::new(InMemory::new());
    let parquet_bytes = record_batch_to_bytes(&batch);
    let file_size = parquet_bytes.len() as u64;
    add_commit(
        storage.as_ref(),
        0,
        actions_to_string(vec![
            TestAction::Metadata,
            TestAction::AddWithSize(PARQUET_FILE1.to_string(), file_size),
        ]),
    )
    .await?;
    add_commit(
        storage.as_ref(),
        1,
        actions_to_string(vec![TestAction::AddWithSize(
            PARQUET_FILE2.to_string(),
            file_size,
        )]),
    )
    .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE1),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE2),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;

    let location = Url::parse("memory:///").unwrap();
    let engine = DefaultEngineBuilder::new(storage.clone()).build();

    let expected = make_top_level_fields_nullable(&batch);
    let expected_data = vec![expected.clone(), expected];

    let snapshot = Snapshot::builder_for(location).build(&engine)?;
    let scan = snapshot.scan_builder().build()?;

    let mut files = 0;
    let stream = scan.execute(Arc::new(engine))?.zip(expected_data);

    for (data, expected) in stream {
        files += 1;
        assert_eq!(into_record_batch(data?), expected);
    }
    assert_eq!(2, files, "Expected to have scanned two files");

    Ok(())
}

#[tokio::test]
async fn remove_action() -> Result<(), Box<dyn std::error::Error>> {
    let batch = generate_simple_batch()?;
    let storage = Arc::new(InMemory::new());
    let parquet_bytes = record_batch_to_bytes(&batch);
    let file_size = parquet_bytes.len() as u64;
    add_commit(
        storage.as_ref(),
        0,
        actions_to_string(vec![
            TestAction::Metadata,
            TestAction::AddWithSize(PARQUET_FILE1.to_string(), file_size),
        ]),
    )
    .await?;
    add_commit(
        storage.as_ref(),
        1,
        actions_to_string(vec![TestAction::AddWithSize(
            PARQUET_FILE2.to_string(),
            file_size,
        )]),
    )
    .await?;
    add_commit(
        storage.as_ref(),
        2,
        actions_to_string(vec![TestAction::RemoveWithSize(
            PARQUET_FILE2.to_string(),
            file_size,
        )]),
    )
    .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE1),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;

    let location = Url::parse("memory:///").unwrap();
    let engine = DefaultEngineBuilder::new(storage.clone()).build();

    let expected = make_top_level_fields_nullable(&batch);
    let expected_data = vec![expected];

    let snapshot = Snapshot::builder_for(location).build(&engine)?;
    let scan = snapshot.scan_builder().build()?;

    let stream = scan.execute(Arc::new(engine))?.zip(expected_data);

    let mut files = 0;
    for (data, expected) in stream {
        files += 1;
        assert_eq!(into_record_batch(data?), expected);
    }
    assert_eq!(1, files, "Expected to have scanned one file");
    Ok(())
}

#[tokio::test]
async fn stats() -> Result<(), Box<dyn std::error::Error>> {
    fn generate_commit2(actions: Vec<TestAction>) -> String {
        actions
            .into_iter()
            .map(|test_action| match test_action {
                TestAction::Add(path) => format!(r#"{{"{action}":{{"path":"{path}","partitionValues":{{}},"size":262,"modificationTime":1587968586000,"dataChange":true, "stats":"{{\"numRecords\":2,\"nullCount\":{{\"id\":0}},\"minValues\":{{\"id\": 5}},\"maxValues\":{{\"id\":7}}}}"}}}}"#, action = "add", path = path),
                TestAction::Remove(path) => format!(r#"{{"{action}":{{"path":"{path}","partitionValues":{{}},"size":262,"modificationTime":1587968586000,"dataChange":true}}}}"#, action = "remove", path = path),
                TestAction::Metadata => METADATA.into(),
                TestAction::AddWithSize(path, size) => format!(r#"{{"{action}":{{"path":"{path}","partitionValues":{{}},"size":{size},"modificationTime":1587968586000,"dataChange":true, "stats":"{{\"numRecords\":2,\"nullCount\":{{\"id\":0}},\"minValues\":{{\"id\": 5}},\"maxValues\":{{\"id\":7}}}}"}}}}"#, action = "add", path = path),
                TestAction::RemoveWithSize(path, size) => format!(r#"{{"{action}":{{"path":"{path}","partitionValues":{{}},"size":{size},"modificationTime":1587968586000,"dataChange":true}}}}"#, action = "remove", path = path),
            })
            .fold(String::new(), |a, b| a + &b + "\n")
    }

    let batch1 = make_top_level_fields_nullable(&generate_simple_batch()?);
    let batch2 = make_top_level_fields_nullable(&generate_batch(vec![
        ("id", vec![5, 7].into_array()),
        ("val", vec!["e", "g"].into_array()),
    ])?);
    let file_size1 = record_batch_to_bytes(&batch1).len() as u64;
    let file_size2 = record_batch_to_bytes(&batch2).len() as u64;
    let storage = Arc::new(InMemory::new());
    // valid commit with min/max (0, 2)
    add_commit(
        storage.as_ref(),
        0,
        actions_to_string(vec![
            TestAction::Metadata,
            TestAction::AddWithSize(PARQUET_FILE1.to_string(), file_size1),
        ]),
    )
    .await?;
    // storage.add_commit(1, &format!("{}\n", r#"{{"add":{{"path":"doesnotexist","partitionValues":{{}},"size":262,"modificationTime":1587968586000,"dataChange":true, "stats":"{{\"numRecords\":2,\"nullCount\":{{\"id\":0}},\"minValues\":{{\"id\": 0}},\"maxValues\":{{\"id\":2}}}}"}}}}"#));
    add_commit(
        storage.as_ref(),
        1,
        generate_commit2(vec![TestAction::AddWithSize(
            PARQUET_FILE2.to_string(),
            file_size2,
        )]),
    )
    .await?;

    storage
        .put(
            &Path::from(PARQUET_FILE1),
            record_batch_to_bytes(&batch1).into(),
        )
        .await?;

    storage
        .put(
            &Path::from(PARQUET_FILE2),
            record_batch_to_bytes(&batch2).into(),
        )
        .await?;

    let location = Url::parse("memory:///").unwrap();
    let engine = Arc::new(DefaultEngineBuilder::new(storage.clone()).build());
    let snapshot = Snapshot::builder_for(location).build(engine.as_ref())?;

    // The first file has id between 1 and 3; the second has id between 5 and 7. For each operator,
    // we validate the boundary values where we expect the set of matched files to change.
    //
    // NOTE: For cases that match both batch1 and batch2, we list batch2 first because log replay
    // returns most recently added files first.
    #[allow(clippy::type_complexity)] // otherwise it's even more complex because no `_`
    let test_cases: Vec<(fn(Expr, Expr) -> _, _, _)> = vec![
        (Pred::eq, 0i32, vec![]),
        (Pred::eq, 1, vec![&batch1]),
        (Pred::eq, 3, vec![&batch1]),
        (Pred::eq, 4, vec![]),
        (Pred::eq, 5, vec![&batch2]),
        (Pred::eq, 7, vec![&batch2]),
        (Pred::eq, 8, vec![]),
        (Pred::lt, 1, vec![]),
        (Pred::lt, 2, vec![&batch1]),
        (Pred::lt, 5, vec![&batch1]),
        (Pred::lt, 6, vec![&batch2, &batch1]),
        (Pred::le, 0, vec![]),
        (Pred::le, 1, vec![&batch1]),
        (Pred::le, 4, vec![&batch1]),
        (Pred::le, 5, vec![&batch2, &batch1]),
        (Pred::gt, 2, vec![&batch2, &batch1]),
        (Pred::gt, 3, vec![&batch2]),
        (Pred::gt, 6, vec![&batch2]),
        (Pred::gt, 7, vec![]),
        (Pred::ge, 3, vec![&batch2, &batch1]),
        (Pred::ge, 4, vec![&batch2]),
        (Pred::ge, 7, vec![&batch2]),
        (Pred::ge, 8, vec![]),
        (Pred::ne, 0, vec![&batch2, &batch1]),
        (Pred::ne, 1, vec![&batch2, &batch1]),
        (Pred::ne, 3, vec![&batch2, &batch1]),
        (Pred::ne, 4, vec![&batch2, &batch1]),
        (Pred::ne, 5, vec![&batch2, &batch1]),
        (Pred::ne, 7, vec![&batch2, &batch1]),
        (Pred::ne, 8, vec![&batch2, &batch1]),
    ];
    for (pred_fn, value, expected_batches) in test_cases {
        let predicate = pred_fn(column_expr!("id"), Expr::literal(value));
        let scan = snapshot
            .clone()
            .scan_builder()
            .with_predicate(Arc::new(predicate.clone()))
            .build()?;

        let expected_files = expected_batches.len();
        let mut files_scanned = 0;
        let stream = scan.execute(engine.clone())?.zip(expected_batches);

        for (batch, expected) in stream {
            files_scanned += 1;
            assert_eq!(into_record_batch(batch?), expected.clone());
        }
        assert_eq!(expected_files, files_scanned, "{predicate:?}");
    }
    Ok(())
}

/// Read a scan using `Scan::execute` and compare results against an expected `RecordBatch`.
fn read_with_execute(
    engine: Arc<dyn Engine>,
    scan: &Scan,
    expected: &RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let batches = read_scan(scan, engine)?;
    assert_batches_eq(expected, &batches);
    Ok(())
}

fn scan_metadata_callback(batches: &mut Vec<ScanFile>, scan_file: ScanFile) {
    batches.push(scan_file);
}

/// Read a scan using `Scan::scan_metadata` and compare results against an expected `RecordBatch`.
fn read_with_scan_metadata(
    location: &Url,
    engine: &dyn Engine,
    scan: &Scan,
    expected: &RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let result_schema = Arc::new(ArrowSchema::try_from_kernel(
        scan.logical_schema().as_ref(),
    )?);
    let scan_metadata = scan.scan_metadata(engine)?;
    let mut scan_files = vec![];
    for res in scan_metadata {
        let scan_metadata = res?;
        scan_files = scan_metadata.visit_scan_files(scan_files, scan_metadata_callback)?;
    }

    let mut batches = vec![];
    for scan_file in scan_files.into_iter() {
        let file_path = location.join(&scan_file.path)?;
        let mut selection_vector = scan_file
            .dv_info
            .get_selection_vector(engine, location)
            .unwrap();
        let meta = FileMeta {
            last_modified: 0,
            size: scan_file.size.try_into().unwrap(),
            location: file_path,
        };
        let read_results = engine
            .parquet_handler()
            .read_parquet_files(
                &[meta],
                scan.physical_schema().clone(),
                scan.physical_predicate().clone(),
            )
            .unwrap();

        for read_result in read_results {
            let read_result = read_result.unwrap();
            let len = read_result.len();
            // to transform the physical data into the correct logical form
            let logical = transform_to_logical(
                engine,
                read_result,
                scan.physical_schema(),
                scan.logical_schema(),
                scan_file.transform.clone(),
            )
            .unwrap();
            let record_batch = logical.try_into_record_batch()?;
            let rest = split_vector(selection_vector.as_mut(), len, Some(true));
            let batch = if let Some(mask) = selection_vector.clone() {
                // apply the selection vector
                filter_record_batch(&record_batch, &mask.into()).unwrap()
            } else {
                record_batch
            };
            selection_vector = rest;
            batches.push(batch);
        }
    }

    if expected.num_rows() == 0 {
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0, "Expected 0 rows but got {total_rows}");
    } else {
        let actual = concat_batches(&result_schema, &batches)?;
        assert_batches_eq(expected, &[actual]);
    }
    Ok(())
}

/// Read a Delta table and compare results (via both execute and scan_metadata paths) against
/// an expected `RecordBatch`.
fn read_table_data(
    path: &str,
    select_cols: Option<&[&str]>,
    predicate: Option<Pred>,
    expected: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = std::fs::canonicalize(PathBuf::from(path))?;
    let predicate = predicate.map(Arc::new);
    let url = url::Url::from_directory_path(path).unwrap();
    let engine = test_utils::create_default_engine(&url)?;

    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;

    let read_schema = select_cols.map(|select_cols| {
        let table_schema = snapshot.schema();
        let selected_fields = select_cols
            .iter()
            .map(|col| table_schema.field(col).cloned().unwrap());
        Arc::new(Schema::new_unchecked(selected_fields))
    });
    println!("Read {url:?} with schema {read_schema:#?} and predicate {predicate:#?}");
    let scan = snapshot
        .scan_builder()
        .with_schema_opt(read_schema)
        .with_predicate(predicate.clone())
        .build()?;

    read_with_scan_metadata(&url, engine.as_ref(), &scan, &expected)?;
    read_with_execute(engine, &scan, &expected)?;
    Ok(())
}

/// Build a `RecordBatch` for the `basic_partitioned` table filtered to the given number values,
/// with columns `(a_float: f64, number: i64)`.
fn table_for_numbers(nums: Vec<u32>) -> RecordBatch {
    let floats: Vec<f64> = nums.iter().map(|n| *n as f64 + (*n as f64 / 10.0)).collect();
    let ints: Vec<i64> = nums.iter().map(|n| *n as i64).collect();
    generate_batch(vec![
        ("a_float", floats.into_array()),
        ("number", ints.into_array()),
    ])
    .unwrap()
}

/// Build a `RecordBatch` for the `basic_partitioned` table filtered to the given letter values,
/// with columns `(letter: Utf8, number: i64)`.
fn table_for_letters(letters: &[char]) -> RecordBatch {
    let rows: Vec<(i64, &'static str)> =
        vec![(1, "a"), (2, "b"), (3, "c"), (4, "a"), (5, "e")];
    let mut letter_vals: Vec<&'static str> = vec![];
    let mut number_vals: Vec<i64> = vec![];
    for (num, letter) in rows {
        if letters.contains(&letter.chars().next().unwrap()) {
            letter_vals.push(letter);
            number_vals.push(num);
        }
    }
    generate_batch(vec![
        ("letter", letter_vals.into_array()),
        ("number", number_vals.into_array()),
    ])
    .unwrap()
}

#[test]
fn data() -> Result<(), Box<dyn std::error::Error>> {
    let expected = generate_batch(vec![
        (
            "letter",
            vec![
                None,
                Some("a"),
                Some("a"),
                Some("b"),
                Some("c"),
                Some("e"),
            ]
            .into_array(),
        ),
        ("number", vec![6i64, 1, 4, 2, 3, 5].into_array()),
        (
            "a_float",
            vec![6.6, 1.1, 4.4, 2.2, 3.3, 5.5].into_array(),
        ),
    ])?;
    read_table_data("./tests/data/basic_partitioned", None, None, expected)?;

    Ok(())
}

#[test]
fn column_ordering() -> Result<(), Box<dyn std::error::Error>> {
    let expected = generate_batch(vec![
        (
            "a_float",
            vec![6.6, 4.4, 5.5, 1.1, 2.2, 3.3].into_array(),
        ),
        (
            "letter",
            vec![
                None,
                Some("a"),
                Some("e"),
                Some("a"),
                Some("b"),
                Some("c"),
            ]
            .into_array(),
        ),
        ("number", vec![6i64, 4, 5, 1, 2, 3].into_array()),
    ])?;
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "letter", "number"]),
        None,
        expected,
    )?;

    Ok(())
}

#[test]
fn column_ordering_and_projection() -> Result<(), Box<dyn std::error::Error>> {
    let expected = generate_batch(vec![
        (
            "a_float",
            vec![6.6, 4.4, 5.5, 1.1, 2.2, 3.3].into_array(),
        ),
        ("number", vec![6i64, 4, 5, 1, 2, 3].into_array()),
    ])?;
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "number"]),
        None,
        expected,
    )?;

    Ok(())
}

#[rstest::rstest]
#[case::less_than(
    column_expr!("number").lt(Expr::literal(4i64)),
    table_for_numbers(vec![1, 2, 3])
)]
#[case::less_than_or_equal(
    column_expr!("number").le(Expr::literal(4i64)),
    table_for_numbers(vec![1, 2, 3, 4])
)]
#[case::greater_than(
    column_expr!("number").gt(Expr::literal(4i64)),
    table_for_numbers(vec![5, 6])
)]
#[case::greater_than_or_equal(
    column_expr!("number").ge(Expr::literal(4i64)),
    table_for_numbers(vec![4, 5, 6])
)]
#[case::equal(
    column_expr!("number").eq(Expr::literal(4i64)),
    table_for_numbers(vec![4])
)]
#[case::not_equal(
    column_expr!("number").ne(Expr::literal(4i64)),
    table_for_numbers(vec![1, 2, 3, 5, 6])
)]
fn predicate_on_number(
    #[case] pred: Pred,
    #[case] expected: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "number"]),
        Some(pred),
        expected,
    )?;
    Ok(())
}

#[rstest::rstest]
#[case::is_null(
    column_expr!("letter").is_null(),
    generate_batch(vec![
        ("letter", vec![None::<&str>].into_array()),
        ("number", vec![6i64].into_array()),
    ]).unwrap()
)]
#[case::is_not_null(
    column_expr!("letter").is_not_null(),
    table_for_letters(&['a', 'b', 'c', 'e'])
)]
#[case::less_than(
    column_expr!("letter").lt(Expr::literal("c")),
    table_for_letters(&['a', 'b'])
)]
#[case::less_than_or_equal(
    column_expr!("letter").le(Expr::literal("c")),
    table_for_letters(&['a', 'b', 'c'])
)]
#[case::greater_than(
    column_expr!("letter").gt(Expr::literal("c")),
    table_for_letters(&['e'])
)]
#[case::greater_than_or_equal(
    column_expr!("letter").ge(Expr::literal("c")),
    table_for_letters(&['c', 'e'])
)]
#[case::equal(
    column_expr!("letter").eq(Expr::literal("c")),
    table_for_letters(&['c'])
)]
#[case::not_equal(
    column_expr!("letter").ne(Expr::literal("c")),
    table_for_letters(&['a', 'b', 'e'])
)]
fn predicate_on_letter(
    #[case] pred: Pred,
    #[case] expected: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test basic column pruning. Note that the actual predicate machinery is already well-tested,
    // so we're just testing wiring here.
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["letter", "number"]),
        Some(pred),
        expected,
    )?;
    Ok(())
}

#[rstest::rstest]
#[case::or_no_pruning(
    Pred::or(
        // No pruning power
        column_expr!("letter").gt(Expr::literal("a")),
        column_expr!("number").gt(Expr::literal(3i64)),
    ),
    generate_batch(vec![
        ("letter", vec![None, Some("a"), Some("a"), Some("b"), Some("c"), Some("e")].into_array()),
        ("number", vec![6i64, 1, 4, 2, 3, 5].into_array()),
    ]).unwrap()
)]
#[case::and_with_pruning(
    Pred::and(
        column_expr!("letter").gt(Expr::literal("a")), // numbers 2, 3, 5
        column_expr!("number").gt(Expr::literal(3i64)), // letters a, e
    ),
    table_for_letters(&['e'])
)]
#[case::and_with_nested_or(
    Pred::and(
        column_expr!("letter").gt(Expr::literal("a")), // numbers 2, 3, 5
        Pred::or(
            // No pruning power
            column_expr!("letter").eq(Expr::literal("c")),
            column_expr!("number").eq(Expr::literal(3i64)),
        ),
    ),
    table_for_letters(&['b', 'c', 'e'])
)]
fn predicate_on_letter_and_number(
    #[case] pred: Pred,
    #[case] expected: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    // Partition skipping and file skipping are currently implemented separately. Mixing them in an
    // AND clause will evaulate each separately, but mixing them in an OR clause disables both.
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["letter", "number"]),
        Some(pred),
        expected,
    )?;
    Ok(())
}

#[rstest::rstest]
#[case::not_less_than(
    Pred::not(column_expr!("number").lt(Expr::literal(4i64))),
    table_for_numbers(vec![4, 5, 6])
)]
#[case::not_less_than_or_equal(
    Pred::not(column_expr!("number").le(Expr::literal(4i64))),
    table_for_numbers(vec![5, 6])
)]
#[case::not_greater_than(
    Pred::not(column_expr!("number").gt(Expr::literal(4i64))),
    table_for_numbers(vec![1, 2, 3, 4])
)]
#[case::not_greater_than_or_equal(
    Pred::not(column_expr!("number").ge(Expr::literal(4i64))),
    table_for_numbers(vec![1, 2, 3])
)]
#[case::not_equal(
    Pred::not(column_expr!("number").eq(Expr::literal(4i64))),
    table_for_numbers(vec![1, 2, 3, 5, 6])
)]
#[case::not_not_equal(
    Pred::not(column_expr!("number").ne(Expr::literal(4i64))),
    table_for_numbers(vec![4])
)]
fn predicate_on_number_not(
    #[case] pred: Pred,
    #[case] expected: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "number"]),
        Some(pred),
        expected,
    )?;
    Ok(())
}

#[test]
fn predicate_on_number_with_not_null() -> Result<(), Box<dyn std::error::Error>> {
    let expected = generate_batch(vec![
        ("a_float", vec![1.1, 2.2].into_array()),
        ("number", vec![1i64, 2].into_array()),
    ])?;
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "number"]),
        Some(Pred::and(
            column_expr!("number").is_not_null(),
            column_expr!("number").lt(Expr::literal(3i64)),
        )),
        expected,
    )?;
    Ok(())
}

#[test]
fn predicate_null() -> Result<(), Box<dyn std::error::Error>> {
    // number is never null, so the result should be empty
    let expected = RecordBatch::new_empty(Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a_float", delta_kernel::arrow::datatypes::DataType::Float64, false),
        ArrowField::new("number", delta_kernel::arrow::datatypes::DataType::Int64, false),
    ])));
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "number"]),
        Some(column_expr!("number").is_null()),
        expected,
    )?;
    Ok(())
}

#[test]
fn mixed_null() -> Result<(), Box<dyn std::error::Error>> {
    let expected = generate_batch(vec![
        (
            "part",
            vec![0i64, 0, 0, 0, 0, 2, 2, 2, 2, 2].into_array(),
        ),
        (
            "n",
            vec![
                None,
                None,
                None,
                None,
                None,
                None,
                Some("non-null-mix"),
                None,
                Some("non-null-mix"),
                None,
            ]
            .into_array(),
        ),
    ])?;
    read_table_data(
        "./tests/data/mixed-nulls",
        Some(&["part", "n"]),
        Some(column_expr!("n").is_null()),
        expected,
    )?;
    Ok(())
}

#[test]
fn mixed_not_null() -> Result<(), Box<dyn std::error::Error>> {
    let expected = generate_batch(vec![
        (
            "part",
            vec![1i64, 1, 1, 1, 1, 2, 2, 2, 2, 2].into_array(),
        ),
        (
            "n",
            vec![
                Some("non-null"),
                Some("non-null"),
                Some("non-null"),
                Some("non-null"),
                Some("non-null"),
                None,
                None,
                None,
                Some("non-null-mix"),
                Some("non-null-mix"),
            ]
            .into_array(),
        ),
    ])?;
    read_table_data(
        "./tests/data/mixed-nulls",
        Some(&["part", "n"]),
        Some(column_expr!("n").is_not_null()),
        expected,
    )?;
    Ok(())
}

#[rstest::rstest]
#[case::and_both_conditions(
    Pred::and(
        column_expr!("number").gt(Expr::literal(4i64)),
        column_expr!("a_float").gt(Expr::literal(5.5)),
    ),
    table_for_numbers(vec![6])
)]
#[case::and_with_negation(
    Pred::and(
        column_expr!("number").gt(Expr::literal(4i64)),
        Pred::not(column_expr!("a_float").gt(Expr::literal(5.5))),
    ),
    table_for_numbers(vec![5])
)]
#[case::or_either_condition(
    Pred::or(
        column_expr!("number").gt(Expr::literal(4i64)),
        column_expr!("a_float").gt(Expr::literal(5.5)),
    ),
    table_for_numbers(vec![5, 6])
)]
#[case::or_with_negation(
    Pred::or(
        column_expr!("number").gt(Expr::literal(4i64)),
        Pred::not(column_expr!("a_float").gt(Expr::literal(5.5))),
    ),
    table_for_numbers(vec![1, 2, 3, 4, 5, 6])
)]
fn and_or_predicates(
    #[case] pred: Pred,
    #[case] expected: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "number"]),
        Some(pred),
        expected,
    )?;
    Ok(())
}

#[rstest::rstest]
#[case::not_and_both_conditions(
    Pred::not(Pred::and(
        column_expr!("number").gt(Expr::literal(4i64)),
        column_expr!("a_float").gt(Expr::literal(5.5)),
    )),
    table_for_numbers(vec![1, 2, 3, 4, 5])
)]
#[case::not_and_with_negation(
    Pred::not(Pred::and(
        column_expr!("number").gt(Expr::literal(4i64)),
        Pred::not(column_expr!("a_float").gt(Expr::literal(5.5))),
    )),
    table_for_numbers(vec![1, 2, 3, 4, 6])
)]
#[case::not_or_either_condition(
    Pred::not(Pred::or(
        column_expr!("number").gt(Expr::literal(4i64)),
        column_expr!("a_float").gt(Expr::literal(5.5)),
    )),
    table_for_numbers(vec![1, 2, 3, 4])
)]
#[case::not_or_with_negation(
    Pred::not(Pred::or(
        column_expr!("number").gt(Expr::literal(4i64)),
        Pred::not(column_expr!("a_float").gt(Expr::literal(5.5))),
    )),
    RecordBatch::new_empty(Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a_float", delta_kernel::arrow::datatypes::DataType::Float64, false),
        ArrowField::new("number", delta_kernel::arrow::datatypes::DataType::Int64, false),
    ])))
)]
fn not_and_or_predicates(
    #[case] pred: Pred,
    #[case] expected: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "number"]),
        Some(pred),
        expected,
    )?;
    Ok(())
}

#[rstest::rstest]
#[case::literal_false(Pred::literal(false), table_for_numbers(vec![]))]
#[case::and_with_literal_false(
    Pred::and(column_pred!("number"), Pred::literal(false)),
    table_for_numbers(vec![])
)]
#[case::literal_true(
    Pred::literal(true),
    table_for_numbers(vec![1, 2, 3, 4, 5, 6])
)]
#[case::from_literal_expr(
    Pred::from_expr(Expr::literal(3i64)),
    table_for_numbers(vec![1, 2, 3, 4, 5, 6])
)]
#[case::distinct_value(
    column_expr!("number").distinct(Expr::literal(3i64)),
    table_for_numbers(vec![1, 2, 4, 5, 6])
)]
#[case::distinct_null(
    column_expr!("number").distinct(Expr::null_literal(DataType::LONG)),
    table_for_numbers(vec![1, 2, 3, 4, 5, 6])
)]
#[case::not_distinct_value(
    Pred::not(column_expr!("number").distinct(Expr::literal(3i64))),
    table_for_numbers(vec![3])
)]
#[case::not_distinct_null(
    Pred::not(column_expr!("number").distinct(Expr::null_literal(DataType::LONG))),
    table_for_numbers(vec![])
)]
#[case::gt_empty_struct(
    column_expr!("number").gt(Expr::struct_from(Vec::<ExpressionRef>::new())),
    table_for_numbers(vec![1, 2, 3, 4, 5, 6])
)]
#[case::not_gt_empty_struct(
    Pred::not(column_expr!("number").gt(Expr::struct_from(Vec::<ExpressionRef>::new()))),
    table_for_numbers(vec![1, 2, 3, 4, 5, 6])
)]
fn invalid_skips_none_predicates(
    #[case] pred: Pred,
    #[case] expected: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    read_table_data(
        "./tests/data/basic_partitioned",
        Some(&["a_float", "number"]),
        Some(pred),
        expected,
    )?;
    Ok(())
}

#[test]
fn with_predicate_and_removes() -> Result<(), Box<dyn std::error::Error>> {
    let expected = generate_batch(vec![(
        "value",
        vec![1i64, 2, 3, 4, 5, 6, 7, 8].into_array(),
    )])?;
    read_table_data(
        "./tests/data/table-with-dv-small/",
        None,
        Some(Pred::gt(column_expr!("value"), Expr::literal(3))),
        expected,
    )?;
    Ok(())
}

#[tokio::test]
async fn predicate_on_non_nullable_partition_column() -> Result<(), Box<dyn std::error::Error>> {
    // Test for https://github.com/delta-io/delta-kernel-rs/issues/698
    let batch = generate_batch(vec![("val", vec!["a", "b", "c"].into_array())])?;

    let storage = Arc::new(InMemory::new());
    let actions = [
        r#"{"protocol":{"minReaderVersion":1,"minWriterVersion":2}}"#.to_string(),
        r#"{"commitInfo":{"timestamp":1587968586154,"operation":"WRITE","operationParameters":{"mode":"ErrorIfExists","partitionBy":"[\"id\"]"},"isBlindAppend":true}}"#.to_string(),
        r#"{"metaData":{"id":"5fba94ed-9794-4965-ba6e-6ee3c0d22af9","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"id\",\"type\":\"integer\",\"nullable\":false,\"metadata\":{}},{\"name\":\"val\",\"type\":\"string\",\"nullable\":false,\"metadata\":{}}]}","partitionColumns":["id"],"configuration":{},"createdTime":1587968585495}}"#.to_string(),
        format!(r#"{{"add":{{"path":"id=1/{PARQUET_FILE1}","partitionValues":{{"id":"1"}},"size":0,"modificationTime":1587968586000,"dataChange":true, "stats":"{{\"numRecords\":3,\"nullCount\":{{\"val\":0}},\"minValues\":{{\"val\":\"a\"}},\"maxValues\":{{\"val\":\"c\"}}}}"}}}}"#),
        format!(r#"{{"add":{{"path":"id=2/{PARQUET_FILE2}","partitionValues":{{"id":"2"}},"size":0,"modificationTime":1587968586000,"dataChange":true, "stats":"{{\"numRecords\":3,\"nullCount\":{{\"val\":0}},\"minValues\":{{\"val\":\"a\"}},\"maxValues\":{{\"val\":\"c\"}}}}"}}}}"#),
    ];

    add_commit(storage.as_ref(), 0, actions.iter().join("\n")).await?;
    storage
        .put(
            &Path::from("id=1").child(PARQUET_FILE1),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;
    storage
        .put(
            &Path::from("id=2").child(PARQUET_FILE2),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;

    let location = Url::parse("memory:///")?;

    let engine = Arc::new(DefaultEngineBuilder::new(storage.clone()).build());
    let snapshot = Snapshot::builder_for(location).build(engine.as_ref())?;

    let predicate = Pred::eq(column_expr!("id"), Expr::literal(2));
    let scan = snapshot
        .scan_builder()
        .with_predicate(Arc::new(predicate))
        .build()?;

    let stream = scan.execute(engine)?;

    let mut files_scanned = 0;
    for engine_data in stream {
        let mut result_batch = into_record_batch(engine_data?);
        let _ = result_batch.remove_column(result_batch.schema().index_of("id")?);
        assert_eq!(&batch, &result_batch);
        files_scanned += 1;
    }
    assert_eq!(1, files_scanned);
    Ok(())
}

#[tokio::test]
async fn predicate_on_non_nullable_column_missing_stats() -> Result<(), Box<dyn std::error::Error>>
{
    let batch_1 = generate_batch(vec![("val", vec!["a", "b", "c"].into_array())])?;
    let batch_2 = generate_batch(vec![("val", vec!["d", "e", "f"].into_array())])?;

    let storage = Arc::new(InMemory::new());
    let actions = [
        r#"{"protocol":{"minReaderVersion":1,"minWriterVersion":2}}"#.to_string(),
        r#"{"commitInfo":{"timestamp":1587968586154,"operation":"WRITE","operationParameters":{"mode":"ErrorIfExists","partitionBy":"[]"},"isBlindAppend":true}}"#.to_string(),
        r#"{"metaData":{"id":"5fba94ed-9794-4965-ba6e-6ee3c0d22af9","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"val\",\"type\":\"string\",\"nullable\":false,\"metadata\":{}}]}","partitionColumns":[],"configuration":{},"createdTime":1587968585495}}"#.to_string(),
        // Add one file with stats, one file without
        format!(r#"{{"add":{{"path":"{PARQUET_FILE1}","partitionValues":{{}},"size":0,"modificationTime":1587968586000,"dataChange":true, "stats":"{{\"numRecords\":3,\"nullCount\":{{\"val\":0}},\"minValues\":{{\"val\":\"a\"}},\"maxValues\":{{\"val\":\"c\"}}}}"}}}}"#),
        format!(r#"{{"add":{{"path":"{PARQUET_FILE2}","partitionValues":{{}},"size":0,"modificationTime":1587968586000,"dataChange":true, "stats":"{{\"numRecords\":3,\"nullCount\":{{}},\"minValues\":{{}},\"maxValues\":{{}}}}"}}}}"#),
    ];

    // Disable writing Parquet statistics so these cannot be used for pruning row groups
    let writer_props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::None)
        .build();

    add_commit(storage.as_ref(), 0, actions.iter().join("\n")).await?;
    storage
        .put(
            &Path::from(PARQUET_FILE1),
            record_batch_to_bytes_with_props(&batch_1, writer_props.clone()).into(),
        )
        .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE2),
            record_batch_to_bytes_with_props(&batch_2, writer_props).into(),
        )
        .await?;

    let location = Url::parse("memory:///")?;

    let engine = Arc::new(DefaultEngineBuilder::new(storage.clone()).build());
    let snapshot = Snapshot::builder_for(location).build(engine.as_ref())?;

    let predicate = Pred::eq(column_expr!("val"), Expr::literal("g"));
    let scan = snapshot
        .scan_builder()
        .with_predicate(Arc::new(predicate))
        .build()?;

    let stream = scan.execute(engine)?;

    let mut files_scanned = 0;
    for engine_data in stream {
        let result_batch = into_record_batch(engine_data?);
        assert_eq!(&batch_2, &result_batch);
        files_scanned += 1;
    }
    // One file is scanned as stats are missing so we don't know the predicate isn't satisfied
    assert_eq!(1, files_scanned);

    Ok(())
}

#[test]
fn short_dv() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::array::{Float64Array, Int64Array, StringArray, TimestampMicrosecondArray};

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", delta_kernel::arrow::datatypes::DataType::Int64, false),
        ArrowField::new("value", delta_kernel::arrow::datatypes::DataType::Utf8, false),
        ArrowField::new(
            "timestamp",
            delta_kernel::arrow::datatypes::DataType::Timestamp(
                delta_kernel::arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            false,
        ),
        ArrowField::new("rand", delta_kernel::arrow::datatypes::DataType::Float64, false),
    ]));
    // 2023-05-31T18:58:33.633Z in microseconds
    let ts_micros = 1685559513633000i64;
    let expected = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![3, 4, 5, 6, 7, 8, 9])),
            Arc::new(StringArray::from(vec!["3", "4", "5", "6", "7", "8", "9"])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![ts_micros; 7])
                    .with_timezone("UTC"),
            ),
            Arc::new(Float64Array::from(vec![
                0.7918174793484931,
                0.9281049271981882,
                0.27796520310701633,
                0.15263801464228832,
                0.1981143710215575,
                0.3069439236599195,
                0.5175919190815845,
            ])),
        ],
    )?;
    read_table_data("./tests/data/with-short-dv/", None, None, expected)?;
    Ok(())
}

#[test]
fn basic_decimal() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::array::Decimal128Array;
    use delta_kernel::arrow::datatypes::DataType as ArrowDT;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("part", ArrowDT::Decimal128(12, 5), false),
        ArrowField::new("col1", ArrowDT::Decimal128(5, 2), false),
        ArrowField::new("col2", ArrowDT::Decimal128(10, 5), false),
        ArrowField::new("col3", ArrowDT::Decimal128(20, 10), false),
    ]));

    // part values: -2342342.23423, 0.00004, 234.00000, 2342222.23454
    // col1 values: -999.99, 0.00, 1.00, 111.11
    // col2 values: -99999.99999, 0.00000, 2.00000, 22222.22222
    // col3 values: -9999999999.9999999999, 0.0000000000, 3.0000000000, 3333333333.3333333333
    let expected = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(
                Decimal128Array::from(vec![
                    -234234223423i128,
                    4i128,
                    23400000i128,
                    234222223454i128,
                ])
                .with_precision_and_scale(12, 5)?,
            ),
            Arc::new(
                Decimal128Array::from(vec![-99999i128, 0i128, 100i128, 11111i128])
                    .with_precision_and_scale(5, 2)?,
            ),
            Arc::new(
                Decimal128Array::from(vec![
                    -9999999999i128,
                    0i128,
                    200000i128,
                    2222222222i128,
                ])
                .with_precision_and_scale(10, 5)?,
            ),
            Arc::new(
                Decimal128Array::from(vec![
                    -99999999999999999999i128,
                    0i128,
                    30000000000i128,
                    33333333333333333333i128,
                ])
                .with_precision_and_scale(20, 10)?,
            ),
        ],
    )?;
    read_table_data("./tests/data/basic-decimal-table/", None, None, expected)?;
    Ok(())
}

#[test]
fn timestamp_ntz() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::array::{Int64Array, TimestampMicrosecondArray};
    use delta_kernel::arrow::datatypes::DataType as ArrowDT;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDT::Int64, false),
        ArrowField::new(
            "tsNtz",
            ArrowDT::Timestamp(
                delta_kernel::arrow::datatypes::TimeUnit::Microsecond,
                None,
            ),
            true,
        ),
        ArrowField::new(
            "tsNtzPartition",
            ArrowDT::Timestamp(
                delta_kernel::arrow::datatypes::TimeUnit::Microsecond,
                None,
            ),
            true,
        ),
    ]));

    // 2021-11-18T02:30:00.123456 in micros
    let ts1 = 1637202600123456i64;
    // 2013-07-05T17:01:00.123456 in micros
    let ts2 = 1373043660123456i64;

    let expected = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(ts1),
                Some(ts2),
                None,
                Some(ts1),
                Some(ts2),
                None,
                Some(ts1),
                Some(ts2),
                None,
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(ts1),
                Some(ts1),
                Some(ts1),
                Some(ts2),
                Some(ts2),
                Some(ts2),
                None,
                None,
                None,
            ])),
        ],
    )?;
    read_table_data(
        "./tests/data/data-reader-timestamp_ntz/",
        None,
        None,
        expected,
    )?;
    Ok(())
}

#[test]
fn type_widening_basic() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::array::{Float64Array, Int64Array, TimestampMicrosecondArray};
    use delta_kernel::arrow::datatypes::DataType as ArrowDT;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("byte_long", ArrowDT::Int64, false),
        ArrowField::new("int_long", ArrowDT::Int64, false),
        ArrowField::new("float_double", ArrowDT::Float64, false),
        ArrowField::new("byte_double", ArrowDT::Float64, false),
        ArrowField::new("short_double", ArrowDT::Float64, false),
        ArrowField::new("int_double", ArrowDT::Float64, false),
        ArrowField::new(
            "date_timestamp_ntz",
            ArrowDT::Timestamp(
                delta_kernel::arrow::datatypes::TimeUnit::Microsecond,
                None,
            ),
            false,
        ),
    ]));

    // 2024-09-09T00:00:00 in micros
    let ts_date = 1725840000000000i64;
    // 2024-09-09T12:34:56.123456 in micros
    let ts_full = 1725885296123456i64;

    let expected = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 9223372036854775807i64])),
            Arc::new(Int64Array::from(vec![2, 9223372036854775807i64])),
            Arc::new(Float64Array::from(vec![
                3.4000000953674316,
                1.234567890123,
            ])),
            Arc::new(Float64Array::from(vec![5.0, 1.234567890123])),
            Arc::new(Float64Array::from(vec![6.0, 1.234567890123])),
            Arc::new(Float64Array::from(vec![7.0, 1.234567890123])),
            Arc::new(TimestampMicrosecondArray::from(vec![ts_date, ts_full])),
        ],
    )?;
    let select_cols: Option<&[&str]> = Some(&[
        "byte_long",
        "int_long",
        "float_double",
        "byte_double",
        "short_double",
        "int_double",
        "date_timestamp_ntz",
    ]);

    read_table_data("./tests/data/type-widening/", select_cols, None, expected)
}

#[test]
fn type_widening_decimal() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::array::Decimal128Array;
    use delta_kernel::arrow::datatypes::DataType as ArrowDT;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("decimal_decimal_same_scale", ArrowDT::Decimal128(16, 2), false),
        ArrowField::new("decimal_decimal_greater_scale", ArrowDT::Decimal128(16, 5), false),
        ArrowField::new("byte_decimal", ArrowDT::Decimal128(11, 1), false),
        ArrowField::new("short_decimal", ArrowDT::Decimal128(11, 1), false),
        ArrowField::new("int_decimal", ArrowDT::Decimal128(20, 1), false),
        ArrowField::new("long_decimal", ArrowDT::Decimal128(20, 1), false),
    ]));

    let expected = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(
                Decimal128Array::from(vec![12345i128, 1234567890123456i128])
                    .with_precision_and_scale(16, 2)?,
            ),
            Arc::new(
                Decimal128Array::from(vec![6789000i128, 1234567890123456i128])
                    .with_precision_and_scale(16, 5)?,
            ),
            Arc::new(
                Decimal128Array::from(vec![10i128, 1234i128])
                    .with_precision_and_scale(11, 1)?,
            ),
            Arc::new(
                Decimal128Array::from(vec![20i128, 123456i128])
                    .with_precision_and_scale(11, 1)?,
            ),
            Arc::new(
                Decimal128Array::from(vec![30i128, 12345678901i128])
                    .with_precision_and_scale(20, 1)?,
            ),
            Arc::new(
                Decimal128Array::from(vec![40i128, 1234567890123456789i128])
                    .with_precision_and_scale(20, 1)?,
            ),
        ],
    )?;
    let select_cols: Option<&[&str]> = Some(&[
        "decimal_decimal_same_scale",
        "decimal_decimal_greater_scale",
        "byte_decimal",
        "short_decimal",
        "int_decimal",
        "long_decimal",
    ]);
    read_table_data("./tests/data/type-widening/", select_cols, None, expected)
}

// Verify that predicates over invalid/missing columns do not cause skipping.
#[test]
fn predicate_references_invalid_missing_column() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::util::pretty::pretty_format_batches;

    let path = std::fs::canonicalize(PathBuf::from(
        "./tests/data/parquet_row_group_skipping/",
    ))?;
    let url = url::Url::from_directory_path(path).unwrap();
    let engine = test_utils::create_default_engine(&url)?;

    // Attempted skipping over a logically valid but physically missing column.
    {
        let columns = &["chrono", "missing"];
        let predicate = column_expr!("missing").lt(Expr::literal(10i64));
        let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;

        let table_schema = snapshot.schema();
        let selected_fields = columns
            .iter()
            .map(|col| table_schema.field(col).cloned().unwrap());
        let read_schema = Arc::new(Schema::new_unchecked(selected_fields));

        let scan = snapshot
            .scan_builder()
            .with_schema(read_schema)
            .with_predicate(Arc::new(predicate))
            .build()?;

        let batches = read_scan(&scan, engine.clone())?;
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let mut actual_lines: Vec<&str> = formatted.trim().lines().collect();
        actual_lines.sort();

        let expected = vec![
            "+-------------------------------------------------------------------------------------------+---------+",
            "| chrono                                                                                    | missing |",
            "+-------------------------------------------------------------------------------------------+---------+",
            "| {date32: 1971-01-01, timestamp: 1970-02-01T08:00:00Z, timestamp_ntz: 1970-01-02T00:00:00} |         |",
            "| {date32: 1971-01-02, timestamp: 1970-02-01T09:00:00Z, timestamp_ntz: 1970-01-02T00:01:00} |         |",
            "| {date32: 1971-01-03, timestamp: 1970-02-01T10:00:00Z, timestamp_ntz: 1970-01-02T00:02:00} |         |",
            "| {date32: 1971-01-04, timestamp: 1970-02-01T11:00:00Z, timestamp_ntz: 1970-01-02T00:03:00} |         |",
            "| {date32: 1971-01-05, timestamp: 1970-02-01T12:00:00Z, timestamp_ntz: 1970-01-02T00:04:00} |         |",
            "+-------------------------------------------------------------------------------------------+---------+",
        ];
        let mut expected_sorted = expected.to_vec();
        expected_sorted.sort();
        assert_eq!(expected_sorted, actual_lines);
    }

    // Attempted skipping over an invalid (logically missing) column.
    {
        let columns = &["chrono"];
        let predicate = column_expr!("invalid").lt(Expr::literal(10));
        let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;

        let table_schema = snapshot.schema();
        let selected_fields = columns
            .iter()
            .map(|col| table_schema.field(col).cloned().unwrap());
        let read_schema = Arc::new(Schema::new_unchecked(selected_fields));

        let result = snapshot
            .scan_builder()
            .with_schema(read_schema)
            .with_predicate(Arc::new(predicate))
            .build();
        result.expect_err("unknown column");
    }
    Ok(())
}

// Note: This test is disabled for windows because it creates a directory with name
// `time=1971-07-22T03:06:40.000000Z`. This is disallowed in windows due to having a `:` in
// the name.
#[cfg(not(windows))]
#[test]
fn timestamp_partitioned_table() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::array::{Float64Array, Int64Array, StringArray, TimestampMicrosecondArray};
    use delta_kernel::arrow::datatypes::DataType as ArrowDT;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDT::Int64, false),
        ArrowField::new("x", ArrowDT::Float64, false),
        ArrowField::new("s", ArrowDT::Utf8, true),
        ArrowField::new(
            "time",
            ArrowDT::Timestamp(
                delta_kernel::arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            false,
        ),
    ]));
    // 1971-07-22T03:06:40Z in micros
    let ts_micros = 48826000000000i64;
    let expected = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Float64Array::from(vec![0.5])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![ts_micros]).with_timezone("UTC"),
            ),
        ],
    )?;
    let test_name = "timestamp-partitioned-table";
    let test_dir = load_test_data("./tests/data", test_name).unwrap();
    let test_path = test_dir.path().join(test_name);
    read_table_data(test_path.to_str().unwrap(), None, None, expected)
}

#[test]
fn compacted_log_files_table() -> Result<(), Box<dyn std::error::Error>> {
    let expected = generate_batch(vec![
        ("id", vec![0i32, 1, 2, 10, 12].into_array()),
        (
            "comment",
            vec![
                Some("new"),
                Some("after-large-delete"),
                Some(""),
                Some("merge1-insert"),
                Some("merge2-insert"),
            ]
            .into_array(),
        ),
    ])?;
    let test_name = "compacted-log-files-table";
    let test_dir = load_test_data("./tests/data", test_name).unwrap();
    let test_path = test_dir.path().join(test_name);
    read_table_data(test_path.to_str().unwrap(), None, None, expected)
}

#[test]
fn unshredded_variant_table() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::util::pretty::pretty_format_batches;

    let expected_lines: Vec<&str> = include!("data/unshredded-variant.expected.in");
    let test_name = "unshredded-variant";
    let test_dir = load_test_data("./tests/data", test_name).unwrap();
    let test_path = test_dir.path().join(test_name);

    let path = std::fs::canonicalize(PathBuf::from(test_path))?;
    let url = url::Url::from_directory_path(path).unwrap();
    let engine = test_utils::create_default_engine(&url)?;

    let snapshot = Snapshot::builder_for(url).build(engine.as_ref())?;
    let scan = snapshot.scan_builder().build()?;
    let batches = read_scan(&scan, engine)?;

    let formatted = pretty_format_batches(&batches).unwrap().to_string();
    let mut actual_lines: Vec<&str> = formatted.trim().lines().collect();
    let num_lines = actual_lines.len();
    if num_lines > 3 {
        actual_lines.as_mut_slice()[2..num_lines - 1].sort_unstable();
    }
    let mut expected_sorted = expected_lines;
    let num_expected = expected_sorted.len();
    if num_expected > 3 {
        expected_sorted.as_mut_slice()[2..num_expected - 1].sort_unstable();
    }
    assert_eq!(
        expected_sorted, actual_lines,
        "\n\nexpected:\n\n{:#?}\nactual:\n\n{:#?}\n\n",
        expected_sorted, actual_lines
    );
    Ok(())
}

#[tokio::test]
async fn test_row_index_metadata_column() -> Result<(), Box<dyn std::error::Error>> {
    // Setup up an in-memory table with different numbers of rows in each file
    let batch1 = generate_batch(vec![
        ("id", vec![1i32, 2, 3, 4, 5].into_array()),
        ("value", vec!["a", "b", "c", "d", "e"].into_array()),
    ])?;
    let batch2 = generate_batch(vec![
        ("id", vec![10i32, 20, 30].into_array()),
        ("value", vec!["x", "y", "z"].into_array()),
    ])?;
    let batch3 = generate_batch(vec![
        ("id", vec![100i32, 200, 300, 400].into_array()),
        ("value", vec!["p", "q", "r", "s"].into_array()),
    ])?;

    let file_size1 = record_batch_to_bytes(&batch1).len() as u64;
    let file_size2 = record_batch_to_bytes(&batch2).len() as u64;
    let file_size3 = record_batch_to_bytes(&batch3).len() as u64;
    let storage = Arc::new(InMemory::new());
    add_commit(
        storage.as_ref(),
        0,
        actions_to_string(vec![
            TestAction::Metadata,
            TestAction::AddWithSize(PARQUET_FILE1.to_string(), file_size1),
            TestAction::AddWithSize(PARQUET_FILE2.to_string(), file_size2),
            TestAction::AddWithSize(PARQUET_FILE3.to_string(), file_size3),
        ]),
    )
    .await?;

    for (parquet_file, batch) in [
        (PARQUET_FILE1, &batch1),
        (PARQUET_FILE2, &batch2),
        (PARQUET_FILE3, &batch3),
    ] {
        storage
            .put(
                &Path::from(parquet_file),
                record_batch_to_bytes(batch).into(),
            )
            .await?;
    }

    let location = Url::parse("memory:///")?;
    let engine = Arc::new(DefaultEngineBuilder::new(storage.clone()).build());

    // Create a schema that includes a row index metadata column
    let schema = Arc::new(StructType::try_new([
        StructField::nullable("id", DataType::INTEGER),
        StructField::create_metadata_column("row_index", MetadataColumnSpec::RowIndex),
        StructField::nullable("value", DataType::STRING),
    ])?);

    let snapshot = Snapshot::builder_for(location).build(engine.as_ref())?;
    let scan = snapshot.scan_builder().with_schema(schema).build()?;

    let mut file_count = 0;
    let expected_row_counts = [5, 3, 4];
    let stream = scan.execute(engine.clone())?;

    for data in stream {
        let batch = into_record_batch(data?);
        file_count += 1;

        // Verify the schema structure
        assert_eq!(batch.num_columns(), 3, "Expected 3 columns in the batch");
        assert_eq!(
            batch.schema().field(0).name(),
            "id",
            "First column should be 'id'"
        );
        assert_eq!(
            batch.schema().field(1).name(),
            "row_index",
            "Second column should be 'row_index'"
        );
        assert_eq!(
            batch.schema().field(2).name(),
            "value",
            "Third column should be 'value'"
        );

        // Each file should have row indexes starting from 0 (file-local indexing)
        let row_index_array = batch.column(1).as_primitive::<Int64Type>();
        let expected_values: Vec<i64> = (0..batch.num_rows() as i64).collect();
        assert_eq!(
            row_index_array.values().to_vec(),
            expected_values,
            "Row index values incorrect for file {} (expected {} rows)",
            file_count,
            expected_row_counts[file_count - 1]
        );
    }

    assert_eq!(file_count, 3, "Expected to scan 3 files");
    Ok(())
}

#[tokio::test]
async fn test_file_path_metadata_column() -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::arrow::array::{Array, StringArray};

    // Set up an in-memory table with multiple data files
    let batch1 = generate_batch(vec![
        ("id", vec![1i32, 2, 3].into_array()),
        ("value", vec!["a", "b", "c"].into_array()),
    ])?;
    let batch2 = generate_batch(vec![
        ("id", vec![10i32, 20].into_array()),
        ("value", vec!["x", "y"].into_array()),
    ])?;

    let file_size1 = record_batch_to_bytes(&batch1).len() as u64;
    let file_size2 = record_batch_to_bytes(&batch2).len() as u64;
    let storage = Arc::new(InMemory::new());
    add_commit(
        storage.as_ref(),
        0,
        actions_to_string(vec![
            TestAction::Metadata,
            TestAction::AddWithSize(PARQUET_FILE1.to_string(), file_size1),
            TestAction::AddWithSize(PARQUET_FILE2.to_string(), file_size2),
        ]),
    )
    .await?;

    for (parquet_file, batch) in [(PARQUET_FILE1, &batch1), (PARQUET_FILE2, &batch2)] {
        storage
            .put(
                &Path::from(parquet_file),
                record_batch_to_bytes(batch).into(),
            )
            .await?;
    }

    let location = Url::parse("memory:///")?;
    let engine = Arc::new(DefaultEngineBuilder::new(storage.clone()).build());

    // Create a schema that includes the file path metadata column
    let schema = Arc::new(StructType::try_new([
        StructField::nullable("id", DataType::INTEGER),
        StructField::create_metadata_column("_file", MetadataColumnSpec::FilePath),
        StructField::nullable("value", DataType::STRING),
    ])?);

    let snapshot = Snapshot::builder_for(location.clone()).build(engine.as_ref())?;
    let scan = snapshot.scan_builder().with_schema(schema).build()?;

    let mut file_count = 0;
    let expected_files = [PARQUET_FILE1, PARQUET_FILE2];
    let expected_row_counts = [3, 2];
    let stream = scan.execute(engine.clone())?;

    for data in stream {
        let batch = into_record_batch(data?);

        // Verify the schema structure
        assert_eq!(batch.num_columns(), 3, "Expected 3 columns in the batch");
        assert_eq!(
            batch.schema().field(0).name(),
            "id",
            "First column should be 'id'"
        );
        assert_eq!(
            batch.schema().field(1).name(),
            "_file",
            "Second column should be '_file'"
        );
        assert_eq!(
            batch.schema().field(2).name(),
            "value",
            "Third column should be 'value'"
        );

        // Verify the file path column contains the expected file name
        let file_path_array = batch.column(1);
        let expected_file_name = expected_files[file_count];
        let expected_path = format!("{}{}", location, expected_file_name);

        // The file path array should be a plain StringArray with the path repeated for each row.
        let string_array = file_path_array
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("File path column should be a StringArray");

        assert_eq!(
            string_array.len(),
            expected_row_counts[file_count],
            "File {} should have {} rows",
            expected_file_name,
            expected_row_counts[file_count]
        );
        assert!(
            string_array
                .iter()
                .all(|v| v == Some(expected_path.as_str())),
            "All rows should contain file path '{}'",
            expected_path
        );

        file_count += 1;
    }

    assert_eq!(file_count, 2, "Expected to scan 2 files");
    Ok(())
}

#[tokio::test]
async fn test_unsupported_metadata_columns() -> Result<(), Box<dyn std::error::Error>> {
    // Prepare an in-memory table with some data
    let batch = generate_simple_batch()?;
    let storage = Arc::new(InMemory::new());
    add_commit(
        storage.as_ref(),
        0,
        actions_to_string(vec![
            TestAction::Metadata,
            TestAction::Add(PARQUET_FILE1.to_string()),
        ]),
    )
    .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE1),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;

    let location = Url::parse("memory:///")?;
    let engine = Arc::new(DefaultEngineBuilder::new(storage.clone()).build());

    // Test that unsupported metadata columns fail with appropriate errors
    let test_cases = [
        (
            "row_id",
            MetadataColumnSpec::RowId,
            "Row ids are not enabled on this table",
        ),
        (
            "row_commit_version",
            MetadataColumnSpec::RowCommitVersion,
            "Row commit versions not supported",
        ),
    ];

    for (column_name, metadata_spec, error_text) in test_cases {
        let snapshot = Snapshot::builder_for(location.clone()).build(engine.as_ref())?;
        let schema = Arc::new(StructType::try_new([
            StructField::nullable("id", DataType::INTEGER),
            StructField::create_metadata_column(column_name, metadata_spec),
        ])?);

        let scan_err = snapshot
            .scan_builder()
            .with_schema(schema)
            .build()
            .unwrap_err();
        let error_msg = scan_err.to_string();
        assert!(
            error_msg.contains(error_text),
            "Expected {error_msg} to contain {error_text}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_invalid_files_are_skipped() -> Result<(), Box<dyn std::error::Error>> {
    let batch = generate_simple_batch()?;
    let storage = Arc::new(InMemory::new());
    add_commit(
        storage.as_ref(),
        0,
        actions_to_string(vec![
            TestAction::Metadata,
            TestAction::Add(PARQUET_FILE1.to_string()),
            TestAction::Add(PARQUET_FILE2.to_string()),
        ]),
    )
    .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE1),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;
    storage
        .put(
            &Path::from(PARQUET_FILE2),
            record_batch_to_bytes(&batch).into(),
        )
        .await?;

    let location = Url::parse("memory:///")?;
    let engine = Arc::new(DefaultEngineBuilder::new(storage.clone()).build());

    let invalid_files = [
        "_delta_log/0.zip",
        "_delta_log/_copy_into_log/0.zip",
        "_delta_log/_ignore_me/00000000000000000000.json",
        "_delta_log/_and_me/00000000000000000000.checkpoint.parquet",
        "_delta_log/02184.json",
        "_delta_log/0x000000000000000000.checkpoint.parquet",
        "00000000000000000000.json",
        "_delta_log/_staged_commits/_staged_commits/00000000000000000000.3a0d65cd-4056-49b8-937b-95f9e3ee90e5.json",
        "_delta_log/my_random_dir/_staged_commits/00000000000000000000.3a0d65cd-4056-49b8-937b-95f9e3ee90e5.json",
        "_delta_log/my_random_dir/_delta_log/_staged_commits/00000000000000000000.3a0d65cd-4056-49b8-937b-95f9e3ee90e5.json",
        "_delta_log/_delta_log/00000000000000000000.json",
        "_delta_log/_delta_log/00000000000000000000.checkpoint.parquet",
        "_delta_log/something/_delta_log/00000000000000000000.crc",
        "_delta_log/something/_delta_log/00000000000000000000.json",
        "_delta_log/something/_delta_log/00000000000000000000.checkpoint.parquet",
    ];

    fn get_file_path_for_test(path: &ParsedLogPath) -> &str {
        &path.location.location.as_str()[10..]
    }

    fn ensure_segment_does_not_contain(invalid_files: &[&str], segment: &LogSegment) {
        assert!(
            !segment.listed.ascending_commit_files.iter().any(|p| {
                let test_path = get_file_path_for_test(p);
                invalid_files.contains(&test_path)
            }),
            "ascending_commit_files contained invalid file"
        );
        assert!(
            !segment.listed.ascending_compaction_files.iter().any(|p| {
                let test_path = get_file_path_for_test(p);
                invalid_files.contains(&test_path)
            }),
            "ascending_compaction_files contained invalid file"
        );
        assert!(
            !segment.listed.checkpoint_parts.iter().any(|p| {
                let test_path = get_file_path_for_test(p);
                invalid_files.contains(&test_path)
            }),
            "checkpoint_parts contained invalid file"
        );
        if let Some(ref crc) = segment.listed.latest_crc_file {
            assert!(
                !invalid_files.contains(&get_file_path_for_test(crc)),
                "Latest crc contained invalid file"
            );
        }
        if let Some(ref latest_commit) = segment.listed.latest_commit_file {
            assert!(
                !invalid_files.contains(&get_file_path_for_test(latest_commit)),
                "Latest commit contained invalid file"
            );
        }
    }

    for invalid_file in invalid_files.iter() {
        let invalid_path = Path::from(*invalid_file);
        storage.put(&invalid_path, vec![1u8].into()).await?;
        let snapshot = Snapshot::builder_for(location.clone()).build(engine.as_ref())?;
        ensure_segment_does_not_contain(&invalid_files, snapshot.log_segment());
        storage.delete(&invalid_path).await?;
    }

    // final test with _all_ the files we should ignore
    for invalid_file in invalid_files.iter() {
        let invalid_path = Path::from(*invalid_file);
        storage.put(&invalid_path, vec![1u8].into()).await?;
    }
    let snapshot = Snapshot::builder_for(location).build(engine.as_ref())?;
    ensure_segment_does_not_contain(&invalid_files, snapshot.log_segment());

    Ok(())
}
