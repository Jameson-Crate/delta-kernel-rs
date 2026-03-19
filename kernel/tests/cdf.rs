use std::error;
use std::sync::Arc;

use delta_kernel::arrow::array::{
    Date32Array, Float64Array, Int16Array, Int32Array, RecordBatch,
};
use delta_kernel::arrow::datatypes::Schema as ArrowSchema;
use delta_kernel::engine::arrow_data::EngineDataArrowExt as _;
use itertools::Itertools;

use delta_kernel::engine::arrow_conversion::TryFromKernel as _;
use delta_kernel::table_changes::TableChanges;
use delta_kernel::{DeltaResult, Error, PredicateRef, Version};

mod common;

use test_utils::{assert_batches_eq, generate_batch, load_test_data, IntoArray};

fn read_cdf_for_table(
    test_name: impl AsRef<str>,
    start_version: Version,
    end_version: impl Into<Option<Version>>,
    predicate: impl Into<Option<PredicateRef>>,
) -> DeltaResult<Vec<RecordBatch>> {
    let test_dir = load_test_data("tests/data", test_name.as_ref()).unwrap();
    let test_path = test_dir.path().join(test_name.as_ref());
    let test_path = delta_kernel::try_parse_uri(test_path.to_str().expect("table path to string"))?;
    let engine = test_utils::create_default_engine(&test_path)?;
    let table_changes = TableChanges::try_new(
        test_path,
        engine.as_ref(),
        start_version,
        end_version.into(),
    )?;

    // Project out the commit timestamp since file modification time may change anytime git clones
    // or switches branches
    let names = table_changes
        .schema()
        .fields()
        .map(|field| field.name())
        .filter(|name| *name != "_commit_timestamp")
        .collect_vec();
    let schema = table_changes.schema().project(&names)?;
    let scan = table_changes
        .into_scan_builder()
        .with_schema(schema)
        .with_predicate(predicate)
        .build()?;
    let scan_schema_as_arrow =
        ArrowSchema::try_from_kernel(scan.logical_schema().as_ref()).unwrap();
    let batches: Vec<RecordBatch> = scan
        .execute(engine)?
        .map(|data| -> DeltaResult<_> {
            let record_batch = data?.try_into_record_batch()?;
            // Verify that the arrow record batches match the expected schema
            assert!(record_batch.schema().as_ref() == &scan_schema_as_arrow);
            Ok(record_batch)
        })
        .try_collect()?;
    Ok(batches)
}

#[test]
fn cdf_with_deletion_vector() -> Result<(), Box<dyn error::Error>> {
    let batches = read_cdf_for_table("cdf-table-with-dv", 0, None, None)?;
    // Each commit performs the following:
    // 0. Insert  0..=9
    // 1. Remove  [0, 9]
    // 2. Restore [0, 9]
    // 3. Remove  [0, 1, 4, 5]
    // 4. Restore [1, 4]
    // 5. Restore [0, 5] and Remove [3]
    // 6. Restore 3
    let expected = generate_batch(vec![
        (
            "value",
            vec![
                0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, // insert v0
                0, 9, // delete v1
                0, 9, // insert v2
                0, 1, 4, 5, // delete v3
                1, 4, // insert v4
                3, 0, 5, // delete+insert v5
                3, // insert v6
            ]
            .into_array(),
        ),
        (
            "_change_type",
            vec![
                "insert", "insert", "insert", "insert", "insert", "insert", "insert", "insert",
                "insert", "insert", // v0
                "delete", "delete", // v1
                "insert", "insert", // v2
                "delete", "delete", "delete", "delete", // v3
                "insert", "insert", // v4
                "delete", "insert", "insert", // v5
                "insert", // v6
            ]
            .into_array(),
        ),
        (
            "_commit_version",
            vec![
                0i64, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v0
                1, 1, // v1
                2, 2, // v2
                3, 3, 3, 3, // v3
                4, 4, // v4
                5, 5, 5, // v5
                6, // v6
            ]
            .into_array(),
        ),
    ])?;
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn basic_cdf() -> Result<(), Box<dyn error::Error>> {
    let batches = read_cdf_for_table("cdf-table", 0, None, None)?;

    // Determine the actual birthday column type from the scan results
    let birthday_dates: Vec<Option<i32>> = vec![
        Some(19713), // 2023-12-22
        Some(19714), // 2023-12-23
        Some(19714), // 2023-12-23
        Some(19714), // 2023-12-23
        Some(19715), // 2023-12-24
        Some(19715), // 2023-12-24
        Some(19715), // 2023-12-24
        Some(19716), // 2023-12-25
        Some(19716), // 2023-12-25
        Some(19716), // 2023-12-25
        Some(19713), // 2023-12-22
        Some(19714), // 2023-12-23
        Some(19713), // 2023-12-22
        Some(19714), // 2023-12-23
        Some(19713), // 2023-12-22
        Some(19714), // 2023-12-23
        Some(19715), // 2023-12-24
        Some(19720), // 2023-12-29
        Some(19715), // 2023-12-24
        Some(19720), // 2023-12-29
        Some(19715), // 2023-12-24
        Some(19720), // 2023-12-29
        Some(19720), // 2023-12-29
    ];

    let expected = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            delta_kernel::arrow::datatypes::Field::new("id", delta_kernel::arrow::datatypes::DataType::Int32, true),
            delta_kernel::arrow::datatypes::Field::new("name", delta_kernel::arrow::datatypes::DataType::Utf8, true),
            delta_kernel::arrow::datatypes::Field::new("birthday", delta_kernel::arrow::datatypes::DataType::Date32, true),
            delta_kernel::arrow::datatypes::Field::new("_change_type", delta_kernel::arrow::datatypes::DataType::Utf8, true),
            delta_kernel::arrow::datatypes::Field::new("_commit_version", delta_kernel::arrow::datatypes::DataType::Int64, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // v0
                3, 3, 4, 4, 2, 2, // v1
                7, 7, 5, 5, 6, 6, // v2
                7, // v3
            ])) as Arc<dyn delta_kernel::arrow::array::Array>,
            Arc::new(delta_kernel::arrow::array::StringArray::from(vec![
                "Steve", "Bob", "Dave", "Kate", "Emily", "Carl", "Dennis", "Claire", "Ada", "Borb",
                "Dave", "Dave", "Kate", "Kate", "Bob", "Bob",
                "Dennis", "Dennis", "Emily", "Emily", "Carl", "Carl",
                "Dennis",
            ])) as Arc<dyn delta_kernel::arrow::array::Array>,
            Arc::new(Date32Array::from(birthday_dates)) as Arc<dyn delta_kernel::arrow::array::Array>,
            Arc::new(delta_kernel::arrow::array::StringArray::from(vec![
                "insert", "insert", "insert", "insert", "insert", "insert", "insert", "insert", "insert", "insert",
                "update_postimage", "update_preimage", "update_postimage", "update_preimage", "update_postimage", "update_preimage",
                "update_preimage", "update_postimage", "update_preimage", "update_postimage", "update_preimage", "update_postimage",
                "delete",
            ])) as Arc<dyn delta_kernel::arrow::array::Array>,
            Arc::new(delta_kernel::arrow::array::Int64Array::from(vec![
                0i64, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2,
                3,
            ])) as Arc<dyn delta_kernel::arrow::array::Array>,
        ],
    )?;
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn cdf_non_partitioned() -> Result<(), Box<dyn error::Error>> {
    let batches = read_cdf_for_table("cdf-table-non-partitioned", 0, None, None)?;

    let birthday_dates: Vec<Option<i32>> = vec![
        Some(19827), // 2024-04-14
        Some(19828), // 2024-04-15
        Some(19828), // 2024-04-15
        Some(19828), // 2024-04-15
        Some(19829), // 2024-04-16
        Some(19829), // 2024-04-16
        Some(19829), // 2024-04-16
        Some(19830), // 2024-04-17
        Some(19830), // 2024-04-17
        Some(19830), // 2024-04-17
        Some(19828), // 2024-04-15
        Some(19827), // 2024-04-14
        Some(19828), // 2024-04-15
        Some(19827), // 2024-04-14
        Some(19828), // 2024-04-15
        Some(19827), // 2024-04-14
        Some(19829), // 2024-04-16
        Some(19827), // 2024-04-14
        Some(19829), // 2024-04-16
        Some(19827), // 2024-04-14
        Some(19829), // 2024-04-16
        Some(19827), // 2024-04-14
        Some(19827), // 2024-04-14
        Some(19827), // 2024-04-14
        Some(19828), // 2024-04-15
    ];

    let expected = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            delta_kernel::arrow::datatypes::Field::new("id", delta_kernel::arrow::datatypes::DataType::Int32, true),
            delta_kernel::arrow::datatypes::Field::new("name", delta_kernel::arrow::datatypes::DataType::Utf8, true),
            delta_kernel::arrow::datatypes::Field::new("birthday", delta_kernel::arrow::datatypes::DataType::Date32, true),
            delta_kernel::arrow::datatypes::Field::new("long_field", delta_kernel::arrow::datatypes::DataType::Int64, true),
            delta_kernel::arrow::datatypes::Field::new("boolean_field", delta_kernel::arrow::datatypes::DataType::Boolean, true),
            delta_kernel::arrow::datatypes::Field::new("double_field", delta_kernel::arrow::datatypes::DataType::Float64, true),
            delta_kernel::arrow::datatypes::Field::new("smallint_field", delta_kernel::arrow::datatypes::DataType::Int16, true),
            delta_kernel::arrow::datatypes::Field::new("_change_type", delta_kernel::arrow::datatypes::DataType::Utf8, true),
            delta_kernel::arrow::datatypes::Field::new("_commit_version", delta_kernel::arrow::datatypes::DataType::Int64, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                3, 3, 4, 4, 2, 2,
                7, 7, 5, 5, 6, 6,
                7,
                1, 2,
            ])),
            Arc::new(delta_kernel::arrow::array::StringArray::from(vec![
                "Steve", "Bob", "Dave", "Kate", "Emily", "Carl", "Dennis", "Claire", "Ada", "Borb",
                "Dave", "Dave", "Kate", "Kate", "Bob", "Bob",
                "Dennis", "Dennis", "Emily", "Emily", "Carl", "Carl",
                "Dennis",
                "Alex", "Alan",
            ])),
            Arc::new(Date32Array::from(birthday_dates)),
            Arc::new(delta_kernel::arrow::array::Int64Array::from(vec![
                1i64, 1, 2, 3, 4, 5, 6, 7, 8, 99999999999999999,
                2, 2, 3, 3, 1, 1,
                6, 6, 4, 4, 5, 5,
                6,
                1, 1,
            ])),
            Arc::new(delta_kernel::arrow::array::BooleanArray::from(vec![
                true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true,
                true, true, true, true, true, true,
                true,
                true, true,
            ])),
            Arc::new(Float64Array::from(vec![
                3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
                3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
                3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
                3.14,
                3.14, 3.14,
            ])),
            Arc::new(Int16Array::from(vec![
                1i16, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
                1,
                1, 1,
            ])),
            Arc::new(delta_kernel::arrow::array::StringArray::from(vec![
                "insert", "insert", "insert", "insert", "insert", "insert", "insert", "insert", "insert", "insert",
                "update_preimage", "update_postimage", "update_preimage", "update_postimage", "update_preimage", "update_postimage",
                "update_preimage", "update_postimage", "update_preimage", "update_postimage", "update_preimage", "update_postimage",
                "delete",
                "insert", "insert",
            ])),
            Arc::new(delta_kernel::arrow::array::Int64Array::from(vec![
                0i64, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2,
                3,
                4, 4,
            ])),
        ],
    )?;
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn cdf_with_cdc_and_dvs() -> Result<(), Box<dyn error::Error>> {
    let batches = read_cdf_for_table("cdf-table-with-cdc-and-dvs", 0, None, None)?;
    let expected = generate_batch(vec![
        (
            "id",
            vec![
                1i32, 2, 3, 4, 5, 3, 3, 4, 5, 4, 5,
                1, 1, 2, 2, 3, 3,
                1, 2, 2,
                6, 7, 8, 9, 8, 7,
                10, 11, 9, 9,
                11, 11,
                12, 11,
                3, 4, 5, 2, 6, 9,
                0, 1, 2,
            ]
            .into_array(),
        ),
        (
            "comment",
            vec![
                "initial",
                "insert1",
                "insert1-delete1",
                "insert1-delete2",
                "insert1-delete2",
                "insert1-delete1",
                "insert1-delete1",
                "insert1-delete2",
                "insert1-delete2",
                "insert1-delete2",
                "insert2",
                "initial",
                "update1",
                "insert1",
                "update1",
                "insert1-delete1",
                "update1",
                "update1",
                "update1",
                "update2",
                "insert3",
                "insert3",
                "insert4",
                "insert4",
                "insert4",
                "insert3",
                "merge1-insert",
                "merge1-insert",
                "merge1-update",
                "insert4",
                "merge1-insert",
                "",
                "merge2-insert",
                "",
                "update1",
                "insert1-delete2",
                "insert2",
                "update2",
                "insert3",
                "merge1-update",
                "new",
                "after-large-delete",
                "",
            ]
            .into_array(),
        ),
        (
            "_change_type",
            vec![
                "insert",           // v0
                "insert",           // v1
                "insert",           // v1
                "insert",           // v1
                "insert",           // v1
                "delete",           // v2
                "insert",           // v4
                "delete",           // v5
                "delete",           // v5
                "insert",           // v7
                "insert",           // v8
                "update_preimage",  // v9
                "update_postimage", // v9
                "update_preimage",  // v9
                "update_postimage", // v9
                "update_preimage",  // v9
                "update_postimage", // v9
                "delete",           // v10
                "update_preimage",  // v12
                "update_postimage", // v12
                "insert",           // v14
                "insert",           // v14
                "insert",           // v15
                "insert",           // v15
                "delete",           // v16
                "delete",           // v16
                "insert",           // v18
                "insert",           // v18
                "update_postimage", // v18
                "update_preimage",  // v18
                "update_preimage",  // v20
                "update_postimage", // v20
                "insert",           // v22
                "delete",           // v22
                "delete",           // v24
                "delete",           // v24
                "delete",           // v24
                "delete",           // v24
                "delete",           // v24
                "delete",           // v24
                "insert",           // v25
                "insert",           // v25
                "insert",           // v25
            ]
            .into_array(),
        ),
        (
            "_commit_version",
            vec![
                0i64, 1, 1, 1, 1, 2, 4, 5, 5, 7, 8, 9, 9, 9, 9, 9, 9, 10, 12, 12, 14, 14, 15,
                15, 16, 16, 18, 18, 18, 18, 20, 20, 22, 22, 24, 24, 24, 24, 24, 24, 25, 25, 25,
            ]
            .into_array(),
        ),
    ])?;
    assert_batches_eq(&expected, &batches);
    Ok(())
}

/// Helper to build expected data for simple id(Int64) + _change_type + _commit_version tables
fn simple_cdf_batch_i64(
    ids: Vec<i64>,
    change_types: Vec<&'static str>,
    versions: Vec<i64>,
) -> RecordBatch {
    generate_batch(vec![
        ("id", ids.into_array()),
        ("_change_type", change_types.into_array()),
        ("_commit_version", versions.into_array()),
    ])
    .unwrap()
}

/// Helper to build expected data for simple id(Int32) + _change_type + _commit_version tables
fn simple_cdf_batch(
    ids: Vec<i32>,
    change_types: Vec<&'static str>,
    versions: Vec<i64>,
) -> RecordBatch {
    generate_batch(vec![
        ("id", ids.into_array()),
        ("_change_type", change_types.into_array()),
        ("_commit_version", versions.into_array()),
    ])
    .unwrap()
}

#[test]
fn simple_cdf_version_ranges() -> DeltaResult<()> {
    let batches = read_cdf_for_table("cdf-table-simple", 0, 0, None)?;
    let expected = simple_cdf_batch_i64(
        (0..10).collect(),
        vec!["insert"; 10],
        vec![0; 10],
    );
    assert_batches_eq(&expected, &batches);

    let batches = read_cdf_for_table("cdf-table-simple", 1, 1, None)?;
    let expected = simple_cdf_batch_i64(
        (0..10).collect(),
        vec!["delete"; 10],
        vec![1; 10],
    );
    assert_batches_eq(&expected, &batches);

    let batches = read_cdf_for_table("cdf-table-simple", 2, 2, None)?;
    let expected = simple_cdf_batch_i64(
        (20..25).collect(),
        vec!["insert"; 5],
        vec![2; 5],
    );
    assert_batches_eq(&expected, &batches);

    let batches = read_cdf_for_table("cdf-table-simple", 0, 2, None)?;
    let mut ids: Vec<i64> = (0..10).collect();
    ids.extend(0..10);
    ids.extend(20..25);
    let mut change_types: Vec<&str> = vec!["insert"; 10];
    change_types.extend(vec!["delete"; 10]);
    change_types.extend(vec!["insert"; 5]);
    let mut versions: Vec<i64> = vec![0; 10];
    versions.extend(vec![1; 10]);
    versions.extend(vec![2; 5]);
    let expected = simple_cdf_batch_i64(ids, change_types, versions);
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn update_operations() -> DeltaResult<()> {
    let batches = read_cdf_for_table("cdf-table-update-ops", 0, 2, None)?;
    let expected = simple_cdf_batch_i64(
        vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // insert v0
            20, 21, 22, 23, 24, // update_pre v1
            30, 31, 32, 33, 34, // update_post v2
        ],
        vec![
            "insert", "insert", "insert", "insert", "insert", "insert", "insert", "insert",
            "insert", "insert", "update_pre", "update_pre", "update_pre", "update_pre",
            "update_pre", "update_post", "update_post", "update_post", "update_post",
            "update_post",
        ],
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v0
            1, 1, 1, 1, 1, // v1
            2, 2, 2, 2, 2, // v2
        ],
    );
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn false_data_change_is_ignored() -> DeltaResult<()> {
    let batches = read_cdf_for_table("cdf-table-data-change", 0, 1, None)?;
    let expected = simple_cdf_batch_i64(
        (0..10).collect(),
        vec!["insert"; 10],
        vec![0; 10],
    );
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn invalid_range_end_before_start() {
    let res = read_cdf_for_table("cdf-table-simple", 1, 0, None);
    let expected_msg =
        "Failed to build LogSegment: start_version cannot be greater than end_version";
    assert!(matches!(res, Err(Error::Generic(msg)) if msg == expected_msg));
}

#[test]
fn invalid_range_start_after_last_version_of_table() {
    let res = read_cdf_for_table("cdf-table-simple", 3, 4, None);
    let expected_msg = "Expected the first commit to have version 3, got None";
    assert!(matches!(res, Err(Error::Generic(msg)) if msg == expected_msg));
}

#[test]
fn partition_table() -> DeltaResult<()> {
    let batches = read_cdf_for_table("cdf-table-partitioned", 0, 2, None)?;
    let expected = generate_batch(vec![
        (
            "id",
            vec![0i64, 1, 2, 3, 4, 5, 3, 1, 1, 0, 2, 4].into_array(),
        ),
        (
            "text",
            vec![
                "old", "old", "old", "old", "old", "old", "old", "old", "new", "old", "old", "old",
            ]
            .into_array(),
        ),
        (
            "part",
            vec![0i64, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0].into_array(),
        ),
        (
            "_change_type",
            vec![
                "insert",
                "insert",
                "insert",
                "insert",
                "insert",
                "insert",
                "delete",
                "update_preimage",
                "update_postimage",
                "delete",
                "delete",
                "delete",
            ]
            .into_array(),
        ),
        (
            "_commit_version",
            vec![0i64, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2].into_array(),
        ),
    ])?;
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn backtick_column_names() -> DeltaResult<()> {
    let batches = read_cdf_for_table("cdf-table-backtick-column-names", 0, None, None)?;

    // This test has struct columns with backtick names - construct with explicit schema
    let struct_fields = delta_kernel::arrow::datatypes::Fields::from(vec![
        Arc::new(delta_kernel::arrow::datatypes::Field::new(
            "field",
            delta_kernel::arrow::datatypes::DataType::Int32,
            true,
        )),
        Arc::new(delta_kernel::arrow::datatypes::Field::new(
            "field.one",
            delta_kernel::arrow::datatypes::DataType::Int32,
            true,
        )),
    ]);
    let struct_col = delta_kernel::arrow::array::StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 1, 1])) as Arc<dyn delta_kernel::arrow::array::Array>,
            Arc::new(Int32Array::from(vec![2, 2, 2, 2, 2])) as Arc<dyn delta_kernel::arrow::array::Array>,
        ],
        None,
    );

    let schema = Arc::new(ArrowSchema::new(vec![
        delta_kernel::arrow::datatypes::Field::new("id.num", delta_kernel::arrow::datatypes::DataType::Int32, true),
        delta_kernel::arrow::datatypes::Field::new("id.num`s", delta_kernel::arrow::datatypes::DataType::Int32, true),
        delta_kernel::arrow::datatypes::Field::new(
            "struct_col",
            delta_kernel::arrow::datatypes::DataType::Struct(struct_fields),
            true,
        ),
        delta_kernel::arrow::datatypes::Field::new("_change_type", delta_kernel::arrow::datatypes::DataType::Utf8, true),
        delta_kernel::arrow::datatypes::Field::new("_commit_version", delta_kernel::arrow::datatypes::DataType::Int64, true),
    ]));

    let expected = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![2, 4, 1, 3, 5])),
            Arc::new(Int32Array::from(vec![10, 10, 10, 10, 10])),
            Arc::new(struct_col) as Arc<dyn delta_kernel::arrow::array::Array>,
            Arc::new(delta_kernel::arrow::array::StringArray::from(vec![
                "insert", "insert", "insert", "insert", "insert",
            ])),
            Arc::new(delta_kernel::arrow::array::Int64Array::from(vec![
                0i64, 0, 1, 1, 1,
            ])),
        ],
    )?;
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn unconditional_delete() -> DeltaResult<()> {
    let batches = read_cdf_for_table("cdf-table-delete-unconditional", 0, None, None)?;
    let mut ids: Vec<i64> = (0..10).collect();
    ids.extend(0..10);
    let mut change_types: Vec<&str> = vec!["insert"; 10];
    change_types.extend(vec!["delete"; 10]);
    let mut versions: Vec<i64> = vec![0; 10];
    versions.extend(vec![1; 10]);
    let expected = simple_cdf_batch_i64(ids, change_types, versions);
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn conditional_delete_all_rows() -> DeltaResult<()> {
    let batches = read_cdf_for_table("cdf-table-delete-conditional-all-rows", 0, None, None)?;
    let mut ids: Vec<i64> = (0..10).collect();
    ids.extend(0..10);
    let mut change_types: Vec<&str> = vec!["insert"; 10];
    change_types.extend(vec!["delete"; 10]);
    let mut versions: Vec<i64> = vec![0; 10];
    versions.extend(vec![1; 10]);
    let expected = simple_cdf_batch_i64(ids, change_types, versions);
    assert_batches_eq(&expected, &batches);
    Ok(())
}

#[test]
fn conditional_delete_two_rows() -> DeltaResult<()> {
    let batches = read_cdf_for_table("cdf-table-delete-conditional-two-rows", 0, None, None)?;
    let mut ids: Vec<i64> = (0..10).collect();
    ids.extend(vec![2, 8]);
    let mut change_types: Vec<&str> = vec!["insert"; 10];
    change_types.extend(vec!["delete"; 2]);
    let mut versions: Vec<i64> = vec![0; 10];
    versions.extend(vec![1; 2]);
    let expected = simple_cdf_batch_i64(ids, change_types, versions);
    assert_batches_eq(&expected, &batches);
    Ok(())
}

/// Helper to build expected data for column mapping CDF tests (id, name, value, _change_type, _commit_version)
fn column_mapping_cdf_batch(
    ids: Vec<i64>,
    names: Vec<&'static str>,
    values: Vec<f64>,
    change_types: Vec<&'static str>,
    versions: Vec<i64>,
) -> RecordBatch {
    generate_batch(vec![
        ("id", ids.into_array()),
        ("name", names.into_array()),
        ("value", values.into_array()),
        ("_change_type", change_types.into_array()),
        ("_commit_version", versions.into_array()),
    ])
    .unwrap()
}

#[test]
fn cdf_with_column_mapping_name_mode() -> Result<(), Box<dyn error::Error>> {
    // NOTE: these tables only have CDF enabled in version 1+, so we start reading from 1. This is
    // due to pyspark limitation while writing: we were unable to create a table with column
    // mapping + CDF enabled in commit 0, so we created with column mapping and enabled CDF in
    // commit 1.
    let batches = read_cdf_for_table("cdf-column-mapping-name-mode", 1, None, None)?;
    let expected = column_mapping_cdf_batch(
        vec![1, 2, 2, 4],
        vec!["Alice", "Bob", "Bob", "David"],
        vec![100.0, 200.0, 250.0, 400.0],
        vec!["delete", "update_preimage", "update_postimage", "insert"],
        vec![4, 2, 2, 3],
    );
    assert_batches_eq(&expected, &batches);

    // same as above but instead of protocol 2,5 this is 3,7 with columnMapping+DV features
    let batches = read_cdf_for_table("cdf-column-mapping-name-mode-3-7", 1, None, None)?;
    let expected = column_mapping_cdf_batch(
        vec![1, 2, 2, 4],
        vec!["Alice", "Bob", "Bob", "David"],
        vec![100.0, 200.0, 250.0, 400.0],
        vec!["delete", "update_preimage", "update_postimage", "insert"],
        vec![4, 2, 2, 3],
    );
    assert_batches_eq(&expected, &batches);

    Ok(())
}

#[test]
fn cdf_with_column_mapping_id_mode() -> Result<(), Box<dyn error::Error>> {
    // NOTE: these tables only have CDF enabled in version 1+, so we start reading from 1. This is
    // due to pyspark limitation while writing: we were unable to create a table with column
    // mapping + CDF enabled in commit 0, so we created with column mapping and enabled CDF in
    // commit 1.
    let batches = read_cdf_for_table("cdf-column-mapping-id-mode", 1, None, None)?;
    let expected = column_mapping_cdf_batch(
        vec![2, 2, 3, 4],
        vec!["Frank", "Frank", "Grace", "Henry"],
        vec![250.0, 275.0, 350.0, 450.0],
        vec!["update_preimage", "update_postimage", "delete", "insert"],
        vec![2, 2, 4, 3],
    );
    assert_batches_eq(&expected, &batches);
    Ok(())
}
