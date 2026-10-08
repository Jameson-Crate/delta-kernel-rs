use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use criterion::Criterion;
#[cfg(feature = "alloc-tracking")]
use delta_kernel_alloc_tracking::TrackingAlloc;
#[cfg(feature = "alloc-tracking")]
use delta_kernel_benchmarks::memory::{MemoryReport, MemoryUsage};
use delta_kernel_benchmarks::registry::BenchRegistry;
use delta_kernel_benchmarks::runners::{
    benchmark_name, configured_benchmark_name, create_read_runner, SnapshotConstructionRunner,
    WorkloadRunner,
};
use delta_kernel_benchmarks::utils::load_all_workloads;
use delta_kernel_workloads::models::{ReadOperation, Spec};
use test_utils::CountingReporter;

#[cfg(feature = "alloc-tracking")]
#[global_allocator]
static GLOBAL_ALLOC: TrackingAlloc = TrackingAlloc::new();

// Checked-in registry mapping each benchmark to its harness configs. Lives under the crate root
// (not the gitignored, downloaded `workloads/` dir), so it is loaded relative to
// CARGO_MANIFEST_DIR.
const REGISTRY_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench-registry.json");

// Loads all workloads and sets up a shared runtime, then registers each as a top-level benchmark.
// For each workload, builds a runner that encapsulates the state (table info, engine, config, etc.)
// and execution logic. After each Criterion timing pass, profiles one execution's IO and,
// with allocation tracking enabled, live heap usage.
fn workload_benchmarks(
    c: &mut Criterion,
    #[cfg(feature = "alloc-tracking")] memory_report: &mut MemoryReport,
) {
    let workloads = match load_all_workloads() {
        Ok(workloads) if !workloads.is_empty() => workloads,
        Ok(_) => panic!("No workloads found"),
        Err(e) => panic!("Failed to load workloads: {e}"),
    };

    let registry = BenchRegistry::load_from_path(Path::new(REGISTRY_PATH))
        .expect("Failed to load bench-registry.json");
    registry
        .validate(&workloads)
        .expect("bench-registry.json must match the loaded workload types");

    let reporter = Arc::new(CountingReporter::new());
    let runtime = Arc::new(tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"));

    for workload in &workloads {
        let case_name = &workload.case_name;
        match &workload.spec {
            Spec::Read(read_spec) => {
                let configs = registry
                    .read_configs(workload)
                    .expect("loaded workload must have a registry table key");
                for operation in [ReadOperation::ReadMetadata] {
                    for config in &configs {
                        let name = configured_benchmark_name(
                            &workload.table_info,
                            case_name,
                            &config.name,
                        );
                        let runner = create_read_runner(
                            name,
                            read_spec,
                            operation,
                            config.clone(),
                            &workload.table_info,
                            runtime.clone(),
                        )
                        .expect("Failed to create read runner");
                        run_benchmark(
                            c,
                            runner.as_ref(),
                            &reporter,
                            #[cfg(feature = "alloc-tracking")]
                            memory_report,
                        );
                    }
                }
            }
            Spec::SnapshotConstruction(snapshot_construction_spec) => {
                let name = benchmark_name(&workload.table_info, case_name);
                let runner = SnapshotConstructionRunner::setup(
                    name,
                    snapshot_construction_spec,
                    &workload.table_info,
                    runtime.clone(),
                )
                .expect("Failed to create snapshot construction runner");
                run_benchmark(
                    c,
                    &runner,
                    &reporter,
                    #[cfg(feature = "alloc-tracking")]
                    memory_report,
                );
            }
        }
    }
}

// Registers a workload with Criterion and benchmarks its `execute()` function.
// After timing completes, runs one profiling iteration for IO and optional live heap usage.
// Profiling is skipped entirely when Criterion filters out the benchmark,
// since Criterion never calls the closure for filtered benchmarks.
fn run_benchmark(
    c: &mut Criterion,
    runner: &dyn WorkloadRunner,
    reporter: &CountingReporter,
    #[cfg(feature = "alloc-tracking")] memory_report: &mut MemoryReport,
) {
    let bench_ran = AtomicBool::new(false);
    c.bench_function(runner.name(), |b| {
        bench_ran.store(true, Ordering::Relaxed);
        b.iter(|| runner.execute().expect("Benchmark execution failed"))
    });
    if bench_ran.load(Ordering::Relaxed) {
        reporter.reset();
        #[cfg(feature = "alloc-tracking")]
        let usage = MemoryUsage::measure(&GLOBAL_ALLOC, || runner.execute())
            .expect("IO and memory profiling iteration failed");
        #[cfg(not(feature = "alloc-tracking"))]
        runner.execute().expect("IO profiling iteration failed");
        reporter.print_summary(runner.name());
        #[cfg(feature = "alloc-tracking")]
        {
            memory_report
                .record(runner.name(), usage)
                .expect("Failed to write memory report");
            println!(
                "{}: peak heap increase {} bytes (baseline {}, peak live {})",
                runner.name(),
                usage.peak_increase_bytes,
                usage.baseline_bytes,
                usage.peak_live_bytes,
            );
        }
    }
}

fn main() {
    let criterion = Criterion::default();
    #[cfg(feature = "alloc-tracking")]
    let output_directory = std::env::var_os("CRITERION_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "target/criterion-alloc-tracking".into());
    #[cfg(feature = "alloc-tracking")]
    let criterion = criterion.output_directory(&output_directory);
    let mut criterion = criterion.configure_from_args();
    #[cfg(feature = "alloc-tracking")]
    let mut memory_report = {
        eprintln!(
            "Allocation tracking enabled: Criterion timings include tracking overhead. \
             Memory report: {}",
            output_directory.join("memory.jsonl").display(),
        );
        MemoryReport::create(&output_directory).expect("Failed to create memory report")
    };
    workload_benchmarks(
        &mut criterion,
        #[cfg(feature = "alloc-tracking")]
        &mut memory_report,
    );
    criterion.final_summary();
}
