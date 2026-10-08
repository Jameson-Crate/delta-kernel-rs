//! Advisory live-heap measurements for individual workload executions.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use delta_kernel_alloc_tracking::TrackingAlloc;
use serde::Serialize;

/// Requested Rust heap bytes observed during one operation.
///
/// These process-wide counters include engine workers and background activity. They do not measure
/// RSS or attribute allocations exclusively to the operation. Concurrent allocation around reset
/// can lose extrema, as described by [`TrackingAlloc::reset_stats`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MemoryUsage {
    /// Live requested bytes immediately before the operation.
    pub baseline_bytes: u64,
    /// Advisory maximum live requested bytes during the operation.
    pub peak_live_bytes: u64,
    /// Peak live bytes above the baseline, clamped to zero.
    pub peak_increase_bytes: u64,
}

impl MemoryUsage {
    /// Resets `allocator`, executes `operation` once, and samples its peak before reporting.
    ///
    /// Setup should finish before calling this method, and the operation must finish consuming and
    /// dropping its results before returning. Returns the operation's error without a measurement
    /// if execution fails. Callers must serialize measurement windows.
    pub fn measure<E>(
        allocator: &TrackingAlloc,
        operation: impl FnOnce() -> Result<(), E>,
    ) -> Result<Self, E> {
        let baseline_bytes = allocator.reset_stats().current_bytes;
        operation()?;
        let peak_live_bytes = allocator.max_usage();
        Ok(Self {
            baseline_bytes,
            peak_live_bytes,
            peak_increase_bytes: peak_live_bytes.saturating_sub(baseline_bytes),
        })
    }
}

/// A JSONL artifact containing completed memory profiling iterations for one invocation.
pub struct MemoryReport {
    output: File,
}

impl MemoryReport {
    /// Creates `output_directory` and truncates its `memory.jsonl` artifact.
    ///
    /// Returns an I/O error if the directory or file cannot be created.
    pub fn create(output_directory: &Path) -> io::Result<Self> {
        fs::create_dir_all(output_directory)?;
        Ok(Self {
            output: File::create(output_directory.join("memory.jsonl"))?,
        })
    }

    /// Writes and flushes `usage` under the unchanged Criterion `benchmark` identifier.
    ///
    /// Returns an I/O error if serialization, writing, or flushing fails. Call this outside the
    /// measurement window so report allocations do not contribute to the recorded peak.
    pub fn record(&mut self, benchmark: &str, usage: MemoryUsage) -> io::Result<()> {
        let record = MemoryRecord {
            benchmark,
            unit: "bytes",
            usage,
        };
        serde_json::to_writer(&mut self.output, &record)?;
        self.output.write_all(b"\n")?;
        self.output.flush()
    }
}

#[derive(Serialize)]
struct MemoryRecord<'a> {
    benchmark: &'a str,
    unit: &'static str,
    #[serde(flatten)]
    usage: MemoryUsage,
}

#[cfg(test)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout};

    use serde_json::json;

    use super::*;

    #[test]
    fn measures_temporary_allocations_above_setup_and_resets_between_operations() {
        let allocator = TrackingAlloc::new();
        let setup_layout = Layout::from_size_align(1024, 8).unwrap();
        let operation_layout = Layout::from_size_align(64, 8).unwrap();
        unsafe {
            let setup = allocator.alloc(setup_layout);
            assert!(!setup.is_null());
            let usage = MemoryUsage::measure(&allocator, || {
                let temporary = allocator.alloc(operation_layout);
                assert!(!temporary.is_null());
                allocator.dealloc(temporary, operation_layout);
                Ok::<_, ()>(())
            })
            .unwrap();
            assert_eq!(
                usage,
                MemoryUsage {
                    baseline_bytes: 1024,
                    peak_live_bytes: 1088,
                    peak_increase_bytes: 64,
                }
            );
            let usage = MemoryUsage::measure(&allocator, || {
                allocator.dealloc(setup, setup_layout);
                Ok::<_, ()>(())
            })
            .unwrap();
            assert_eq!(usage.baseline_bytes, 1024);
            assert_eq!(usage.peak_live_bytes, 1024);
            assert_eq!(usage.peak_increase_bytes, 0);
        }
    }

    #[test]
    fn failed_operation_returns_its_error_without_measurement() {
        let result = MemoryUsage::measure(&TrackingAlloc::new(), || Err("operation failed"));
        assert_eq!(result, Err("operation failed"));
    }

    #[test]
    fn report_preserves_identifiers_flushes_records_and_truncates_previous_invocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("memory.jsonl");
        let usage = MemoryUsage {
            baseline_bytes: 128,
            peak_live_bytes: 192,
            peak_increase_bytes: 64,
        };
        let mut report = MemoryReport::create(directory.path()).unwrap();
        for name in [
            "table/snapshot",
            "table/read/parallel2",
            "table/\"quoted\"\ncase",
        ] {
            report.record(name, usage).unwrap();
        }
        let contents = fs::read_to_string(&path).unwrap();
        let records: Vec<serde_json::Value> = contents
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 3);
        assert_eq!(
            records[1],
            json!({
                "benchmark": "table/read/parallel2",
                "unit": "bytes",
                "baseline_bytes": 128,
                "peak_live_bytes": 192,
                "peak_increase_bytes": 64,
            })
        );
        assert_eq!(records[2]["benchmark"], "table/\"quoted\"\ncase");
        drop(report);
        let _report = MemoryReport::create(directory.path()).unwrap();
        assert_eq!(fs::read_to_string(path).unwrap(), "");
    }

    #[test]
    fn report_propagates_creation_failure() {
        let file = tempfile::NamedTempFile::new().unwrap();
        assert!(MemoryReport::create(file.path()).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn report_propagates_write_failure() {
        let mut report = MemoryReport {
            output: File::options().write(true).open("/dev/full").unwrap(),
        };
        let usage = MemoryUsage {
            baseline_bytes: 0,
            peak_live_bytes: 0,
            peak_increase_bytes: 0,
        };
        assert!(report.record("table/snapshot", usage).is_err());
    }
}
