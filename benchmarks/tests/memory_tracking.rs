#![cfg(feature = "alloc-tracking")]

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;

use delta_kernel_alloc_tracking::TrackingAlloc;
use delta_kernel_benchmarks::memory::MemoryUsage;

#[global_allocator]
static ALLOCATOR: TrackingAlloc = TrackingAlloc::new();

// Keep the process-global measurement in a single test, including under cargo test.
#[test]
fn global_tracking_includes_worker_temporaries_and_excludes_retained_setup() {
    const SETUP_BYTES: usize = 64 * 1024 * 1024;
    const WORKER_BYTES: usize = 8 * 1024 * 1024;
    let setup = black_box(vec![0u8; SETUP_BYTES]);
    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = barrier.clone();
    let worker = thread::spawn(move || {
        worker_barrier.wait();
        worker_barrier.wait();
        let temporary = black_box(vec![0u8; WORKER_BYTES]);
        black_box(&temporary);
        drop(temporary);
    });
    barrier.wait();
    let usage = MemoryUsage::measure(&ALLOCATOR, || {
        barrier.wait();
        worker.join().unwrap();
        Ok::<_, ()>(())
    })
    .unwrap();
    black_box(&setup);
    assert!(usage.baseline_bytes >= SETUP_BYTES as u64);
    // Leave room for thread teardown and incidental test-harness allocations.
    assert!(usage.peak_increase_bytes >= (WORKER_BYTES / 2) as u64);
    assert!(usage.peak_increase_bytes < SETUP_BYTES as u64);
    let next = MemoryUsage::measure(&ALLOCATOR, || Ok::<_, ()>(())).unwrap();
    assert!(next.peak_increase_bytes < (WORKER_BYTES / 2) as u64);
}
