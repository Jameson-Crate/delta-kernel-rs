#!/usr/bin/env python3
"""Smoke-test memory reporting through the workload benchmark executable."""

import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class MemoryReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executables = {tracking: build_benchmark(tracking) for tracking in (False, True)}

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="kernel-bench-workloads-")
        self.addCleanup(directory.cleanup)
        self.workloads = Path(directory.name)
        write_workloads(self.workloads)

    def test_memory_reporting(self):
        cases = (
            ("snapshot", ["--test", "^fixture/snapshotLatest$"], "fixture/snapshotLatest"),
            (
                "metadata",
                ["--test", "^fixture/readMetadataLatest/serial$"],
                "fixture/readMetadataLatest/serial",
            ),
            ("filtered", ["--test", "^does-not-exist$"], None),
            ("list", ["--list"], None),
        )
        for tracking, override, (case, args, expected) in itertools.product(
            (False, True), (False, True), cases
        ):
            with self.subTest(tracking=tracking, override=override, case=case):
                with tempfile.TemporaryDirectory(prefix="kernel-bench-report-") as directory:
                    self.check_run(Path(directory), tracking, override, case, args, expected)

    def check_run(self, directory, tracking, override, case, args, expected):
        env = os.environ.copy()
        for key in ("CRITERION_HOME", "CARGO_CRITERION_PORT", "BENCH_TAGS"):
            env.pop(key, None)
        env["KERNEL_BENCH_WORKLOAD_DIR"] = str(self.workloads)
        env["CARGO_TARGET_DIR"] = str(directory / "target")
        output = directory / "target" / "criterion-alloc-tracking"
        if override:
            output = directory / "custom-criterion"
            env["CRITERION_HOME"] = str(output)

        result = subprocess.run(
            [str(self.executables[tracking]), *args],
            cwd=directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        if expected is not None:
            self.assertIn(expected, result.stdout)
        if case == "list":
            self.assertIn("fixture/snapshotLatest", result.stdout)
            self.assertIn("fixture/readMetadataLatest/serial", result.stdout)

        reports = list(directory.rglob("memory.jsonl"))
        if not tracking:
            self.assertEqual(reports, [])
            return

        report = output / "memory.jsonl"
        self.assertEqual(reports, [report])
        records = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            [record["benchmark"] for record in records],
            [] if expected is None else [expected],
        )
        for record in records:
            self.assertEqual(
                set(record),
                {"benchmark", "unit", "baseline_bytes", "peak_live_bytes", "peak_increase_bytes"},
            )
            self.assertEqual(record["unit"], "bytes")
            for field in ("baseline_bytes", "peak_live_bytes", "peak_increase_bytes"):
                self.assertIs(type(record[field]), int)
                self.assertGreaterEqual(record[field], 0)
            self.assertGreater(record["baseline_bytes"], 0)
            self.assertEqual(
                record["peak_increase_bytes"],
                max(record["peak_live_bytes"] - record["baseline_bytes"], 0),
            )


def build_benchmark(tracking):
    command = [
        "cargo", "bench", "--locked", "-p", "delta_kernel_benchmarks",
        "--bench", "workload_bench", "--no-run", "--profile", "dev",
        "--message-format=json-render-diagnostics",
    ]
    if tracking:
        command.extend(["--features", "alloc-tracking"])
    result = subprocess.run(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE, universal_newlines=True, check=True
    )
    artifacts = (json.loads(line) for line in result.stdout.splitlines())
    executables = {
        Path(artifact["executable"])
        for artifact in artifacts
        if artifact.get("reason") == "compiler-artifact"
        and artifact["target"]["name"] == "workload_bench"
        and "bench" in artifact["target"]["kind"]
        and artifact.get("executable")
    }
    if len(executables) != 1:
        raise RuntimeError(f"Expected one workload_bench executable, got {executables}")
    return executables.pop()


def write_workloads(directory):
    table = directory / "fixture"
    specs = table / "specs"
    specs.mkdir(parents=True)
    table_info = {
        "name": "fixture",
        "description": "Memory reporting smoke test",
        "tablePath": (REPO_ROOT / "kernel/tests/data/basic_partitioned").as_uri() + "/",
        "schema": {
            "type": "struct",
            "fields": [
                {"name": name, "type": data_type, "nullable": True, "metadata": {}}
                for name, data_type in (
                    ("letter", "string"), ("number", "long"), ("a_float", "double")
                )
            ],
        },
        "protocol": {"minReaderVersion": 1, "minWriterVersion": 2},
        "logInfo": {
            "numAddFiles": 6, "numRemoveFiles": 0, "sizeInBytes": 4505,
            "numCommits": 2, "numActions": 10,
        },
        "properties": {},
        "dataLayout": {"numPartitionColumns": 1, "numDistinctPartitions": 5},
        "tags": [],
    }
    (table / "tableInfo.json").write_text(json.dumps(table_info), encoding="utf-8")
    for name, spec_type in (
        ("snapshotLatest", "snapshotConstruction"), ("readMetadataLatest", "read")
    ):
        (specs / f"{name}.json").write_text(json.dumps({"type": spec_type}), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
