# Delta error catalog prototype

`delta-error-classes.json` is an intentionally small subset copied verbatim from the OSS Delta
catalog at commit `250aa903c7566f597ef7ccb5827fb29744f30592`:

<https://github.com/delta-io/delta/blob/250aa903c7566f597ef7ccb5827fb29744f30592/spark/src/main/resources/error/delta-error-classes.json>

The prototype vendors only the conditions exercised by kernel. A production implementation should
pin and import the complete catalog through a repeatable update process.

`kernel-error-classes.json` contains the option 2 prototype's custom fallback. It keeps legacy
kernel errors in the diagnostic source chain while exposing a stable generic condition and
SQLSTATE.

The option 2 facade is additive and Rust-only. Existing engine callbacks still use the legacy
result type, and FFI continues to project structured errors as its generic ABI-safe code.
