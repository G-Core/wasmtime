# G-Core fork of Wasmtime

## Purpose

This repository is a fork of [bytecodealliance/wasmtime](https://github.com/bytecodealliance/wasmtime)
maintained by G-Core to introduce custom modifications and enhancements of the
`wasi-nn` interface for machine learning inference in WebAssembly, along with a
few targeted fixes required for our integration scenarios.

The fork is used to experiment with new features, fix specific issues, and adapt
the project for integration with tools and workflows not (yet) supported by the
upstream repository.

## Upstream Base

- Branch: `release-48.0.0` of `bytecodealliance/wasmtime`.
- Version: Wasmtime `48.0.5` (tag `v48.0.5`).
- All fork-specific commits sit on top of that tag; see `git log v48.0.5..HEAD`.

## Notable Changes vs. Upstream

### `wasi-nn`
- **New Candle backend** — added a new `wasi-nn` backend based on the
  [Candle](https://github.com/huggingface/candle) machine learning framework,
  with initial support for the LLaMA model family
  (`crates/wasi-nn/src/backend/candle.rs`).
- **Candle dependency bump** — updated Candle dependencies to `v0.8.4`.
- **Candle WIT compute** — the Candle backend accepts WIT named-tensor inputs in
  `compute()` and reports a missing input as `BackendError::NoInputs`.
- **Candle output tensor fix** — fixed returning of the output tensor from the
  Candle backend.
- **OpenVINO backend is upstream** — the fork no longer pins the legacy
  `openvino-rs@0.6` / OpenVINO `2023.1` backend. `openvino.rs` is untouched, so
  it uses `openvino-rs 0.11` and needs an OpenVINO runtime **>= 2025.1** (the
  bindings refuse older runtimes at load; tested upstream against 2026.1 and
  2026.4.1).
- **OpenVINO device properties** — `OpenvinoBackend::with_properties` takes
  plain `(key, value)` OpenVINO property strings (`NUM_STREAMS`,
  `INFERENCE_NUM_THREADS`, `ENABLE_CPU_PINNING`, …) and sets them on the target
  device before each `compile_model`. Without it the CPU plugin's `LATENCY`
  default compiles one stream whose single graph is mutex-guarded per
  inference, so a graph shared across request threads runs serialized. Not
  upstream as of `v48.0.5`.

### `wasmtime` runtime
- **`process_madvise` fallback** — in `crates/wasmtime/src/runtime/vm/sys/unix/vm.rs`
  a failed batched `process_madvise` is never reported as a decommit failure.
  On kernels before 6.14, `PIDFD_SELF` fails with `EBADF`; upstream treats that
  as a real error, so the pooling allocator drops the memory images of the
  whole batch and remaps every slot on its next use. The fork falls back to
  per-region `madvise` instead. Not upstream as of `v48.0.5`.

### `wasi-http`
- **Prevent duplicate `HOST` header** — in
  `crates/wasi-http/src/p2/http_impl.rs` the outgoing handler now only injects
  the `Host` header derived from the request authority when the guest hasn't
  already supplied one, avoiding duplicate `Host` headers on outbound
  requests.

### Misc
- `Cargo.lock` is upstream's, plus only what Candle needs (its crates and the
  minimum bumps of `crc32fast`, `half`, `rayon`, `rayon-core`, `version_check`).
- Added this `FORK.md` document describing the divergence from upstream.

## Updating from Upstream

The fork follows upstream **release branches** (`release-XX.0.0`) rather than
`main`. Each time we move to a new Wasmtime release, the fork-specific commits
are replayed on top of the new upstream release branch and published as a new
`release-XX.0.0` branch in this repository (see `release-36.0.0`,
`release-48.0.0`, ...). The previous release branch is kept for reference.

### 1. Make sure the `upstream` remote exists

```bash
git remote add upstream https://github.com/bytecodealliance/wasmtime.git  # once
git fetch upstream --tags
```

### 2. Identify the fork-specific commits

All commits after the upstream base are ours. On the current branch:

```bash
git log --oneline v48.0.5..release-48.0.0
```

This list (Candle backend, Candle deps bump, Candle WIT compute, output tensor
fix, `HOST` header fix, `process_madvise` fallback, `.gitignore`, `FORK.md`) is what has to be carried over.

### 3. Create the new fork branch from the upstream release

Pick the target upstream release branch (for example `release-49.0.0`) and
start the new fork branch from it:

```bash
git checkout -b release-49.0.0 upstream/release-49.0.0
git submodule update --init --recursive
```

### 4. Replay the fork commits

Either rebase the old fork branch onto the new base:

```bash
git rebase --onto release-49.0.0 v48.0.5 release-48.0.0 \
  && git branch -f release-49.0.0 HEAD
```

or cherry-pick the commits from step 2 one by one, which is usually easier
to reason about when conflicts appear:

```bash
git cherry-pick <first-fork-commit>^..<last-fork-commit>
```

Conflicts are most likely in:

- `crates/wasi-nn/` — `Cargo.toml` features and `src/backend/mod.rs` (backend
  registration, `BackendError::NoInputs`). If upstream changed the `BackendInner` / `BackendExecutionContext`
  traits, adapt `src/backend/candle.rs` accordingly.
- `crates/wasi-http/src/p2/http_impl.rs` — the `Host` header injection.
- `crates/wasmtime/src/runtime/vm/sys/unix/vm.rs` — the `process_madvise`
  fallback; drop it if upstream starts treating `EBADF` as "unsupported".
- `Cargo.lock` — take the new base's lockfile (`git checkout --ours Cargo.lock`
  during a cherry-pick) and let `cargo metadata` add only what Candle needs,
  rather than merging lock hunks.

### 5. Build and verify

```bash
cargo check --workspace
cargo build -p wasmtime-wasi-nn --features candle
cargo test -p wasmtime-wasi-nn
cargo test -p wasmtime-wasi-http
```

Run the `wasi-nn` examples in `crates/wasi-nn/examples/` against a real model
to confirm the Candle backend still produces an output tensor.

### 6. Update this document and publish

- Update the **Upstream Base** section above with the new branch, version and
  tag.
- Amend **Notable Changes vs. Upstream** if any fork change was dropped
  because upstream absorbed it, or if new ones were added.
- Push the new branch to `origin` and switch downstream consumers to it:

```bash
git push -u origin release-49.0.0
```

### Picking up point releases on the same branch

When upstream publishes a point release on the branch we already track (for
example `v48.0.2` on `release-48.0.0`), simply rebase the fork commits onto
the updated upstream branch instead of creating a new branch:

```bash
git fetch upstream --tags
git rebase upstream/release-48.0.0
git push --force-with-lease origin release-48.0.0
```

Pull requests to the upstream repository are never opened from this fork;
upstream contributions are made manually by humans in line with the
[Bytecode Alliance AI Tool Use Policy](https://github.com/bytecodealliance/governance/blob/main/AI_TOOL_POLICY.md).
