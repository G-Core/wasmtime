## Cause

This repository is a fork of [bytecodealliance/wasmtime](https://github.com/bytecodealliance/wasmtime)
maintained by G-Core to introduce custom modifications and enhancements of the
`wasi-nn` interface for machine learning inference in WebAssembly, along with a
few targeted fixes required for our integration scenarios.

The fork is used to experiment with new features, fix specific issues, and adapt
the project for integration with tools and workflows not (yet) supported by the
upstream repository.

Upstream base: `release-36.0.0` branch of `bytecodealliance/wasmtime`.

## Notable Changes vs. Upstream

### `wasi-nn`
- **New Candle backend** — added a new `wasi-nn` backend based on the
  [Candle](https://github.com/huggingface/candle) machine learning framework,
  with initial support for the LLaMA model family
  (`crates/wasi-nn/src/backend/candle.rs`).
- **Candle dependency bump** — updated Candle dependencies to `v0.8.4`.
- **Candle output tensor fix** — fixed returning of the output tensor from the
  Candle backend.
- **OpenVINO backend pinned to legacy version** — reverted the OpenVINO backend
  to rely on the legacy `openvino-rs@0.6` dependency and the OpenVINO runtime
  `2023.1` for compatibility with our deployment environment.

### `wasi-http`
- **Allow `HOST` header** — removed `HOST` from `DEFAULT_FORBIDDEN_HEADERS` in
  `crates/wasi-http/src/types.rs` so guests can set/forward the `Host` header
  when making outbound HTTP requests.

### Misc
- Refreshed `Cargo.lock` to the latest compatible dependency versions.
- Added this `FORK.md` document describing the divergence from upstream.
