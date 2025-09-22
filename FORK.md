## Cause

This repository is a fork of wasmtime to introduce custom modifications and enhancements of 
the wasi-nn interface for machine learning inference in WebAssembly. 

The fork may be used to experiment with new features, fix specific issues, or adapt the 
project for integration with other tools or workflows not supported by the upstream repository. 

### Notable Changes

- Introduced a new `wasi-nn` backend based on the Candle machine learning framework.
- Modified the OpenVINO backend to rely on the legacy `openvino-rs@0.6` dependency for compatibility reasons.

