//! tiny-gpt in Rust: a decoder-only GPT (nanoGPT-style) built on the `burn`
//! deep-learning framework. Mirrors the companion Python project file-for-file:
//!
//!   backend.rs  -> device / backend selection (Python: utils.py)
//!   config.rs   -> GPTConfig + tiny/small presets (Python: config.py)
//!   dataset.rs  -> char tokenizer + JSONL loader  (Python: dataset.py)
//!   model.rs    -> the GPT itself                 (Python: model.py)
//!   bin/train.rs, bin/generate.rs                 (Python: train.py, generate.py)
pub mod backend;
pub mod config;
pub mod dataset;
pub mod model;
