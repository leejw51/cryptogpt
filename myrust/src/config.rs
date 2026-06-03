//! Model + training presets (Python: config.py).
//!
//! `GptConfig` describes the architecture. `Preset` bundles the architecture
//! with training hyper-parameters so you can pick a target wall-clock time:
//!
//! - `tiny`  -> ~seconds-to-a-minute on a laptop
//! - `small` -> minutes, a noticeably better model
use serde::{Deserialize, Serialize};

/// Architecture description. Serialized into the checkpoint so `generate` can
/// rebuild the exact same model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GptConfig {
    pub vocab_size: usize, // set at train time from the char tokenizer
    pub block_size: usize, // context length (tokens the model can attend to)
    pub n_layer: usize,    // number of transformer blocks
    pub n_head: usize,     // attention heads per block
    pub n_embd: usize,     // embedding / residual stream width
    pub dropout: f64,
    pub bias: bool,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            block_size: 128,
            n_layer: 4,
            n_head: 4,
            n_embd: 128,
            dropout: 0.0,
            bias: true,
        }
    }
}

/// Architecture + optimizer/loop settings, tuned to hit a target wall-clock.
#[derive(Clone, Debug)]
pub struct Preset {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub batch_size: usize,
    pub max_iters: usize,
    pub eval_interval: usize,
    pub eval_iters: usize,
    pub lr: f64,
    pub warmup: usize,
}

impl Preset {
    /// Look up a preset by name (`"tiny"` or `"small"`).
    pub fn get(name: &str) -> Option<Preset> {
        match name {
            "tiny" => Some(Preset {
                n_layer: 4,
                n_head: 4,
                n_embd: 128,
                block_size: 128,
                batch_size: 32,
                max_iters: 600,
                eval_interval: 100,
                eval_iters: 50,
                lr: 1e-3,
                warmup: 50,
            }),
            "small" => Some(Preset {
                n_layer: 6,
                n_head: 6,
                n_embd: 384,
                block_size: 256,
                batch_size: 64,
                max_iters: 3000,
                eval_interval: 250,
                eval_iters: 100,
                lr: 3e-4,
                warmup: 150,
            }),
            _ => None,
        }
    }

    /// Build a `GptConfig` from this preset and a vocabulary size.
    pub fn gpt_config(&self, vocab_size: usize) -> GptConfig {
        GptConfig {
            vocab_size,
            block_size: self.block_size,
            n_layer: self.n_layer,
            n_head: self.n_head,
            n_embd: self.n_embd,
            dropout: 0.0,
            bias: true,
        }
    }
}
