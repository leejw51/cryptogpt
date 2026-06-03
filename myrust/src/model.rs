//! A minimal decoder-only GPT (nanoGPT-style), in burn (Python: model.py).
//!
//! Decoder-only == every token may only attend to itself and earlier tokens
//! (causal masking). That single constraint is what turns a transformer into an
//! autoregressive language model: predict the next token, one at a time.
//!
//! Difference from the Python reference: PyTorch ties `lm_head.weight` to the
//! token-embedding weight. burn modules own their parameters, so here the
//! language-model head is a separate (untied) `Linear`. The architecture is
//! otherwise identical.
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::config::GptConfig;

/// Multi-head self-attention with a causal (look-back-only) mask.
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    c_attn: Linear<B>, // fused query/key/value projection
    c_proj: Linear<B>,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    n_head: usize,
    n_embd: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        assert!(config.n_embd % config.n_head == 0);
        Self {
            c_attn: LinearConfig::new(config.n_embd, 3 * config.n_embd)
                .with_bias(config.bias)
                .init(device),
            c_proj: LinearConfig::new(config.n_embd, config.n_embd)
                .with_bias(config.bias)
                .init(device),
            attn_dropout: DropoutConfig::new(config.dropout).init(),
            resid_dropout: DropoutConfig::new(config.dropout).init(),
            n_head: config.n_head,
            n_embd: config.n_embd,
        }
    }

    /// `mask`: a `(T, T)` boolean tensor, true where key `j > query i` (future
    /// positions to hide). Built once per forward and shared across blocks.
    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        let head_dim = c / self.n_head;

        // one big projection -> query, key, value, then split along the last dim
        let qkv = self.c_attn.forward(x);
        let q = qkv.clone().slice([0..b, 0..t, 0..self.n_embd]);
        let k = qkv.clone().slice([0..b, 0..t, self.n_embd..2 * self.n_embd]);
        let v = qkv.slice([0..b, 0..t, 2 * self.n_embd..3 * self.n_embd]);

        // (B, T, C) -> (B, n_head, T, head_dim)
        let split = |t_in: Tensor<B, 3>| {
            t_in.reshape([b, t, self.n_head, head_dim]).swap_dims(1, 2)
        };
        let q = split(q);
        let k = split(k);
        let v = split(v);

        // scaled dot-product attention with the (shared) causal mask
        let scale = (head_dim as f64).sqrt();
        let att = q.matmul(k.swap_dims(2, 3)).div_scalar(scale); // (B, nh, T, T)
        let att = att.mask_fill(mask.unsqueeze::<4>(), f32::NEG_INFINITY);
        let att = softmax(att, 3);
        let att = self.attn_dropout.forward(att);

        let y = att.matmul(v); // (B, nh, T, head_dim)
        let y = y.swap_dims(1, 2).reshape([b, t, c]); // (B, T, C)
        self.resid_dropout.forward(self.c_proj.forward(y))
    }
}

/// Position-wise feed-forward network: Linear -> GELU -> Linear (4x hidden).
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    c_fc: Linear<B>,
    gelu: Gelu,
    c_proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        Self {
            c_fc: LinearConfig::new(config.n_embd, 4 * config.n_embd)
                .with_bias(config.bias)
                .init(device),
            gelu: Gelu::new(),
            c_proj: LinearConfig::new(4 * config.n_embd, config.n_embd)
                .with_bias(config.bias)
                .init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = self.gelu.forward(x);
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

/// Transformer block: pre-LayerNorm, attention, MLP, both with residuals.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln_1: LayerNorm<B>,
    attn: CausalSelfAttention<B>,
    ln_2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        Self {
            ln_1: LayerNormConfig::new(config.n_embd).init(device),
            attn: CausalSelfAttention::new(config, device),
            ln_2: LayerNormConfig::new(config.n_embd).init(device),
            mlp: Mlp::new(config, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x), mask);
        x.clone() + self.mlp.forward(self.ln_2.forward(x))
    }
}

/// The full decoder-only GPT.
#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    wte: Embedding<B>, // token embeddings
    wpe: Embedding<B>, // position embeddings
    drop: Dropout,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>, // untied (see module docs)
    block_size: usize,
}

impl<B: Backend> Gpt<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let blocks = (0..config.n_layer)
            .map(|_| Block::new(config, device))
            .collect();
        Self {
            wte: EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device),
            wpe: EmbeddingConfig::new(config.block_size, config.n_embd).init(device),
            drop: DropoutConfig::new(config.dropout).init(),
            blocks,
            ln_f: LayerNormConfig::new(config.n_embd).init(device),
            lm_head: LinearConfig::new(config.n_embd, config.vocab_size)
                .with_bias(false)
                .init(device),
            block_size: config.block_size,
        }
    }

    /// Run the transformer and project to vocabulary logits at every position.
    /// `idx`: (B, T) int token ids. Returns (B, T, vocab_size) logits.
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_b, t] = idx.dims();
        assert!(t <= self.block_size, "sequence length {t} > block_size");
        let device = idx.device();

        // positions 0..t, shared across the batch
        let pos = Tensor::<B, 1, Int>::arange(0..t as i64, &device).reshape([1, t]);
        let tok_emb = self.wte.forward(idx); // (B, T, C)
        let pos_emb = self.wpe.forward(pos); // (1, T, C)
        let mut x = self.drop.forward(tok_emb + pos_emb);

        // Build the causal mask once and share it across every block. In burn,
        // `tril_mask` with offset 0 is true above the diagonal (key j > query i)
        // — the opposite of numpy's naming — which is exactly what we hide.
        let mask = Tensor::<B, 2, Bool>::tril_mask([t, t], 0, &device);
        for block in &self.blocks {
            x = block.forward(x, mask.clone());
        }
        let x = self.ln_f.forward(x);
        self.lm_head.forward(x)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total number of scalar parameters in the model.
    pub fn num_params(&self) -> usize {
        burn::module::Module::num_params(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GptConfig;

    type B = burn::backend::ndarray::NdArray;

    #[test]
    fn forward_is_finite() {
        let device = Default::default();
        let cfg = GptConfig { vocab_size: 10, block_size: 8, n_layer: 2, n_head: 2, n_embd: 16, dropout: 0.0, bias: true };
        let model: Gpt<B> = Gpt::new(&cfg, &device);
        let idx = Tensor::<B, 1, Int>::from_ints([1i64, 2, 3, 4], &device).reshape([1, 4]);
        let logits = model.forward(idx);
        let [b, t, v] = logits.dims();
        assert_eq!([b, t, v], [1, 4, 10]);
        let data: Vec<f32> = logits.into_data().to_vec().unwrap();
        assert!(data.iter().all(|x| x.is_finite()), "forward produced non-finite logits");
    }

    /// The mask must be causal: feeding a longer prefix must not change the
    /// logits already produced for earlier positions.
    #[test]
    fn attention_is_causal() {
        let device = Default::default();
        let cfg = GptConfig { vocab_size: 10, block_size: 8, n_layer: 2, n_head: 2, n_embd: 16, dropout: 0.0, bias: true };
        let model: Gpt<B> = Gpt::new(&cfg, &device);

        let short = Tensor::<B, 1, Int>::from_ints([1i64, 2, 3], &device).reshape([1, 3]);
        let long = Tensor::<B, 1, Int>::from_ints([1i64, 2, 3, 4, 5], &device).reshape([1, 5]);

        let ls: Vec<f32> = model.forward(short).into_data().to_vec().unwrap(); // 3*10
        let ll: Vec<f32> = model.forward(long).into_data().to_vec().unwrap(); // 5*10

        // first 3 positions (30 logits) must match within float tolerance
        for i in 0..30 {
            assert!((ls[i] - ll[i]).abs() < 1e-4, "non-causal at logit {i}: {} vs {}", ls[i], ll[i]);
        }
    }
}
