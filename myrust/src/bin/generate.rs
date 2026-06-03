//! Generate text from a trained checkpoint (Python: generate.py).
//!
//!     cargo run --release --bin generate -- --prompt "ROMEO:" --max-new-tokens 400
use std::fs;
use std::path::PathBuf;

use burn::prelude::*;
use burn::tensor::activation::softmax;
use clap::Parser;
use rand::distributions::{Distribution, WeightedIndex};

use cryptogpt::backend::{self, Infer};
use cryptogpt::dataset::CharTokenizer;
use cryptogpt::model::Gpt;

// The on-disk metadata format written by `train` (kept in sync by hand).
#[derive(serde::Deserialize)]
struct Meta {
    config: cryptogpt::config::GptConfig,
    vocab: String,
}

#[derive(Parser, Debug)]
#[command(about = "Sample text from a trained tiny GPT")]
struct Args {
    #[arg(long, default_value = "out")]
    ckpt: String,
    #[arg(long, default_value = "\n")]
    prompt: String,
    #[arg(long, default_value_t = 500)]
    max_new_tokens: usize,
    /// Lower = safer/repetitive, higher = wilder.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    /// Sample only from the k most likely next characters.
    #[arg(long, default_value_t = 200)]
    top_k: usize,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    let device = backend::device();
    <Infer as Backend>::seed(&device, args.seed);

    let ckpt = PathBuf::from(&args.ckpt);
    let meta: Meta = serde_json::from_str(
        &fs::read_to_string(ckpt.join("meta.json")).expect("read meta.json"),
    )
    .expect("parse meta.json");
    let tok = CharTokenizer::from_vocab_string(&meta.vocab);

    let recorder = burn::record::NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::new();
    let model: Gpt<Infer> = Gpt::new(&meta.config, &device)
        .load_file(ckpt.join("model"), &recorder, &device)
        .expect("load weights");

    // Seed the context with the prompt (or a single token 0 if it's empty).
    let mut ids = tok.encode(&args.prompt);
    if ids.is_empty() {
        ids.push(0);
    }

    let block_size = model.block_size();
    let mut rng = rand::thread_rng();

    for _ in 0..args.max_new_tokens {
        // crop to the context window
        let start = ids.len().saturating_sub(block_size);
        let cond = &ids[start..];
        let t = cond.len();
        let x = Tensor::<Infer, 1, Int>::from_ints(cond, &device).reshape([1, t]);

        let logits = model.forward(x); // (1, T, V)
        let vocab = logits.dims()[2];
        // logits at the final position, scaled by temperature
        let last = logits
            .slice([0..1, t - 1..t, 0..vocab])
            .reshape([vocab])
            .div_scalar(args.temperature.max(1e-8));
        let probs: Vec<f32> = softmax(last, 0).into_data().to_vec::<f32>().unwrap();

        let next = sample_top_k(&probs, args.top_k, &mut rng);
        ids.push(next as i64);
    }

    println!("{}", tok.decode(&ids));
}

/// Sample an index from `probs`, restricted to the `k` most-likely entries.
fn sample_top_k(probs: &[f32], k: usize, rng: &mut impl rand::Rng) -> usize {
    let k = k.min(probs.len());
    // indices of the top-k probabilities
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let top = &order[..k];

    let weights: Vec<f32> = top.iter().map(|&i| probs[i]).collect();
    let dist = WeightedIndex::new(&weights).expect("valid distribution");
    top[dist.sample(rng)]
}
