//! Train the tiny GPT on a JSONL corpus (Python: train.py).
//!
//!     cargo run --release --bin train -- --preset tiny
//!     cargo run --release --features wgpu --bin train -- --preset small
//!
//! The checkpoint bundles the weights (out/model.mpk) plus the config and char
//! vocabulary (out/meta.json), so `generate` can run standalone.
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::grad_clipping::GradientClippingConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use rand::Rng;
use serde::{Deserialize, Serialize};

use cryptogpt::backend::{self, Train};
use cryptogpt::config::{GptConfig, Preset};
use cryptogpt::dataset::{load_jsonl_text, CharTokenizer};
use cryptogpt::model::Gpt;

#[derive(Parser, Debug)]
#[command(about = "Train a tiny decoder-only GPT")]
struct Args {
    #[arg(long, default_value = "tiny")]
    preset: String,
    #[arg(long, default_value = "../data/corpus.jsonl")]
    data: String,
    #[arg(long, default_value = "out")]
    out: String,
    /// Override the preset's iteration count.
    #[arg(long)]
    max_iters: Option<usize>,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
}

/// What we persist alongside the weights so `generate` is self-contained.
#[derive(Serialize, Deserialize)]
pub struct Meta {
    pub config: GptConfig,
    pub vocab: String,
}

fn progress_bar(frac: f64, elapsed: f64, loss: f64, baseline: f64) -> String {
    let width = 24;
    let frac = frac.clamp(0.0, 1.0);
    let filled = (frac * width as f64) as usize;
    let bar: String = "█".repeat(filled) + &"░".repeat(width - filled);
    let eta = if frac > 0.0 { elapsed * (1.0 - frac) / frac } else { 0.0 };
    let learned = 100.0 * (baseline - loss) / baseline;
    format!(
        "\r  [{bar}] {:5.1}%  loss {loss:.3} ({learned:4.1}%)  {elapsed:4.0}s  ETA {eta:4.0}s ",
        frac * 100.0
    )
}

/// Sample `batch_size` random (input, target) windows of length `block_size`.
fn get_batch<B: Backend>(
    data: &[i64],
    batch_size: usize,
    block_size: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let mut rng = rand::thread_rng();
    let mut xs = Vec::with_capacity(batch_size * block_size);
    let mut ys = Vec::with_capacity(batch_size * block_size);
    for _ in 0..batch_size {
        let i = rng.gen_range(0..data.len() - block_size);
        xs.extend_from_slice(&data[i..i + block_size]);
        ys.extend_from_slice(&data[i + 1..i + 1 + block_size]);
    }
    let x = Tensor::<B, 1, Int>::from_ints(xs.as_slice(), device).reshape([batch_size, block_size]);
    let y = Tensor::<B, 1, Int>::from_ints(ys.as_slice(), device).reshape([batch_size, block_size]);
    (x, y)
}

/// Cross-entropy next-token loss for a batch.
fn batch_loss<B: Backend>(
    model: &Gpt<B>,
    x: Tensor<B, 2, Int>,
    y: Tensor<B, 2, Int>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [b, t] = x.dims();
    let logits = model.forward(x); // (B, T, V)
    let vocab = logits.dims()[2];
    let loss_fn = CrossEntropyLossConfig::new().with_logits(true).init(device);
    loss_fn.forward(logits.reshape([b * t, vocab]), y.reshape([b * t]))
}

fn main() {
    let args = Args::parse();
    let preset = Preset::get(&args.preset).unwrap_or_else(|| {
        eprintln!("unknown preset '{}'; use tiny or small", args.preset);
        std::process::exit(1);
    });
    let max_iters = args.max_iters.unwrap_or(preset.max_iters);

    let device = backend::device();
    <Train as Backend>::seed(&device, args.seed);
    println!("preset={}  backend={}", args.preset, backend::name());

    // ---- data ----------------------------------------------------------------
    let text = load_jsonl_text(&args.data).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", args.data);
        std::process::exit(1);
    });
    let tok = CharTokenizer::from_text(&text);
    let data = tok.encode(&text);
    let n = (0.9 * data.len() as f64) as usize;
    let (train_data, val_data) = data.split_at(n);
    let block_size = preset.block_size;
    let batch_size = preset.batch_size;
    println!(
        "chars={}  vocab={}  train={}  val={}",
        text.chars().count(),
        tok.vocab_size(),
        train_data.len(),
        val_data.len()
    );

    // ---- model ----------------------------------------------------------------
    let config = preset.gpt_config(tok.vocab_size());
    let mut model: Gpt<Train> = Gpt::new(&config, &device);
    println!("params: {:.2}M", model.num_params() as f64 / 1e6);

    let mut optim = AdamWConfig::new()
        .with_beta_2(0.99)
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // linear warmup, then cosine decay to 10% of base lr
    let lr_at = |it: usize| -> f64 {
        if it < preset.warmup {
            preset.lr * (it + 1) as f64 / preset.warmup as f64
        } else {
            let ratio = (it - preset.warmup) as f64 / (max_iters - preset.warmup).max(1) as f64;
            preset.lr * (0.1 + 0.9 * 0.5 * (1.0 + (PI * ratio).cos()))
        }
    };

    // Estimate loss on both splits using the inference (non-autodiff) model so
    // no gradient graph is built.
    let estimate_loss = |model: &Gpt<Train>| -> (f64, f64) {
        let valid = model.valid();
        let vdevice = backend::device();
        let mut out = [0.0f64; 2];
        for (s, split) in [train_data, val_data].iter().enumerate() {
            // Accumulate the loss on-device and sync just once per split, instead
            // of pulling each iteration's scalar to the host. On async backends
            // (wgpu) every host read drains the pipeline, so this is a big win.
            let mut acc: Option<Tensor<<Train as AutodiffBackend>::InnerBackend, 1>> = None;
            for _ in 0..preset.eval_iters {
                let (x, y) = get_batch::<<Train as AutodiffBackend>::InnerBackend>(
                    split, batch_size, block_size, &vdevice,
                );
                let loss = batch_loss(&valid, x, y, &vdevice);
                acc = Some(match acc {
                    Some(a) => a + loss,
                    None => loss,
                });
            }
            out[s] = acc.unwrap().into_scalar().elem::<f64>() / preset.eval_iters as f64;
        }
        (out[0], out[1])
    };

    // ---- training loop --------------------------------------------------------
    // Random-guess baseline: a fresh model's loss == ln(vocab_size). We express
    // progress as "learned %" = how far val loss has fallen from that baseline.
    let baseline = (tok.vocab_size() as f64).ln();
    let t0 = Instant::now();
    let mut disp_loss = baseline;

    for it in 0..max_iters {
        let lr = lr_at(it);

        if it % preset.eval_interval == 0 || it == max_iters - 1 {
            let (train_l, val_l) = estimate_loss(&model);
            let progress = 100.0 * (it + 1) as f64 / max_iters as f64;
            let learned = 100.0 * (baseline - val_l) / baseline;
            // \r + clear-to-EOL wipes the live bar, then print a line that stays
            print!(
                "\r\x1b[Kiter {it:5} [{progress:5.1}%] | train {train_l:.3} | val {val_l:.3} \
                 | learned {learned:5.1}% | {:5.1}s\n",
                t0.elapsed().as_secs_f64()
            );
            disp_loss = train_l;
        }

        let (x, y) = get_batch::<Train>(train_data, batch_size, block_size, &device);
        let loss = batch_loss(&model, x, y, &device);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        // Refresh the displayed loss occasionally. Each read syncs the GPU, so
        // keep it infrequent; the bar's progress/time/ETA still move every iter.
        if it % 20 == 0 {
            disp_loss = loss.into_scalar().elem::<f64>();
        }
        print!(
            "{}",
            progress_bar((it + 1) as f64 / max_iters as f64, t0.elapsed().as_secs_f64(), disp_loss, baseline)
        );
        std::io::stdout().flush().ok();
    }
    println!();

    // ---- save -----------------------------------------------------------------
    let out_dir = PathBuf::from(&args.out);
    fs::create_dir_all(&out_dir).expect("create out dir");
    let recorder = burn::record::NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::new();
    let model_path = out_dir.join("model");
    model
        .clone()
        .save_file(model_path.clone(), &recorder)
        .expect("save weights");
    let meta = Meta { config, vocab: tok.vocab_string() };
    fs::write(out_dir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap())
        .expect("save meta");
    println!(
        "saved {}.mpk + meta.json  ({:.1}s total)",
        model_path.display(),
        t0.elapsed().as_secs_f64()
    );
}
