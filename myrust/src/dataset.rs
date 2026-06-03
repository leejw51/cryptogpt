//! Character-level tokenizer + JSONL corpus loader (Python: dataset.py).
//!
//! One token == one character. The vocabulary is built from the training text
//! itself (sorted unique chars) and stored inside the checkpoint, so generation
//! uses the exact same mapping.
use std::collections::HashMap;
use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharTokenizer {
    /// The vocabulary, in id order: `vocab[id] == char`. This single ordered
    /// list captures both the id->char (itos) and char->id (stoi) mappings.
    vocab: Vec<char>,
    #[serde(skip)]
    stoi: HashMap<char, usize>,
}

impl CharTokenizer {
    fn from_vocab(vocab: Vec<char>) -> Self {
        let stoi = vocab.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { vocab, stoi }
    }

    /// Build a tokenizer from text: the vocabulary is the sorted set of chars,
    /// matching Python's `sorted(set(text))`.
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        chars.sort_unstable(); // BTreeSet is already sorted; explicit for clarity
        Self::from_vocab(chars)
    }

    /// Rebuild a tokenizer from a saved vocabulary string (chars in id order).
    pub fn from_vocab_string(vocab: &str) -> Self {
        Self::from_vocab(vocab.chars().collect())
    }

    /// The vocabulary as a single string, chars in id order — what we persist.
    pub fn vocab_string(&self) -> String {
        self.vocab.iter().collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode text to token ids, silently dropping out-of-vocabulary chars
    /// (matches Python's `if c in self.stoi`).
    pub fn encode(&self, s: &str) -> Vec<i64> {
        s.chars()
            .filter_map(|c| self.stoi.get(&c).map(|&i| i as i64))
            .collect()
    }

    /// Decode token ids back to text.
    pub fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .filter_map(|&i| self.vocab.get(i as usize))
            .collect()
    }
}

/// Read a JSONL file of `{"text": ...}` records and join them into one string
/// (Python: load_jsonl_text).
pub fn load_jsonl_text(path: &str) -> std::io::Result<String> {
    let raw = fs::read_to_string(path)?;
    let mut texts = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
            texts.push(t.to_string());
        }
    }
    Ok(texts.join("\n"))
}
