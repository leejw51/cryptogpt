# ── tiny-gpt (Rust / burn) Makefile ─────────────────────────────────────────
# Common flow:
#   make train            train the 'tiny' preset
#   make generate         sample text from the trained model
#
# Knobs:
#   make train PRESET=small               bigger model
#   make train BACKEND=metal              train on the Apple GPU (fastest on Mac)
#   make train BACKEND=wgpu               train on the GPU (Metal/Vulkan/DX12)
#   make generate PROMPT="JULIET:" TOKENS=300

PRESET  ?= tiny
PROMPT  ?= ROMEO:
TOKENS  ?= 500
DATA    ?= ../data/corpus.jsonl
BACKEND ?= ndarray        # ndarray (CPU) | wgpu (GPU) | metal (Apple GPU, fastest)

ifeq ($(BACKEND),metal)
  FEATURES := --no-default-features --features metal
else ifeq ($(BACKEND),wgpu)
  FEATURES := --no-default-features --features wgpu
else
  FEATURES :=
endif

.PHONY: build train train-small generate sample test clean help

help:
	@echo "targets: build | train | train-small | generate | test | clean"
	@echo "vars:    PRESET=$(PRESET)  BACKEND=$(BACKEND)  PROMPT=\"$(PROMPT)\"  TOKENS=$(TOKENS)"

build:
	cargo build --release $(FEATURES)

train:
	cargo run --release $(FEATURES) --bin train -- --preset $(PRESET) --data $(DATA)

train-small:
	cargo run --release $(FEATURES) --bin train -- --preset small --data $(DATA)

generate sample:
	cargo run --release $(FEATURES) --bin generate -- --prompt "$(PROMPT)" --max-new-tokens $(TOKENS)

test:
	cargo test

clean:
	cargo clean
	rm -rf out
