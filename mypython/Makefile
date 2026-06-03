# ── tiny-gpt Makefile ───────────────────────────────────────────────────────
# Common flow:
#   make setup            install deps (torch)
#   make data             download TinyShakespeare -> data/corpus.jsonl
#   make train            train the 'tiny' preset (~1 min on Apple silicon)
#   make generate         sample text from the trained model
#
# Knobs:
#   make data SIZE=0.25            keep 25% of the corpus (faster)
#   make train PRESET=small        ~10 min, bigger model
#   make generate PROMPT="JULIET:" TOKENS=300

# Uses a local virtualenv at .venv (created by `make setup`).
PY     ?= .venv/bin/python
PRESET ?= tiny
SIZE   ?= 1.0
PROMPT ?= ROMEO:
TOKENS ?= 500

.PHONY: setup data train train-small generate sample clean help

help:
	@echo "targets: setup | data | train | train-small | generate | clean"
	@echo "vars:    PRESET=$(PRESET)  SIZE=$(SIZE)  PROMPT=\"$(PROMPT)\"  TOKENS=$(TOKENS)"

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) data.py --size $(SIZE)

train:
	$(PY) train.py --preset $(PRESET)

train-small:
	$(PY) train.py --preset small

generate sample:
	$(PY) generate.py --prompt "$(PROMPT)" --max-new-tokens $(TOKENS)

clean:
	rm -rf out data/*.jsonl data/input.txt __pycache__
