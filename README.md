# ❒ Kairo Core

Core server for Kairo — an NLP-powered note taker.

## Features

Refer to this [OpenAPI Spec](./reports/oas.yml) for the complete list of API endpoints.

## How to use this?

> [!IMPORTANT]
> Install [uv](https://docs.astral.sh/uv/getting-started/installation/) before proceedings.

Before you begin, ensure the following prerequisites are met:

- Run `make` to install dependencies.

- Duplicate `.env.example` to `.env` and set up your READ-ONLY Hugging Face token.

### 1. Download artifacts (Trained models, etc.)

```bash
make dvc-setup
```

_This will redirect you to browser for OAuth. You might encounter unauthorized warning cause I lazy to KYC for GCloud zzZ_

### 2. Start the local server

```bash
make server
```

_This usually starts the server at `http://localhost:8000`_

> [!NOTE]
> Pair this server with Kairo Web 🕸️ [read more...](https://github.com/eesuhn/kairo-web)

## How to train ~~your dragon~~ these models?

> [!CAUTION]
> This is experimental and pretty compute-intensive.

### 1. Download associated datasets

```bash
make data
```

_This will download the Hugging Face datasets based on [this config](./configs/datasets.yml)_

### 2. Training the NER model

```bash
uv run -m scripts.train_ner \
    --mode new \
    --batch-size 32 \
    --epochs 10 \
    --learning-rate 1e-4 \
```

- or you can simply opt out all the configs and run with `--mode new` only

- include `--use-wandb` to log the training to [Weights & Biases](https://docs.wandb.ai/), make sure to login with `uv run wandb login` first

### 3. Training the summarization model

```bash
uv run -m scripts.train.abs_sum
```

```bash
uv run -m scripts.train.ext_sum
```

---

_want to contribute? feel free to drop a PR or raise an issue!_
