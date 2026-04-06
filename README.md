---
title: Customer Support Openenv
emoji: 🌍
colorFrom: red
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Customer Support Chat Environment (OpenEnv Round 1)

This project implements a real-world OpenEnv-style environment where an AI agent responds to customer complaints.

## Problem Focus

- Domain: customer support complaint handling
- Agent objective: provide correct resolution guidance with appropriate tone
- Evaluation: correctness + tone scores in range 0.0 to 1.0

## Implemented Requirements Mapping

- Real-world task simulation: support complaints for order, billing, and refund issues
- OpenEnv API endpoints: `/reset`, `/step`, `/state`
- Typed request/response models: implemented with Pydantic in `app/models.py`
- Minimum 3 tasks with graders:
  - `easy_wrong_item`
  - `medium_billing_double_charge`
  - `hard_refund_delayed_shipment`
- Reward function with partial progress:
  - correctness score = required intent coverage minus penalties
  - tone score = tone requirement coverage with toxicity guard
  - overall reward = `0.7 * correctness + 0.3 * tone`
- Baseline inference script: `inference.py` (reproducible, deterministic temperature)
- Docker deployment support: `Dockerfile`
- OpenEnv metadata file: `openenv.yaml`

## Environment Variables (Mandatory)

Set these before running inference:

- `API_BASE_URL`: LLM API base URL
- `MODEL_NAME`: model identifier
- `HF_TOKEN`: Hugging Face/API token

Optional:

- `ENV_BASE_URL`: defaults to `http://localhost:7860`

Recommended defaults for Hugging Face router:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=Qwen/Qwen2.5-7B-Instruct`

Copy `.env.example` to `.env` and fill values.

## Quick Start (Local)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Start environment server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

3. Run baseline inference

```bash
python inference.py
```

4. Run pre-submission validator

```bash
python validate_submission.py
```

## API Summary

### POST /reset

Request:

```json
{
  "task_id": "easy_wrong_item"
}
```

Response includes current complaint context.

### POST /step

Request:

```json
{
  "action": "assistant response text"
}
```

Response includes:

- `reward` in `[0.0, 1.0]`
- `score.correctness` in `[0.0, 1.0]`
- `score.tone` in `[0.0, 1.0]`
- `done` flag

### GET /state

Returns current environment state and history.

## Docker

Build:

```bash
docker build -t customer-support-openenv .
```

Run:

```bash
docker run -p 7860:7860 customer-support-openenv
```

## Hugging Face Space Deployment

Use the step-by-step deployment guide in `DEPLOY_HF_SPACE.md`.

## Notes for Submission

- Ensure your hosted service returns HTTP 200 on root and responds correctly to `/reset`
- Keep `inference.py` in repo root (already done)
- Verify runtime is below 20 minutes (this baseline is usually under 1 minute)
- If the organizer's sample logging format differs, align field names exactly in `inference.py`
