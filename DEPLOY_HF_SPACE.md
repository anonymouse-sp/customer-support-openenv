# Hugging Face Space Deployment Guide

This guide takes your local project to a working Hugging Face Space for Round 1 submission.

## 1. Prepare GitHub Repository

1. Create a public GitHub repository (or private if event allows).
2. Push this project root as-is.
3. Ensure these files are present in repo root:
   - `Dockerfile`
   - `requirements.txt`
   - `openenv.yaml`
   - `inference.py`
   - `README.md`
   - `app/` folder

## 2. Create Hugging Face Space

1. Go to Hugging Face Spaces and click Create new Space.
2. Choose:
   - SDK: Docker
   - Space visibility: as required by organizers
   - Hardware: CPU basic is typically enough for this environment API
3. Create the Space.

## 3. Connect Space to GitHub

1. In your new Space settings, connect the GitHub repository.
2. Select the main branch.
3. Trigger build/deploy.

## 4. Add Required Secrets/Variables

In Space Settings -> Variables and secrets, add:

1. Variable: `API_BASE_URL` = `https://router.huggingface.co/v1`
2. Variable: `MODEL_NAME` = `Qwen/Qwen2.5-7B-Instruct`
3. Secret: `HF_TOKEN` = your real token
4. Variable: `ENV_BASE_URL` = your deployed Space URL (after first deploy), for example:
   - `https://your-space-name.hf.space`

Notes:
- Keep `HF_TOKEN` only as a secret.
- Do not commit `.env`.

## 5. Confirm App Health

After deploy completes, open:

1. `https://your-space-name.hf.space/`
2. `https://your-space-name.hf.space/tasks`

Expected:
- Root returns HTTP 200 with status `ok`.
- Tasks endpoint returns 3 tasks.

## 6. Submission Validation Before Final Submit

Run locally one final time:

1. `python validate_submission.py`
2. `python inference.py`

Expected:
- Validator JSON shows `"passed": true`
- Inference prints `[START]`, `[STEP]`, `[END]` logs.

## 7. What URL to Submit

Submit the repository URL and the Space URL exactly as requested by organizer form.

## 8. Common Issues and Fixes

1. Build fails on Space:
   - Check `Dockerfile` is at root.
   - Check `requirements.txt` exists and has valid package versions.

2. Inference fails with model not found:
   - Confirm `MODEL_NAME` matches a router-available model.
   - Keep `API_BASE_URL` as `https://router.huggingface.co/v1`.

3. Inference fails with auth error:
   - Recreate token and update Space secret `HF_TOKEN`.

4. Health check fails:
   - Confirm app starts at port 7860 as defined in `Dockerfile` command.
