# VLM reasoning

## What the model is asked

`src/dobot_vlm_keypoints.py` draws a numbered white tag on every candidate object in the captured image and
sends that image with the typed task. The model answers one JSON object:

```json
{"target_mark": 1, "reference_mark": 2, "relation": "beside", "avoid_marks": [3], "confidence": 0.9, "...": "..."}
```

The model never gives coordinates. Pixels come from the capture's own geometry and are verified with depth rays
before the arm moves. Relations are read as seen in the image: `left_of`, `right_of`, `behind` (further up),
`in_front_of` (further down), `opposite_side`, or `beside`.

Answers are checked against a typo-tolerant reading of the task wording. An answer that contradicts an
unambiguous wording is rejected. If no model answers in time, an unambiguous task is decided from its wording.

## Providers (`.env`)

### OpenAI-compatible (`VLM_PROVIDER=openai`)

Any endpoint with `POST /chat/completions` that accepts `image_url` content (a base64 JPEG is sent).

```ini
VLM_PROVIDER=openai
VLM_API_BASE=https://api.openai.com/v1
VLM_MODEL=gpt-4o-mini
VLM_API_KEY=sk-...
```

| Service | `VLM_API_BASE` | `VLM_MODEL` (example) |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | any vision model id |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` |
| Ollama (local) | `http://127.0.0.1:11434/v1` | `qwen2.5vl:7b` (no key) |
| vLLM / LM Studio / a local router | its `/v1` address | its model name |

JSON mode (`response_format`) is requested first. If the server rejects it, the request is retried without it.

### Claude API (`VLM_PROVIDER=anthropic`)

```ini
VLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
VLM_MODEL=claude-haiku-4-5
```

### Claude Code CLI (`VLM_PROVIDER=claude_code`)

This route uses a signed-in Claude Code CLI on a Claude subscription. There is no API key: run `claude` once in a
terminal and log in. Each request runs `claude -p` headless, with read-only access to the tagged image. API-key
variables are removed from its environment, so the call never bills API usage. `VLM_CLAUDE_CODE_MODEL=sonnet`
(stronger) or `haiku` (faster).

### No model (`VLM_PROVIDER=none`)

The task wording alone decides. This works when every object is named unambiguously ("the red cylinder", "the
purple cone").

### Several at once

`VLM_PROVIDER=openai,claude_code` races both routes. With an ambiguous task, two routes must agree, or one
answer with confidence ≥ 0.7 is accepted at the deadline (`VLM_DEADLINE_S`, default 40 s).

## Keys

- Keep keys in `.env` (ignored by git) or in a key file named by `VLM_API_KEY_FILE`, with `chmod 600`.
- Keys are only sent in the request header to their own endpoint. They are never printed or logged, and never
  written to the run folders: those hold the tagged image, a redacted request and the raw answers.
- Requests to `localhost` / `127.x` bypass the HTTP proxy; others use it.

## Check

```bash
python3 src/dobot_vlm_keypoints.py --check     # "route openai up ok (model offered)"
python3 src/dobot_vlm_keypoints.py --task "pick the red cylinder and place it left of the cone"   # needs a capture
```

Every decision is kept in `output/isaac_capture/<capture>/vlm_runs/<run>/`.
