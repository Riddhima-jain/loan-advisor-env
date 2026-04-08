---
title: Loan Advisor Environment
emoji: 💰
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - finance
  - education
  - rl-environment
---

# 🎓 Loan Advisor Environment

An OpenEnv-compatible RL environment for education loan decision-making. AI agents evaluate whether a student should take a loan and recommend the optimal financing option.

## 🚀 Quick Start

\`\`\`bash
# Install & start server
pip install -e .
python server/app.py &

# Set credentials (choose one provider)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token"

# Run inference
python inference.py
\`\`\`

## 📊 Baseline Scores

| Task | Score | Status |
|------|-------|--------|
| task_easy | 0.900 | ✓ |
| task_medium | 1.000 | ✓ |
| task_hard | 1.000 | ✓ |
| **Average** | **0.967** | ✓ |

*Total LLM calls: 3 (one per task)*

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| \`API_BASE_URL\` | LLM API endpoint | \`https://router.huggingface.co/v1\` |
| \`MODEL_NAME\` | Model identifier | \`Qwen/Qwen2.5-72B-Instruct\` |
| \`HF_TOKEN\` | API key (required) | - |
| \`ENV_BASE_URL\` | Environment server URL | \`http://localhost:7860\` |

---

## Tasks

| Task ID | Difficulty | Correct Decision | Description |
|---------|------------|------------------|-------------|
| \`task_easy\` | Easy | GO + loan_A | IIT Bombay B.Tech CS - clear positive ROI |
| \`task_medium\` | Medium | GO + loan_A | MBA with scholarship - borderline ROI |
| \`task_hard\` | Hard | NO-GO | Arts degree - negative ROI, bond scholarship trap |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/reset\` | POST | Start new episode |
| \`/step\` | POST | Execute action |
| \`/state\` | GET | Get current state |
| \`/tasks\` | GET | List available tasks |
| \`/health\` | GET | Health check |

---

## Project Structure

\`\`\`
loan-advisor-env/
├── inference.py          # Main inference script (MANDATORY)
├── models.py             # Pydantic models (Action, Observation)
├── client.py             # Python client
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Package config
├── Dockerfile            # Container definition (port 7860)
└── server/
    ├── app.py            # FastAPI server
    ├── environment.py    # Core logic & grading
    └── requirements.txt  # Dependencies
\`\`\`

---

## Inference Approach

**Hybrid Strategy** (scripted info gathering + LLM decision):

1. **Phase 1** (No LLM): Query loan products, scholarships, salary outlook, user profile → Calculate ROI → Compare loans
2. **Phase 2** (1 LLM call): Make final GO/NO-GO recommendation with loan selection

This achieves 96.7% average score with only 3 LLM calls total.

---

## Reward Function

- **Base score**: 0.60 for correct decision (GO/NO-GO)
- **Loan bonus**: +0.20 for selecting optimal loan (when GO)
- **Process bonuses** (up to +0.20):
  - +0.05 for ≥3 information queries
  - +0.05 for calculating ROI
  - +0.05 for comparing loans
  - +0.05 for querying scholarships (medium/hard tasks)

---

## Links

- **HuggingFace Space**: https://huggingface.co/spaces/RiddhimaJ/loan-advisor-env
- **GitHub**: https://github.com/Riddhima-jain/loan-advisor-env

---

## License

MIT License
