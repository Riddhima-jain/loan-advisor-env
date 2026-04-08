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

An OpenEnv-compatible RL environment for education loan decision-making. AI agents evaluate whether a student should take an education loan and recommend the optimal financing option.

## 🌍 Why This Matters

**Over 10 million Indian students** apply for education loans each year (RBI data, 2024). Poor loan choices — wrong interest rate, ignoring scholarships, or pursuing low-ROI degrees — can trap families in debt for decades. Most students and parents lack the financial literacy to evaluate complex trade-offs between tuition costs, expected salary growth, EMI affordability, and opportunity costs.

This environment lets you **train and benchmark AI agents** on realistic education loan scenarios, from clear positive-ROI cases (IIT CS) to deceptive negative-ROI traps (expensive arts degrees with bond scholarships). Every data point — tuition fees, salary ranges, loan rates — is sourced from real Indian institutions and published salary surveys.

**Use cases:**
- Benchmark LLM financial reasoning capabilities
- Train RL agents for multi-step financial advisory
- Evaluate agent decision quality on high-stakes, multi-factor problems
- Test robustness against deceptive options (bond scholarship traps)

## 📊 Data Sources

All financial data is sourced from publicly available, verifiable sources:

| Data Type | Source | Year |
|-----------|--------|------|
| IIT Bombay fees | IIT Bombay fee structure website | 2024-25 |
| Private B-school fees | Shiksha.com MBA fee survey | 2024 |
| Symbiosis Design fees | Symbiosis International fee structure | 2024-25 |
| Placement salaries | NIRF Placement Reports, AmbitionBox | 2024 |
| Job market data | LinkedIn Salary Insights (India) | 2024 |
| Loan interest rates | SBI, HDFC Credila, Axis Bank (RBI guidelines) | 2024-25 |

## 🚀 Quick Start

```bash
# Install & start server
pip install -e .
python server/app.py &

# Set credentials (choose one provider)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token"

# Run inference
python inference.py
```

## 📊 Baseline Scores

| Task | Score | Status |
|------|-------|--------|
| task_easy | 0.900 | ✓ |
| task_medium | 1.000 | ✓ |
| task_hard | 1.000 | ✓ |
| **Average** | **0.967** | ✓ |

*Total LLM calls: 3 (one per task)*

---

## Environment Overview

**Why this problem?**  
Education loan decisions are high-stakes, multi-factor problems that millions of students face. This environment trains agents to reason through trade-offs between tuition costs, expected salary growth, and loan affordability.

**What the agent does:**
- Queries tuition fees, salary outlook, and loan products
- Calculates ROI, EMI, and affordability
- Makes a GO/NO-GO recommendation with loan selection

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | API key (required) | - |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |

---

## Tasks

| Task ID | Difficulty | Correct Decision | Description |
|---------|------------|------------------|-------------|
| `task_easy` | Easy | GO + loan_A | IIT Bombay B.Tech CS - clear positive ROI |
| `task_medium` | Medium | GO + loan_A | MBA with scholarship - borderline ROI |
| `task_hard` | Hard | NO-GO | Arts degree - negative ROI, bond scholarship trap |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (`{"task_id": "task_easy"}`) |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List available tasks |
| `/health` | GET | Health check |

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `query_info` / `compare` / `calculate` / `recommend` | Type of action |
| `query_field` | string | For `query_info`: `tuition_fees`, `loan_products`, `salary_outlook`, `user_profile`, `scholarship_options` |
| `loan_ids` | list | For `compare`: two loan IDs to compare |
| `calculation_type` | string | For `calculate`: `roi`, `emi`, `total_cost`, `affordability`, `net_benefit` |
| `recommended_decision` | `go` / `no_go` | For `recommend` |
| `recommended_loan_id` | string | For `recommend` with `go`: best loan product |
| `reasoning` | string | Agent's explanation for the recommendation |

---

## Reward Function

Shaped reward with **partial progress signals** during info gathering, plus a **deterministic grader** on the final recommendation.

### Step Rewards (during research phase)
| Action | Reward | Condition |
|--------|--------|-----------|
| `query_info` | +0.05 | Per unique query field (up to 5 fields) |
| `calculate` | +0.08 | Per unique calculation type |
| `compare` | +0.10 | First comparison |

### Final Grading (on `recommend` action)

| Scenario | Base Score | Process Bonuses | Max |
|----------|-----------|-----------------|-----|
| Correct decision + correct loan | 0.60 | up to +0.40 | **1.00** |
| Correct decision + wrong loan | 0.40 | up to +0.30 | ~0.70 |
| Wrong decision | 0.00 | up to +0.30 | 0.30 |

**Process bonuses** (each +0.10):
- ✅ Comparison done
- ✅ ROI calculated
- ✅ ≥3 unique queries made
- ✅ Scholarship queried (medium/hard tasks only)

---

## Project Structure

```
loan-advisor-env/
├── inference.py          # Main inference script (MANDATORY)
├── models.py             # Pydantic models (Action, Observation, State)
├── client.py             # Python client (OpenEnv + fallback HTTP)
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Package config & dependencies
├── Dockerfile            # Container definition (port 7860)
├── tests/
│   └── test_environment.py   # 54 tests covering all actions, grading, boundaries
└── server/
    ├── app.py            # FastAPI server (6 endpoints)
    ├── environment.py    # Core logic, financial calculations & grading
    └── requirements.txt  # Server dependencies
```

## 🧪 Tests

54 tests covering environment reset, all action types, grading logic, financial calculation accuracy, episode boundaries, deterministic reproducibility, observation structure, close/cleanup, task-specific mechanics (3-loan medium task, bond scholarship trap), and score ranges.

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Inference Approach

**Hybrid Strategy** (scripted info gathering + LLM decision):

1. **Phase 1** (No LLM): Query loan products, scholarships, salary outlook, user profile → Calculate ROI → Compare loans
2. **Phase 2** (1 LLM call): Make final GO/NO-GO recommendation with loan selection

This achieves **96.7% average score** with only **3 LLM calls** total.

---

## Links

- **HuggingFace Space**: https://huggingface.co/spaces/RiddhimaJ/loan-advisor-env
- **GitHub**: https://github.com/Riddhima-jain/loan-advisor-env

---

## License

MIT License
