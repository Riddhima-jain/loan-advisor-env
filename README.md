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
  - india
---

# Loan Advisor OpenEnv

An OpenEnv-compatible reinforcement learning environment where AI agents learn to make optimal **education loan decisions**.

Given a student profile and a course/university they want to attend, the agent must:
1. Research tuition costs, salary outlook, and available loan products
2. Calculate ROI, EMI affordability, and total repayment cost
3. Decide: **Go** (take a loan and pick the best product) or **No-Go** (education not financially worthwhile)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# 3. In another terminal, set up LLM credentials
# Option A: Using HuggingFace Inference Router
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"  # Get token at https://huggingface.co/settings/tokens

# Option B: Using Groq (free tier - recommended for testing)
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your-groq-api-key"  # Get free key at https://console.groq.com

# 4. Run inference
python inference.py

# 5. (Optional) Try interactive mode with your own details
python interactive_client.py
```

---

## Environment Description

**Why this problem?**
Education loan decisions are high-stakes, multi-factor problems that millions of Indian students face every year. The right decision requires balancing tuition costs, expected salary growth, monthly affordability, and alternative options like scholarships. This environment trains agents to reason through these trade-offs systematically.

**Primary market**: Indian consumers (INR). The architecture is globally extensible — loan tasks for USD, EUR, GBP markets can be added by adding new entries to the `TASKS` dict in `server/environment.py` without any code changes.

**What the agent does**:
- Queries a simulated tuition + job opportunity lookup table (deterministic, no real HTTP calls)
- Calculates EMI, total repayment cost, ROI, and affordability
- Makes a Go/No-go recommendation with optional loan product selection

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | `"query_info"` \| `"compare"` \| `"calculate"` \| `"recommend"` | Type of action |
| `query_field` | `"tuition_fees"` \| `"user_profile"` \| `"salary_outlook"` \| `"loan_products"` \| `"scholarship_options"` | For `query_info`: which data to retrieve |
| `loan_ids` | `list[str]` | For `compare`: two loan IDs to compare |
| `calculation_type` | `"emi"` \| `"total_cost"` \| `"roi"` \| `"affordability"` \| `"net_benefit"` | For `calculate` |
| `loan_id` | `str` | For `calculate`: target loan ID |
| `recommended_decision` | `"go"` \| `"no_go"` | For `recommend` |
| `recommended_loan_id` | `str` \| `null` | For `recommend` with `go`: best loan product |
| `reasoning` | `str` | Agent's explanation (used in partial reward) |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Current task identifier |
| `task_description` | `str` | Natural-language task description |
| `action_result` | `str` | Result of the last action |
| `student_profile_summary` | `str` | Student financial snapshot |
| `course_university` | `str` | Target course and university |
| `available_loan_ids` | `list[str]` | Loan product IDs for this task |
| `steps_taken` | `int` | Steps used so far |
| `max_steps` | `int` | Maximum steps allowed |
| `episode_done` | `bool` | `True` when episode ends |
| `final_reward` | `float \| null` | Score in [0, 1]; set only at episode end |
| `correct_answer` | `str \| null` | Revealed at end: `"go:loan_A"` or `"no_go"` |

---

## Task Descriptions

### task_easy — IIT CS Degree (Clear ROI Case)
**Difficulty**: Easy | **Max Steps**: 8

Rahul wants to pursue B.Tech Computer Science at IIT Bombay. Total tuition: ₹8,00,000 over 4 years. Average starting salary for CS graduates: ₹15L/year. Two loan options: SBI Education Loan (8.5% with 4-year moratorium) and HDFC Credila (10.5%, no moratorium). No scholarships available.

**Expected behavior**: Agent queries loan products and salary outlook, calculates ROI, and correctly recommends **GO + loan_A** (lower rate + moratorium = significantly lower total cost and easily affordable on ₹15L salary).

---

### task_medium — Mid-Tier MBA (Borderline ROI with Scholarship)
**Difficulty**: Medium | **Max Steps**: 10

Divya currently earns ₹40,000/month. She wants an MBA at a private B-school (₹20L, 2 years). Post-MBA salary outlook: ₹8L/year (only ₹3.2L increment). Three loan options with varying rates. A ₹5L merit scholarship is available, reducing the principal to ₹15L.

**Expected behavior**: Agent must discover the scholarship, apply it to the calculation, and compare options. Correct answer: **GO + loan_A** (SBI Scholar with scholarship applied — best ROI and lowest total cost).

---

### task_hard — Arts Degree (Negative ROI / No-Go)
**Difficulty**: Hard | **Max Steps**: 12

Meera earns ₹30,000/month and supports 2 aging parents. She wants to study BFA + Film Studies at Symbiosis International (₹25L, 3 years). Post-grad arts salary: ₹4.5L/year — *less than her current income*. A bond scholarship covers ₹20L but obligates her to 3 years at below-market salary (₹3L opportunity cost).

**Expected behavior**: After calculating ROI (negative) and affordability (EMI exceeds 30% of post-grad income), agent should recommend **NO-GO**. The education does not pay off financially, and the loan would be unaffordable given parent dependency.

---

## Reward Function

The reward function provides **shaped intermediate signals** throughout the episode:

| Action | Intermediate Reward |
|---|---|
| New `query_info` (first time) | +0.05 per unique field |
| First `compare` | +0.10 |
| New `calculate` (first time for that type) | +0.08 per unique calculation |

**Final reward** (on `recommend`):

| Outcome | Base | + Process Bonus | Max |
|---|---|---|---|
| Correct decision + correct loan | 0.60 | +0.10 comparison, +0.10 ROI, +0.10 scholarship queried, +0.10 if ≥3 queries | 1.00 |
| Correct decision + wrong loan | 0.40 | same process bonuses | ~0.70 |
| Wrong decision | 0.00 | +0.10 comparison, +0.10 ROI, +0.10 if ≥2 queries | 0.30 |

`success = final_reward >= 0.5`

---

## Setup & Usage

### Prerequisites
```bash
pip install openenv-core fastapi uvicorn pydantic openai requests
```

### Local Development
```bash
cd loan-advisor-env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

The server will be available at `http://localhost:7860`.

**Quick test:**
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

### Docker Build & Run
```bash
# Build (from loan-advisor-env/ root)
docker build -t loan-advisor-env .

# Run
docker run -p 7860:7860 loan-advisor-env

# Test
curl -X POST http://localhost:7860/reset -d '{"task_id": "task_easy"}' -H "Content-Type: application/json"
```

### Run Inference (Hybrid Baseline)

The inference script uses a **hybrid approach** for optimal efficiency:
- **Phase 1**: Scripted information gathering (no LLM calls)
- **Phase 2**: LLM-powered final recommendation (1 call per task)
- **Total**: Only 3 LLM calls for all tasks!

```bash
# Set environment variables (choose one provider)

# Option A: Groq (free tier, fast)
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"  # or llama-3.3-70b-versatile
export HF_TOKEN="your-groq-api-key"

# Option B: HuggingFace Inference Router
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"

# Environment URL (default)
export ENV_BASE_URL="http://localhost:7860"

# Run inference
python inference.py
```

**Expected output:**
```
[INFO] Starting Loan Advisor inference with model: llama-3.1-8b-instant
[START] task=task_easy env=loan_advisor_env model=llama-3.1-8b-instant
[STEP] step=1 action=query_info reward=0.05 done=false error=null
...
[END] success=true steps=6 score=0.900 rewards=0.05,0.05,0.05,0.08,0.10,0.90

==================================================
HYBRID BASELINE SUMMARY
==================================================
✓ PASS  task_easy: score=0.900
✓ PASS  task_medium: score=1.000
✓ PASS  task_hard: score=1.000
--------------------------------------------------
Average Score: 0.967
LLM Calls: 3 total (1 per task)
==================================================
```

### Interactive Client (User Input Mode)

Try the loan advisor with your own details:

```bash
python interactive_client.py
```

This interactive tool:
- Takes your personal financial details (income, expenses, savings)
- Lets you choose from pre-defined courses or enter custom details
- Calculates EMI, ROI, and affordability
- Provides personalized loan recommendations

Works in **demo mode** (rule-based) without LLM, or **full mode** with LLM for AI-powered advice.

<details>
<summary>📋 Example Interactive Session (click to expand)</summary>

```
============================================================
🎓 EDUCATION LOAN ADVISOR
============================================================

I'll help you decide whether taking an education loan is
a good financial decision for your situation.

------------------------------------------------------------

📋 STEP 1: Your Personal Details

Your name [Student]: Riddhima Jain
Your current monthly income (₹) [0]: 150000
Your monthly expenses (₹) [20000]: 25000
Your total savings (₹) [100000]: 45000
Number of dependents (0 if none) [0]: 0

📚 STEP 2: Course Selection

Select a course option:
  1. B.Tech Computer Science @ IIT Bombay
     Fees: ₹875,000 | Duration: 4yr | Avg Salary: ₹1,600,000/yr
  2. MBA @ Private B-School (PGDM)
     Fees: ₹1,800,000 | Duration: 2yr | Avg Salary: ₹850,000/yr
  3. BFA + Film Studies @ Symbiosis
     Fees: ₹2,400,000 | Duration: 4yr | Avg Salary: ₹420,000/yr
  4. Enter custom course details

Select option (1-4) [1]: 4
Course name: MSc in Artificial Intelligence
University/Institute: Nanyang Technological University, Singapore
Total fees (₹) [0]: 4500000
Duration (years) [4]: 1
Expected salary after graduation (₹/year) [0]: 6000000
Field (e.g., Engineering, Arts, Business) [General]: Engineering

============================================================
🔄 Analyzing your situation...
============================================================

============================================================
📊 LOAN ADVISOR RECOMMENDATION
============================================================

👤 Riddhima Jain, here's my analysis of your education loan decision:

📚 Course: MSc in Artificial Intelligence @ Nanyang Technological University, Singapore
💰 Loan Needed: ₹4,455,000
📅 Best Option: SBI Education Loan at 8.5%

💳 Monthly EMI (after graduation): ₹91,401
💵 Total Repayment: ₹5,484,069

============================================================
🎯 RECOMMENDATION: GO
============================================================

Analysis:

  ✅ Positive ROI: Net gain of ₹36,515,931 over 10 years
  ✅ Salary increase: +₹4,200,000/year
  ✅ EMI is 18.3% of post-grad income (affordable)

✅ Based on the numbers, taking this loan makes financial sense.
   The SBI Education Loan at 8.5% offers the best terms.
   Your expected salary increase will more than cover the loan costs.

============================================================

Would you like to analyze another scenario? [y/N]: N

Thank you for using the Education Loan Advisor! Good luck with your decision! 🎓
```

</details>

### HuggingFace Spaces Deployment
1. Create a new Space (Docker SDK)
2. Push the `loan-advisor-env/` directory
3. Set `ENV_BASE_URL` to your Space URL in inference.py
4. The Space will auto-build and expose port 7860

### OpenEnv Validation
```bash
pip install openenv-core
cd loan-advisor-env
openenv validate
```

---

## Baseline Scores

Achieved using the **hybrid inference approach** with `llama-3.1-8b-instant` on Groq:

| Task | Difficulty | Score | Status |
|---|---|---|---|
| task_easy | Easy | **0.900** | ✓ PASS |
| task_medium | Medium | **1.000** | ✓ PASS |
| task_hard | Hard | **1.000** | ✓ PASS |
| **Average** | | **0.967** | ✓ ALL PASSED |

**Key metrics:**
- Total LLM calls: 3 (one per task)
- Average score: 96.7%
- All tasks passed (score ≥ 0.5)

---

## Extending to Other Regions

The environment is globally extensible. To add a USD-based US university task:

1. Add a course entry to `TUITION_LOOKUP` in `server/environment.py`:
```python
"msc_cs_mit": {
    "course": "M.Sc. Computer Science",
    "university": "MIT",
    "total_fees": 110_000,   # USD
    "currency": "USD",
    ...
}
```

2. Add a task to the `TASKS` dict with `"currency": "USD"` and `"currency_symbol": "$"`.

3. Add the task ID to `TASK_ORDER`.

No changes to financial helpers, models, or the FastAPI server are needed.

---

## Project Structure

```
loan-advisor-env/
├── __init__.py              — Package exports
├── models.py                — Typed Pydantic models (Action, Observation, State)
├── client.py                — Python client for the environment server
├── openenv.yaml             — OpenEnv metadata
├── pyproject.toml           — Package configuration
├── inference.py             — Hybrid baseline inference script (MANDATORY)
├── interactive_client.py    — Interactive loan advisor with user input
├── Dockerfile               — Container definition (port 7860)
├── README.md                — This file
└── server/
    ├── __init__.py
    ├── environment.py       — Core logic: tasks, financial helpers, grader
    ├── app.py               — FastAPI server
    └── requirements.txt     — Python dependencies
```

---

## Inference Architecture

The inference script uses a **hybrid approach** that balances efficiency with accuracy:

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID INFERENCE                         │
├─────────────────────────────────────────────────────────────┤
│  PHASE 1: Scripted Actions (No LLM)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. query_info: loan_products                        │   │
│  │ 2. query_info: scholarship_options                  │   │
│  │ 3. query_info: salary_outlook                       │   │
│  │ 4. query_info: user_profile                         │   │
│  │ 5. calculate: roi                                   │   │
│  │ 6. compare: loan_A vs loan_B                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  PHASE 2: LLM Decision (1 call)                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ LLM receives all gathered info and makes            │   │
│  │ final recommendation: GO/NO-GO + loan selection     │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Benefits:                                                  │
│  • Only 3 LLM calls total (1 per task)                     │
│  • All scoring bonuses collected                            │
│  • Robust fallbacks if LLM fails                           │
│  • 96.7% average score                                      │
└─────────────────────────────────────────────────────────────┘
```
