"""
Core environment logic for the Loan Advisor OpenEnv environment.

Tasks model education loan decisions for Indian students (INR).
Architecture is globally extensible: add non-INR tasks to TASKS dict
without changing any core logic — financial helpers are currency-agnostic.
"""
from __future__ import annotations

import sys
import os
import uuid
from typing import Any, Optional

# Add parent to path so models can be imported when running from server/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LoanAdvisorAction, LoanAdvisorObservation, LoanAdvisorState

# ---------------------------------------------------------------------------
# Tuition + Job Opportunity Lookup Table
# Currency-agnostic — amounts are in the task's native currency (see TASKS).
# Add entries here to support new courses/universities without code changes.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Data sources:
#   Tuition  — IIT/NIT websites, Shiksha.com, Collegedunia (2024-25)
#   Salaries — NIRF Placement Reports, AmbitionBox, LinkedIn Salary Insights
#   Loan rates — SBI/HDFC Credila/Axis Bank published rates (RBI guidelines)
# ---------------------------------------------------------------------------
TUITION_LOOKUP: dict[str, dict] = {
    # IIT Bombay B.Tech CS — NIRF #3, top recruiter salaries ₹15-20L avg
    "btech_cs_iit_bombay": {
        "course": "B.Tech Computer Science",
        "university": "IIT Bombay",
        "total_fees": 875_000,       # ₹8.75L over 4 years (hostel + tuition, 2024-25)
        "duration_years": 4,
        "currency": "INR",
        "field": "Engineering / Technology",
        "avg_starting_salary": 1_600_000,   # NIRF 2024 median placement ₹16L
        "salary_range": "₹12L – ₹30L per year (fresher, NIRF 2024)",
        "data_source": "IIT Bombay fee structure 2024-25, NIRF placement report 2024",
        "job_opportunities": [
            {"role": "Software Engineer", "company_type": "Product / FAANG", "salary_lpa": 20},
            {"role": "Data Scientist", "company_type": "Analytics / AI Firm", "salary_lpa": 16},
            {"role": "Backend Developer", "company_type": "IT Services (TCS/Infosys)", "salary_lpa": 12},
            {"role": "ML Engineer", "company_type": "AI Startup", "salary_lpa": 18},
            {"role": "Quant Analyst", "company_type": "Finance / HFT", "salary_lpa": 25},
        ],
    },

    # Mid-tier MBA — avg 2-yr MBA at private B-school (not IIM), salary bump modest
    "mba_private_bschool": {
        "course": "MBA (General Management)",
        "university": "Private B-School (PGDM, AICTE approved, non-IIM)",
        "total_fees": 1_800_000,     # ₹18L typical for mid-tier (Shiksha avg 2024)
        "duration_years": 2,
        "currency": "INR",
        "field": "Management / Business",
        "avg_starting_salary": 850_000,     # AmbitionBox mid-tier MBA median ₹8.5L
        "salary_range": "₹6L – ₹14L per year (post-MBA, mid-tier school)",
        "data_source": "Shiksha.com MBA fee survey 2024, AmbitionBox salary data",
        "job_opportunities": [
            {"role": "Business Analyst", "company_type": "Consulting Firm", "salary_lpa": 9},
            {"role": "Marketing Manager", "company_type": "FMCG / Retail", "salary_lpa": 8},
            {"role": "Operations Manager", "company_type": "Logistics / Supply Chain", "salary_lpa": 8},
            {"role": "Sales Manager", "company_type": "B2B / SaaS", "salary_lpa": 10},
            {"role": "HR Manager", "company_type": "MNC", "salary_lpa": 7},
        ],
    },

    # Symbiosis BFA — creative arts, low placement rates, entry salaries below ₹5L
    "bfa_symbiosis": {
        "course": "BFA + Film & Visual Communication",
        "university": "Symbiosis Institute of Design, Pune",
        "total_fees": 2_400_000,     # ₹24L over 4 years (Symbiosis fee structure 2024-25)
        "duration_years": 4,
        "currency": "INR",
        "field": "Arts / Creative / Design / Film",
        "avg_starting_salary": 420_000,     # LinkedIn India creative roles, entry-level median
        "salary_range": "₹2.5L – ₹6L per year (entry-level, creative field)",
        "data_source": "Symbiosis Design fee structure 2024, LinkedIn Salary Insights India",
        "job_opportunities": [
            {"role": "Junior Graphic Designer", "company_type": "Design Agency", "salary_lpa": 3},
            {"role": "Content Creator / Video Editor", "company_type": "Media / OTT", "salary_lpa": 3.5},
            {"role": "Assistant Director", "company_type": "Film / Advertising", "salary_lpa": 2.5},
            {"role": "UX Designer (entry)", "company_type": "Tech Startup", "salary_lpa": 5.5},
            {"role": "Social Media Manager", "company_type": "Brand / Agency", "salary_lpa": 4},
        ],
    },

    # --- Global-ready entries (no tasks yet — demonstrates extensibility) ---
    # MS CS at NUS Singapore — global benchmark entry
    "msc_cs_nus": {
        "course": "M.Sc. Computer Science",
        "university": "National University of Singapore (NUS)",
        "total_fees": 4_200_000,     # ~SGD 45K ≈ ₹28L (2024 fee, INR approx)
        "duration_years": 2,
        "currency": "INR",
        "field": "Engineering / Technology",
        "avg_starting_salary": 7_500_000,   # ~SGD 90K ≈ ₹50L (Singapore tech market)
        "salary_range": "SGD 80K – 120K per year (Singapore tech)",
        "data_source": "NUS School of Computing fee schedule 2024, MOM Singapore salary survey",
        "job_opportunities": [
            {"role": "Software Engineer", "company_type": "FAANG / Big Tech (Singapore)", "salary_lpa": 80},
            {"role": "ML Engineer", "company_type": "AI Research Lab", "salary_lpa": 90},
        ],
    },
}

# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------
TASKS: dict[str, dict] = {
    "task_easy": {
        "description": (
            "Rahul wants to pursue a B.Tech in Computer Science at IIT Bombay. "
            "His parents earn ₹80,000/month with ₹30,000 monthly expenses. "
            "He has ₹2,00,000 in savings. Rahul has no current income (full-time student). "
            "Decide: should Rahul take an education loan, and if so, which one?"
        ),
        "student": {
            "name": "Rahul",
            "monthly_income": 0,
            "parent_monthly_income": 80_000,
            "monthly_expenses": 30_000,
            "savings": 200_000,
            "dependents": 0,
            "credit_score": 720,
            "current_annual_income": 0,
        },
        "course_key": "btech_cs_iit_bombay",
        "currency": "INR",
        "currency_symbol": "₹",
        "loan_products": {
            "loan_A": {
                "name": "SBI Education Loan",
                "principal": 800_000,
                "annual_rate_pct": 8.5,
                "tenure_years": 5,
                "moratorium_years": 0,
                "processing_fee": 0,
                "features": "Lowest interest rate. No processing fee. Government-backed.",
            },
            "loan_B": {
                "name": "HDFC Credila Education Loan",
                "principal": 800_000,
                "annual_rate_pct": 12.0,
                "tenure_years": 5,
                "moratorium_years": 0,
                "processing_fee": 2_000,
                "features": "Higher interest rate. ₹2,000 processing fee. Quick disbursement.",
            },
        },
        "scholarship_options": None,
        "max_steps": 8,
        "correct_decision": "go",
        "correct_loan_id": "loan_A",
        "correct_answer_explanation": (
            "GO + loan_A: SBI at 8.5% is significantly cheaper than HDFC at 12%. "
            "Same tenure (5 years), same principal. SBI saves ~₹77K in total interest. "
            "On ₹15L/yr CS salary, EMI (₹16,479/mo) is well within the 30% affordability threshold."
        ),
    },
    "task_medium": {
        "description": (
            "Divya currently earns ₹40,000/month as a marketing executive with ₹25,000 "
            "monthly expenses. She wants to do an MBA at a mid-tier private B-school. "
            "Savings: ₹3,00,000. She has no dependents. "
            "A ₹5,00,000 merit scholarship is available. "
            "Decide: should Divya take an education loan (possibly with scholarship), and which one?"
        ),
        "student": {
            "name": "Divya",
            "monthly_income": 40_000,
            "parent_monthly_income": 0,
            "monthly_expenses": 25_000,
            "savings": 300_000,
            "dependents": 0,
            "credit_score": 710,
            "current_annual_income": 480_000,
        },
        "course_key": "mba_private_bschool",
        "currency": "INR",
        "currency_symbol": "₹",
        "loan_products": {
            "loan_A": {
                "name": "SBI Scholar Loan",
                "principal": 2_000_000,
                "annual_rate_pct": 11.5,
                "tenure_years": 7,
                "moratorium_years": 2,
                "processing_fee": 5_000,
                "features": "Moratorium during MBA. Can be reduced to ₹15L with scholarship.",
            },
            "loan_B": {
                "name": "Axis Bank Education Loan",
                "principal": 2_000_000,
                "annual_rate_pct": 13.0,
                "tenure_years": 5,
                "moratorium_years": 0,
                "processing_fee": 3_000,
                "features": "No moratorium. Fixed rate.",
            },
            "loan_C": {
                "name": "NBFC QuickEdu Loan",
                "principal": 2_000_000,
                "annual_rate_pct": 18.0,
                "tenure_years": 3,
                "moratorium_years": 0,
                "processing_fee": 1_000,
                "features": "Fastest approval. Very high rate.",
            },
        },
        "scholarship_options": {
            "merit_scholarship": {
                "name": "B-School Merit Scholarship",
                "amount": 500_000,
                "description": "₹5,00,000 merit scholarship covering 25% of tuition. Reduces net loan to ₹15,00,000.",
                "adjusted_principal": 1_500_000,
            }
        },
        "max_steps": 10,
        "correct_decision": "go",
        "correct_loan_id": "loan_A",
        "correct_answer_explanation": (
            "GO + loan_A with scholarship: scholarship reduces principal to ₹15L. "
            "Post-MBA salary of ₹8L/yr (vs current ₹4.8L/yr) = ₹3.2L increment. "
            "Over 7 years, net ROI is positive. loan_A has lowest total cost with moratorium."
        ),
    },
    "task_hard": {
        "description": (
            "Meera earns ₹30,000/month as a junior content writer. Monthly expenses ₹20,000. "
            "She supports aging parents (2 dependents). Savings: ₹1,50,000. "
            "She wants to pursue BFA + Film Studies at Symbiosis International. "
            "A bond scholarship is available: covers ₹20L of fees, but obligates her to work "
            "at a media company for 3 years at ₹3.5L/year (market rate: ₹4.5L/year). "
            "Arts graduates earn ~₹4.5L/year starting — less than her current income after "
            "a 3-year study gap. "
            "Decide: should Meera take an education loan for this course?"
        ),
        "student": {
            "name": "Meera",
            "monthly_income": 30_000,
            "parent_monthly_income": 0,
            "monthly_expenses": 20_000,
            "savings": 150_000,
            "dependents": 2,
            "credit_score": 680,
            "current_annual_income": 360_000,
        },
        "course_key": "bfa_symbiosis",
        "currency": "INR",
        "currency_symbol": "₹",
        "loan_products": {
            "loan_A": {
                "name": "Bank of Baroda Education Loan",
                "principal": 2_500_000,
                "annual_rate_pct": 12.0,
                "tenure_years": 10,
                "moratorium_years": 3,
                "processing_fee": 8_000,
                "features": "Moratorium during study. Requires collateral.",
            },
            "loan_B": {
                "name": "Bond Scholarship Loan",
                "principal": 500_000,
                "annual_rate_pct": 9.0,
                "tenure_years": 5,
                "moratorium_years": 3,
                "processing_fee": 0,
                "features": (
                    "Covers ₹20L via bond scholarship + ₹5L personal loan. "
                    "OBLIGATION: 3 years at media company at ₹3.5L/yr "
                    "(₹3L opportunity cost vs market ₹4.5L/yr)."
                ),
            },
        },
        "scholarship_options": {
            "bond_scholarship": {
                "name": "Media Bond Scholarship",
                "amount": 2_000_000,
                "description": (
                    "Covers ₹20L of the ₹25L tuition. "
                    "Obligation: 3-year bond at ₹3.5L/yr salary (market: ₹4.5L/yr). "
                    "Opportunity cost: ₹3,00,000 over 3 years."
                ),
                "adjusted_principal": 500_000,
                "opportunity_cost": 300_000,
            }
        },
        "max_steps": 12,
        "correct_decision": "no_go",
        "correct_loan_id": None,
        "correct_answer_explanation": (
            "NO-GO: Arts field salary (₹4.5L/yr) is BELOW Meera's current income (₹3.6L/yr). "
            "After 3-year study gap, she loses ₹10.8L in income plus loan repayment burden. "
            "2 dependents mean EMI (₹28K+/month for loan_A) exceeds 30% of projected income. "
            "Bond scholarship still leaves negative ROI due to opportunity cost + salary cut."
        ),
    },
}

TASK_ORDER = ["task_easy", "task_medium", "task_hard"]


# ---------------------------------------------------------------------------
# Financial Helper Functions (pure Python stdlib — no numpy/pandas)
# All amounts are in the task's native currency (currency-agnostic).
# ---------------------------------------------------------------------------

def calculate_emi(principal: float, annual_rate_pct: float, tenure_years: int) -> float:
    """Standard EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)."""
    if principal <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    n = tenure_years * 12
    if r == 0 or annual_rate_pct == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def calculate_total_cost(
    principal: float,
    annual_rate_pct: float,
    tenure_years: int,
    processing_fee: float = 0,
    moratorium_years: int = 0,
) -> float:
    """Total repayment = EMI × repayment months + interest during moratorium + fee."""
    if principal <= 0:
        return processing_fee
    r_monthly = annual_rate_pct / 100.0 / 12.0
    moratorium_months = moratorium_years * 12
    moratorium_interest = principal * r_monthly * moratorium_months
    # Principal after moratorium (simple interest accrual added to principal)
    principal_after = principal + moratorium_interest
    repayment_months = tenure_years * 12
    emi = calculate_emi(principal_after, annual_rate_pct, tenure_years)
    total_repayment = emi * repayment_months
    return total_repayment + processing_fee


def calculate_roi(
    current_annual_income: float,
    post_grad_annual_salary: float,
    loan_total_cost: float,
    years: int = 10,
) -> dict:
    """
    Net ROI over `years` years.
    ROI = cumulative salary increment - total loan cost.
    Positive = education is financially worthwhile.
    """
    annual_increment = post_grad_annual_salary - current_annual_income
    cumulative_increment = annual_increment * years
    net_roi = cumulative_increment - loan_total_cost
    return {
        "current_annual_income": current_annual_income,
        "post_grad_annual_salary": post_grad_annual_salary,
        "annual_salary_increment": annual_increment,
        "cumulative_increment_10yr": cumulative_increment,
        "total_loan_cost": loan_total_cost,
        "net_roi_10yr": net_roi,
        "roi_positive": net_roi > 0,
    }


def calculate_affordability(
    emi: float,
    monthly_income: float,
    monthly_expenses: float,
    threshold_pct: float = 30.0,
) -> dict:
    """
    Affordability check: EMI should be ≤ threshold_pct% of net disposable income.
    Net disposable = monthly_income - monthly_expenses.
    Rule of thumb: 20–30% is sustainable; >40% is risky.
    """
    disposable = monthly_income - monthly_expenses
    max_affordable_emi = disposable * threshold_pct / 100.0
    affordable = emi <= max_affordable_emi
    return {
        "emi": round(emi, 2),
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "disposable_income": disposable,
        "threshold_pct": threshold_pct,
        "max_affordable_emi": round(max_affordable_emi, 2),
        "affordable": affordable,
        "emi_to_disposable_ratio_pct": round((emi / disposable * 100) if disposable > 0 else 999, 1),
    }


def calculate_savings_comparison(
    loan_id_a: str,
    loan_id_b: str,
    loan_products: dict,
) -> dict:
    """Compare total costs of two loan products. Returns who is cheaper and by how much."""
    a = loan_products[loan_id_a]
    b = loan_products[loan_id_b]
    cost_a = calculate_total_cost(
        a["principal"], a["annual_rate_pct"], a["tenure_years"],
        a.get("processing_fee", 0), a.get("moratorium_years", 0),
    )
    cost_b = calculate_total_cost(
        b["principal"], b["annual_rate_pct"], b["tenure_years"],
        b.get("processing_fee", 0), b.get("moratorium_years", 0),
    )
    cheaper = loan_id_a if cost_a <= cost_b else loan_id_b
    savings = abs(cost_a - cost_b)
    return {
        loan_id_a: {"name": a["name"], "total_cost": round(cost_a, 2)},
        loan_id_b: {"name": b["name"], "total_cost": round(cost_b, 2)},
        "cheaper": cheaper,
        "savings": round(savings, 2),
    }


def format_currency(amount: float, symbol: str = "₹") -> str:
    """Format a number as currency with Indian-style comma separation."""
    return f"{symbol}{amount:,.0f}"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

try:
    from openm.core.environment import Environment as _BaseEnvironment  # type: ignore
    _ENV_BASE = _BaseEnvironment
except ImportError:
    try:
        from openenv.core.environment import Environment as _BaseEnvironment  # type: ignore
        _ENV_BASE = _BaseEnvironment
    except ImportError:
        class _ENV_BASE:  # type: ignore
            pass


class LoanAdvisorEnvironment(_ENV_BASE):  # type: ignore[misc]
    """
    OpenEnv-compatible environment for education loan decision-making.

    Episodes cycle through tasks: easy → medium → hard → easy → …
    Each reset() starts a fresh episode. No state persists between episodes.
    """

    def __init__(self) -> None:
        self._state: Optional[LoanAdvisorState] = None
        self._task_index: int = 0  # cycles through TASK_ORDER

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, **kwargs: Any) -> LoanAdvisorObservation:
        """
        Start a new episode.

        Args:
            task_id: Optional task to run. If omitted, cycles easy→medium→hard→easy.

        Returns:
            Initial LoanAdvisorObservation.
        """
        if task_id and task_id in TASKS:
            chosen_task_id = task_id
        else:
            chosen_task_id = TASK_ORDER[self._task_index % len(TASK_ORDER)]
            self._task_index += 1

        task = TASKS[chosen_task_id]
        self._state = LoanAdvisorState(
            episode_id=str(uuid.uuid4()),
            task_id=chosen_task_id,
            max_steps=task["max_steps"],
            correct_decision=task["correct_decision"],
            correct_loan_id=task.get("correct_loan_id"),
        )

        return self._make_observation(
            action_result=(
                "Episode started. Gather information before recommending.\n"
                "Suggested workflow: query_info (tuition_fees, user_profile, salary_outlook, "
                "loan_products, scholarship_options) → calculate (roi, affordability, total_cost) "
                "→ compare → recommend (go/no_go)."
            )
        )

    def step(
        self, action: LoanAdvisorAction
    ) -> tuple[LoanAdvisorObservation, float, bool, dict]:
        """
        Execute one agent action.

        Returns:
            (observation, reward, done, info)
        """
        if self._state is None or self._state.done:
            obs = LoanAdvisorObservation(
                task_id="none",
                task_description="No active episode. Call reset() first.",
                action_result="No active episode.",
                student_profile_summary="",
                course_university="",
                available_loan_ids=[],
                steps_taken=0,
                max_steps=0,
                episode_done=True,
            )
            return obs, 0.0, True, {}

        self._state.step_count += 1
        state = self._state
        task = TASKS[state.task_id]
        reward = 0.0

        # --- Dispatch ---
        action_type = action.action_type

        if action_type == "query_info":
            result, reward = self._handle_query_info(action, task, state)

        elif action_type == "compare":
            result, reward = self._handle_compare(action, task, state)

        elif action_type == "calculate":
            result, reward = self._handle_calculate(action, task, state)

        elif action_type == "recommend":
            result, reward, done = self._handle_recommend(action, task, state)
            # _handle_recommend already sets state.reward = final_score; don't add again
            obs = self._make_observation(action_result=result)
            return obs, reward, done, {
                "correct_decision": task["correct_decision"],
                "correct_loan_id": task.get("correct_loan_id"),
            }

        else:
            result = f"Unknown action_type '{action_type}'. Valid: query_info, compare, calculate, recommend."

        # Check step limit
        state.reward += reward
        done = state.step_count >= state.max_steps
        if done and not state.done:
            state.done = True
            result += f"\n[Step limit of {state.max_steps} reached. Episode ended without recommendation. Score: {state.reward:.3f}]"

        obs = self._make_observation(action_result=result)
        return obs, reward, done, {}

    def state(self) -> dict:
        """Return current internal state for debugging/monitoring."""
        if self._state is None:
            return {"status": "not_started"}
        return self._state.model_dump()

    # ------------------------------------------------------------------
    # Action Handlers
    # ------------------------------------------------------------------

    def _handle_query_info(
        self, action: LoanAdvisorAction, task: dict, state: LoanAdvisorState
    ) -> tuple[str, float]:
        qf = action.query_field
        sym = task["currency_symbol"]
        course_data = TUITION_LOOKUP.get(task["course_key"], {})
        reward = 0.0

        if qf == "tuition_fees":
            fees = course_data.get("total_fees", 0)
            duration = course_data.get("duration_years", "N/A")
            result = (
                f"Course: {course_data.get('course')} at {course_data.get('university')}\n"
                f"Total Tuition: {format_currency(fees, sym)}\n"
                f"Duration: {duration} years\n"
                f"Field: {course_data.get('field')}"
            )

        elif qf == "user_profile":
            s = task["student"]
            inc = s.get("monthly_income", 0)
            p_inc = s.get("parent_monthly_income", 0)
            result = (
                f"Student: {s['name']}\n"
                f"Monthly Income: {format_currency(inc, sym)}\n"
                f"Parent Monthly Income: {format_currency(p_inc, sym)}\n"
                f"Monthly Expenses: {format_currency(s['monthly_expenses'], sym)}\n"
                f"Savings: {format_currency(s['savings'], sym)}\n"
                f"Dependents: {s['dependents']}\n"
                f"Credit Score: {s['credit_score']}\n"
                f"Current Annual Income: {format_currency(s['current_annual_income'], sym)}"
            )

        elif qf == "salary_outlook":
            avg_sal = course_data.get("avg_starting_salary", 0)
            sal_range = course_data.get("salary_range", "N/A")
            jobs = course_data.get("job_opportunities", [])
            job_lines = "\n".join(
                f"  • {j['role']} at {j['company_type']}: ~{sym}{j['salary_lpa']}L/yr"
                for j in jobs
            )
            result = (
                f"Field: {course_data.get('field')}\n"
                f"Average Starting Salary: {format_currency(avg_sal, sym)}/year\n"
                f"Salary Range: {sal_range}\n"
                f"Typical Job Opportunities:\n{job_lines}"
            )

        elif qf == "loan_products":
            lines = []
            for lid, lp in task["loan_products"].items():
                lines.append(
                    f"{lid} — {lp['name']}:\n"
                    f"  Principal: {format_currency(lp['principal'], sym)}, "
                    f"Rate: {lp['annual_rate_pct']}% p.a., "
                    f"Tenure: {lp['tenure_years']}yr, "
                    f"Moratorium: {lp.get('moratorium_years', 0)}yr, "
                    f"Processing Fee: {format_currency(lp.get('processing_fee', 0), sym)}\n"
                    f"  Features: {lp.get('features', '')}"
                )
            result = "Available Loan Products:\n" + "\n\n".join(lines)

        elif qf == "scholarship_options":
            scholarships = task.get("scholarship_options")
            if not scholarships:
                result = "No scholarships available for this task."
            else:
                lines = []
                for sid, sc in scholarships.items():
                    lines.append(
                        f"{sc['name']}: {format_currency(sc['amount'], sym)}\n"
                        f"  {sc['description']}"
                    )
                result = "Available Scholarships:\n" + "\n".join(lines)
            state.scholarship_queried = True

        else:
            return (
                f"Unknown query_field '{qf}'. Valid: tuition_fees, user_profile, "
                "salary_outlook, loan_products, scholarship_options.",
                0.0,
            )

        if qf and qf not in state.queries_made:
            state.queries_made.append(qf)
            reward = 0.05

        return result, reward

    def _handle_compare(
        self, action: LoanAdvisorAction, task: dict, state: LoanAdvisorState
    ) -> tuple[str, float]:
        loan_ids = action.loan_ids
        sym = task["currency_symbol"]
        if not loan_ids or len(loan_ids) < 2:
            return "compare requires at least 2 loan_ids.", 0.0

        valid_ids = [lid for lid in loan_ids if lid in task["loan_products"]]
        if len(valid_ids) < 2:
            available = list(task["loan_products"].keys())
            return f"Invalid loan_ids. Available: {available}", 0.0

        cmp = calculate_savings_comparison(valid_ids[0], valid_ids[1], task["loan_products"])
        a_name = task["loan_products"][valid_ids[0]]["name"]
        b_name = task["loan_products"][valid_ids[1]]["name"]
        result = (
            f"Comparison: {valid_ids[0]} ({a_name}) vs {valid_ids[1]} ({b_name})\n"
            f"  {valid_ids[0]} total cost: {format_currency(cmp[valid_ids[0]]['total_cost'], sym)}\n"
            f"  {valid_ids[1]} total cost: {format_currency(cmp[valid_ids[1]]['total_cost'], sym)}\n"
            f"  Cheaper: {cmp['cheaper']} | Saves: {format_currency(cmp['savings'], sym)}"
        )

        reward = 0.0
        if not state.comparison_done:
            state.comparison_done = True
            reward = 0.10

        return result, reward

    def _handle_calculate(
        self, action: LoanAdvisorAction, task: dict, state: LoanAdvisorState
    ) -> tuple[str, float]:
        calc_type = action.calculation_type
        loan_id = action.loan_id
        sym = task["currency_symbol"]
        student = task["student"]
        course_data = TUITION_LOOKUP.get(task["course_key"], {})
        reward = 0.0

        if calc_type == "emi":
            if not loan_id or loan_id not in task["loan_products"]:
                return f"Specify a valid loan_id for emi. Available: {list(task['loan_products'].keys())}", 0.0
            lp = task["loan_products"][loan_id]
            emi = calculate_emi(lp["principal"], lp["annual_rate_pct"], lp["tenure_years"])
            result = (
                f"EMI for {loan_id} ({lp['name']}):\n"
                f"  Principal: {format_currency(lp['principal'], sym)}\n"
                f"  Rate: {lp['annual_rate_pct']}% p.a.\n"
                f"  Tenure: {lp['tenure_years']} years\n"
                f"  Monthly EMI: {format_currency(emi, sym)}"
            )

        elif calc_type == "total_cost":
            if not loan_id or loan_id not in task["loan_products"]:
                return f"Specify a valid loan_id for total_cost. Available: {list(task['loan_products'].keys())}", 0.0
            lp = task["loan_products"][loan_id]
            total = calculate_total_cost(
                lp["principal"], lp["annual_rate_pct"], lp["tenure_years"],
                lp.get("processing_fee", 0), lp.get("moratorium_years", 0),
            )
            result = (
                f"Total Cost for {loan_id} ({lp['name']}):\n"
                f"  Principal: {format_currency(lp['principal'], sym)}\n"
                f"  Total Repayment (incl. interest + fee): {format_currency(total, sym)}\n"
                f"  Interest Paid: {format_currency(total - lp['principal'], sym)}"
            )

        elif calc_type == "roi":
            # Use best loan (loan_A) for ROI calculation if no loan_id specified
            lid = loan_id if loan_id and loan_id in task["loan_products"] else list(task["loan_products"].keys())[0]
            lp = task["loan_products"][lid]
            total_cost = calculate_total_cost(
                lp["principal"], lp["annual_rate_pct"], lp["tenure_years"],
                lp.get("processing_fee", 0), lp.get("moratorium_years", 0),
            )
            post_sal = course_data.get("avg_starting_salary", 0)
            curr_income = student.get("current_annual_income", 0)
            roi = calculate_roi(curr_income, post_sal, total_cost)

            state.roi_calculated = True
            result = (
                f"ROI Analysis (10 years) for {lid}:\n"
                f"  Current Annual Income: {format_currency(roi['current_annual_income'], sym)}\n"
                f"  Post-Graduation Salary: {format_currency(roi['post_grad_annual_salary'], sym)}\n"
                f"  Annual Salary Increment: {format_currency(roi['annual_salary_increment'], sym)}\n"
                f"  Cumulative 10-yr Increment: {format_currency(roi['cumulative_increment_10yr'], sym)}\n"
                f"  Total Loan Cost (no scholarship): {format_currency(roi['total_loan_cost'], sym)}\n"
                f"  Net ROI (10yr, no scholarship): {format_currency(roi['net_roi_10yr'], sym)}\n"
                f"  ROI Positive (no scholarship): {'YES' if roi['roi_positive'] else 'NO — Education may not be financially worthwhile'}"
            )

            # If scholarship is available, show scholarship-adjusted ROI alongside
            scholarships = task.get("scholarship_options")
            if scholarships:
                for _, sc in scholarships.items():
                    adj_principal = sc.get("adjusted_principal")
                    if adj_principal and adj_principal < lp["principal"]:
                        adj_total_cost = calculate_total_cost(
                            adj_principal, lp["annual_rate_pct"], lp["tenure_years"],
                            lp.get("processing_fee", 0), lp.get("moratorium_years", 0),
                        )
                        adj_roi = calculate_roi(curr_income, post_sal, adj_total_cost)
                        result += (
                            f"\n\n  With {sc['name']} (principal reduced to {format_currency(adj_principal, sym)}):\n"
                            f"  Total Loan Cost (with scholarship): {format_currency(adj_total_cost, sym)}\n"
                            f"  Net ROI (10yr, with scholarship): {format_currency(adj_roi['net_roi_10yr'], sym)}\n"
                            f"  ROI Positive (with scholarship): {'YES' if adj_roi['roi_positive'] else 'NO'}\n"
                            f"  RECOMMENDATION: Apply the scholarship to {lid} for best ROI."
                        )

        elif calc_type == "affordability":
            if not loan_id or loan_id not in task["loan_products"]:
                return f"Specify a valid loan_id for affordability. Available: {list(task['loan_products'].keys())}", 0.0
            lp = task["loan_products"][loan_id]
            # For students, use parent income; for working professionals, use own income
            effective_income = student.get("monthly_income", 0) or student.get("parent_monthly_income", 0)
            post_grad_monthly = course_data.get("avg_starting_salary", 0) / 12
            emi = calculate_emi(lp["principal"], lp["annual_rate_pct"], lp["tenure_years"])
            # Check affordability on post-graduation income
            aff = calculate_affordability(emi, post_grad_monthly, student["monthly_expenses"])
            result = (
                f"Affordability Check for {loan_id} ({lp['name']}) on post-grad income:\n"
                f"  Monthly EMI: {format_currency(aff['emi'], sym)}\n"
                f"  Post-Grad Monthly Income: {format_currency(post_grad_monthly, sym)}\n"
                f"  Monthly Expenses: {format_currency(aff['monthly_expenses'], sym)}\n"
                f"  Disposable Income: {format_currency(aff['disposable_income'], sym)}\n"
                f"  Max Affordable EMI (30%): {format_currency(aff['max_affordable_emi'], sym)}\n"
                f"  EMI/Disposable Ratio: {aff['emi_to_disposable_ratio_pct']}%\n"
                f"  AFFORDABLE: {'YES' if aff['affordable'] else 'NO — EMI exceeds 30% of disposable income'}"
            )

        elif calc_type == "net_benefit":
            # 10-year comprehensive cashflow
            lid = loan_id if loan_id and loan_id in task["loan_products"] else list(task["loan_products"].keys())[0]
            lp = task["loan_products"][lid]
            total_cost = calculate_total_cost(
                lp["principal"], lp["annual_rate_pct"], lp["tenure_years"],
                lp.get("processing_fee", 0), lp.get("moratorium_years", 0),
            )
            post_sal = course_data.get("avg_starting_salary", 0)
            curr_income = student.get("current_annual_income", 0)
            study_years = course_data.get("duration_years", 2)
            # Income gap during study (no income if student)
            income_gap = curr_income * study_years
            net_benefit = (post_sal - curr_income) * (10 - study_years) - total_cost - income_gap
            state.roi_calculated = True
            result = (
                f"10-Year Net Benefit Analysis for {lid}:\n"
                f"  Income gap during {study_years}-yr study: -{format_currency(income_gap, sym)}\n"
                f"  Post-grad salary gain ({10 - study_years}yr): +{format_currency((post_sal - curr_income) * (10 - study_years), sym)}\n"
                f"  Total loan repayment: -{format_currency(total_cost, sym)}\n"
                f"  NET 10-YEAR BENEFIT: {format_currency(net_benefit, sym)} "
                f"({'POSITIVE' if net_benefit > 0 else 'NEGATIVE'})"
            )

        else:
            return (
                f"Unknown calculation_type '{calc_type}'. "
                "Valid: emi, total_cost, roi, affordability, net_benefit.",
                0.0,
            )

        key = f"{calc_type}_{loan_id or 'default'}"
        if key not in state.calculations_done:
            state.calculations_done.append(key)
            reward = 0.08

        return result, reward

    def _handle_recommend(
        self, action: LoanAdvisorAction, task: dict, state: LoanAdvisorState
    ) -> tuple[str, float, bool]:
        state.agent_decision = action.recommended_decision
        state.agent_loan_id = action.recommended_loan_id
        state.done = True

        final_score = self._grade(state)
        state.reward = final_score

        correct_ans = task["correct_decision"]
        if task.get("correct_loan_id"):
            correct_ans += f":{task['correct_loan_id']}"

        result = (
            f"Recommendation recorded: decision={action.recommended_decision}, "
            f"loan={action.recommended_loan_id}\n"
            f"Episode complete. Score: {final_score:.3f}\n"
            f"Correct answer: {correct_ans}\n"
            f"Explanation: {task['correct_answer_explanation']}"
        )
        return result, final_score, True

    # ------------------------------------------------------------------
    # Grader
    # ------------------------------------------------------------------

    def _grade(self, state: LoanAdvisorState) -> float:
        """
        Deterministic grader. Returns float in [0.0, 1.0].

        Correct decision + correct loan (if applicable):
          base = 0.60
          + 0.10 if comparison done
          + 0.10 if ROI calculated
          + 0.10 if scholarship queried (relevant tasks)
          + 0.10 if ≥3 unique queries made
          → max 1.0

        Correct decision but wrong loan (go tasks only):
          base = 0.40 + same process bonuses → max ~0.70

        Wrong decision:
          base = 0.0
          + 0.10 if comparison done
          + 0.10 if ROI calculated
          + 0.10 if ≥2 queries made
          → max 0.30
        """
        task = TASKS[state.task_id]
        agent_decision = state.agent_decision
        correct_decision = state.correct_decision
        correct_loan = state.correct_loan_id

        # Process quality bonuses
        def process_bonus(full: bool) -> float:
            bonus = 0.0
            if state.comparison_done:
                bonus += 0.10
            if state.roi_calculated or any("roi" in c for c in state.calculations_done):
                bonus += 0.10
            if full and state.scholarship_queried and task.get("scholarship_options"):
                bonus += 0.10
            if full and len(state.queries_made) >= 3:
                bonus += 0.10
            elif not full and len(state.queries_made) >= 2:
                bonus += 0.10
            return bonus

        if agent_decision == correct_decision:
            if correct_decision == "no_go":
                # no_go: loan_id irrelevant
                score = 0.60 + process_bonus(full=True)
            else:
                # go: check if loan_id matches
                if state.agent_loan_id == correct_loan:
                    score = 0.60 + process_bonus(full=True)
                else:
                    score = 0.40 + process_bonus(full=True)
        else:
            score = process_bonus(full=False)

        return min(1.0, max(0.0, score))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self, action_result: str) -> LoanAdvisorObservation:
        state = self._state
        if state is None:
            raise RuntimeError("No active episode. Call reset() first.")

        task = TASKS[state.task_id]
        course_data = TUITION_LOOKUP.get(task["course_key"], {})
        s = task["student"]

        student_summary = (
            f"{s['name']} | Income: {task['currency_symbol']}{s['monthly_income']:,}/mo "
            f"(Parents: {task['currency_symbol']}{s.get('parent_monthly_income', 0):,}/mo) | "
            f"Expenses: {task['currency_symbol']}{s['monthly_expenses']:,}/mo | "
            f"Savings: {task['currency_symbol']}{s['savings']:,} | "
            f"Dependents: {s['dependents']}"
        )

        correct_answer = None
        final_reward = None
        if state.done:
            final_reward = state.reward
            correct_answer = task["correct_decision"]
            if task.get("correct_loan_id"):
                correct_answer += f":{task['correct_loan_id']}"

        return LoanAdvisorObservation(
            task_id=state.task_id,
            task_description=task["description"],
            action_result=action_result,
            student_profile_summary=student_summary,
            course_university=f"{course_data.get('course', '')} @ {course_data.get('university', '')}",
            available_loan_ids=list(task["loan_products"].keys()),
            steps_taken=state.step_count,
            max_steps=state.max_steps,
            episode_done=state.done,
            final_reward=final_reward,
            correct_answer=correct_answer,
        )