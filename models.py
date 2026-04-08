"""
Typed Pydantic models for the Loan Advisor OpenEnv environment.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LoanAdvisorAction(BaseModel):
    """
    Agent action for the Loan Advisor environment.

    Workflow: query_info → calculate → compare → recommend
    """

    action_type: Literal["query_info", "compare", "calculate", "recommend"] = Field(
        ...,
        description="Type of action the agent wants to take.",
    )

    # --- query_info fields ---
    query_field: Optional[
        Literal[
            "tuition_fees",
            "user_profile",
            "salary_outlook",
            "loan_products",
            "scholarship_options",
        ]
    ] = Field(
        None,
        description=(
            "For query_info: which information to retrieve. "
            "'tuition_fees' returns course cost from lookup table. "
            "'salary_outlook' returns job opportunities and salary ranges for the field. "
            "'scholarship_options' returns available scholarships."
        ),
    )

    # --- compare fields ---
    loan_ids: Optional[list[str]] = Field(
        None,
        description="For compare: list of loan IDs to compare (e.g. ['loan_A', 'loan_B']).",
    )

    # --- calculate fields ---
    calculation_type: Optional[
        Literal["emi", "total_cost", "roi", "affordability", "net_benefit"]
    ] = Field(
        None,
        description=(
            "For calculate: type of calculation to run. "
            "'roi' computes net 10-year benefit (salary increment - loan cost). "
            "'affordability' checks if EMI is within 30% of net monthly income. "
            "'net_benefit' computes comprehensive 10-year cashflow."
        ),
    )
    loan_id: Optional[str] = Field(
        None,
        description="For calculate: the loan ID to calculate for.",
    )

    # --- recommend fields ---
    recommended_decision: Optional[Literal["go", "no_go"]] = Field(
        None,
        description=(
            "'go' = take the loan (education is worth it). "
            "'no_go' = do not take the loan (negative ROI or unaffordable)."
        ),
    )
    recommended_loan_id: Optional[str] = Field(
        None,
        description="For recommend with decision='go': the best loan product ID.",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Agent's explanation for the recommendation (used in partial reward scoring).",
    )

    class Config:
        extra = "allow"


class LoanAdvisorObservation(BaseModel):
    """
    Observation returned by the Loan Advisor environment after each step.
    """

    task_id: str = Field(..., description="Current task identifier.")
    task_description: str = Field(..., description="Natural-language description of the task.")
    action_result: str = Field(..., description="Result of the last action taken by the agent.")
    student_profile_summary: str = Field(
        ..., description="Summary of the student's financial profile."
    )
    course_university: str = Field(
        ..., description="Course and university the student is considering."
    )
    available_loan_ids: list[str] = Field(
        ..., description="List of loan product IDs available in this task."
    )
    steps_taken: int = Field(..., description="Number of steps taken so far in this episode.")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode.")
    episode_done: bool = Field(
        False, description="True when the episode has ended (recommend called or step limit reached)."
    )
    final_reward: Optional[float] = Field(
        None, description="Final episode score in [0.0, 1.0]. Only set when episode_done=True."
    )
    correct_answer: Optional[str] = Field(
        None,
        description=(
            "Revealed at episode end. Format: 'go:loan_A' or 'no_go'. "
            "Null during episode."
        ),
    )

    class Config:
        extra = "allow"


class LoanAdvisorState(BaseModel):
    """
    Internal environment state. Not exposed via the public API.
    Used by the grader to determine final reward.
    """

    episode_id: str
    task_id: str
    step_count: int = 0
    max_steps: int = 8
    done: bool = False

    # Process tracking (used in grader)
    queries_made: list[str] = Field(default_factory=list)
    calculations_done: list[str] = Field(default_factory=list)
    comparison_done: bool = False
    scholarship_queried: bool = False
    roi_calculated: bool = False

    # Agent's final decision
    agent_decision: Optional[str] = None   # "go" or "no_go"
    agent_loan_id: Optional[str] = None

    # Ground truth (used by grader)
    correct_decision: str = "go"           # "go" or "no_go"
    correct_loan_id: Optional[str] = None  # None for no_go tasks

    # Accumulated reward
    reward: float = 0.0
