"""
Tests for the Loan Advisor OpenEnv environment.

Covers:
  - Environment reset and state management
  - All action types (query_info, compare, calculate, recommend)
  - Grading logic (correct/wrong decisions, process bonuses)
  - Episode boundaries and step limits
  - Financial calculation accuracy
  - Deterministic and reproducible grading
"""
import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LoanAdvisorAction, LoanAdvisorObservation, LoanAdvisorState
from server.environment import (
    LoanAdvisorEnvironment,
    TASKS,
    TASK_ORDER,
    calculate_emi,
    calculate_total_cost,
    calculate_roi,
    calculate_affordability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh environment instance."""
    return LoanAdvisorEnvironment()


# ---------------------------------------------------------------------------
# Reset Tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="task_easy")
        assert isinstance(obs, LoanAdvisorObservation)
        assert obs.task_id == "task_easy"
        assert obs.steps_taken == 0
        assert obs.episode_done is False
        assert obs.final_reward is None

    def test_reset_cycles_tasks(self, env):
        obs1 = env.reset()
        obs2 = env.reset()
        obs3 = env.reset()
        assert obs1.task_id == "task_easy"
        assert obs2.task_id == "task_medium"
        assert obs3.task_id == "task_hard"

    def test_reset_explicit_task(self, env):
        obs = env.reset(task_id="task_hard")
        assert obs.task_id == "task_hard"
        assert "Meera" in obs.student_profile_summary

    def test_reset_clears_state(self, env):
        env.reset(task_id="task_easy")
        env.step(LoanAdvisorAction(action_type="query_info", query_field="loan_products"))
        obs2 = env.reset(task_id="task_easy")
        assert obs2.steps_taken == 0
        assert obs2.episode_done is False

    def test_reset_has_loan_ids(self, env):
        obs = env.reset(task_id="task_easy")
        assert len(obs.available_loan_ids) >= 2
        assert "loan_A" in obs.available_loan_ids

    def test_all_tasks_exist(self):
        for task_id in TASK_ORDER:
            assert task_id in TASKS


# ---------------------------------------------------------------------------
# Action Tests — query_info
# ---------------------------------------------------------------------------

class TestQueryInfo:
    def test_query_loan_products(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="query_info", query_field="loan_products")
        obs, reward, done, info = env.step(action)
        assert reward == 0.05  # first unique query
        assert "loan_A" in obs.action_result
        assert done is False

    def test_query_salary_outlook(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="query_info", query_field="salary_outlook")
        obs, reward, done, info = env.step(action)
        assert reward == 0.05
        assert "Software Engineer" in obs.action_result

    def test_query_user_profile(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="query_info", query_field="user_profile")
        obs, reward, done, info = env.step(action)
        assert reward == 0.05
        assert "Rahul" in obs.action_result

    def test_query_scholarship_none(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="query_info", query_field="scholarship_options")
        obs, reward, done, info = env.step(action)
        assert "No scholarships" in obs.action_result

    def test_query_scholarship_exists(self, env):
        env.reset(task_id="task_medium")
        action = LoanAdvisorAction(action_type="query_info", query_field="scholarship_options")
        obs, reward, done, info = env.step(action)
        assert "Merit Scholarship" in obs.action_result

    def test_duplicate_query_no_reward(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="query_info", query_field="loan_products")
        obs1, r1, _, _ = env.step(action)
        obs2, r2, _, _ = env.step(action)
        assert r1 == 0.05
        assert r2 == 0.0  # duplicate

    def test_invalid_query_field(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="query_info", query_field=None)
        obs, reward, done, info = env.step(action)
        assert reward == 0.0


# ---------------------------------------------------------------------------
# Action Tests — compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_two_loans(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="compare", loan_ids=["loan_A", "loan_B"])
        obs, reward, done, info = env.step(action)
        assert reward == 0.10
        assert "Cheaper" in obs.action_result

    def test_compare_duplicate_no_reward(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="compare", loan_ids=["loan_A", "loan_B"])
        _, r1, _, _ = env.step(action)
        _, r2, _, _ = env.step(action)
        assert r1 == 0.10
        assert r2 == 0.0

    def test_compare_invalid_ids(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="compare", loan_ids=["loan_X", "loan_Y"])
        obs, reward, _, _ = env.step(action)
        assert reward == 0.0
        assert "Invalid" in obs.action_result


# ---------------------------------------------------------------------------
# Action Tests — calculate
# ---------------------------------------------------------------------------

class TestCalculate:
    def test_calculate_roi(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="calculate", calculation_type="roi", loan_id="loan_A")
        obs, reward, done, info = env.step(action)
        assert reward == 0.08
        assert "ROI" in obs.action_result

    def test_calculate_emi(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="calculate", calculation_type="emi", loan_id="loan_A")
        obs, reward, done, info = env.step(action)
        assert reward == 0.08
        assert "EMI" in obs.action_result

    def test_calculate_affordability(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="calculate", calculation_type="affordability", loan_id="loan_A")
        obs, reward, done, info = env.step(action)
        assert reward == 0.08
        assert "AFFORDABLE" in obs.action_result

    def test_calculate_net_benefit(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="calculate", calculation_type="net_benefit", loan_id="loan_A")
        obs, reward, done, info = env.step(action)
        assert reward == 0.08
        assert "Net Benefit" in obs.action_result or "NET" in obs.action_result

    def test_calculate_total_cost(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(action_type="calculate", calculation_type="total_cost", loan_id="loan_A")
        obs, reward, done, info = env.step(action)
        assert reward == 0.08
        assert "Total Cost" in obs.action_result


# ---------------------------------------------------------------------------
# Action Tests — recommend
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_correct_go_correct_loan(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(
            action_type="recommend",
            recommended_decision="go",
            recommended_loan_id="loan_A",
            reasoning="Positive ROI",
        )
        obs, reward, done, info = env.step(action)
        assert done is True
        assert obs.episode_done is True
        assert reward >= 0.60  # base for correct decision
        assert obs.final_reward is not None
        assert obs.correct_answer is not None

    def test_correct_go_wrong_loan(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(
            action_type="recommend",
            recommended_decision="go",
            recommended_loan_id="loan_B",
            reasoning="Testing",
        )
        obs, reward, done, info = env.step(action)
        assert done is True
        assert reward >= 0.40  # correct decision, wrong loan
        assert reward < 0.60

    def test_wrong_decision(self, env):
        env.reset(task_id="task_easy")
        action = LoanAdvisorAction(
            action_type="recommend",
            recommended_decision="no_go",
            reasoning="Wrong decision",
        )
        obs, reward, done, info = env.step(action)
        assert done is True
        assert reward <= 0.30  # wrong decision max

    def test_correct_no_go(self, env):
        env.reset(task_id="task_hard")
        action = LoanAdvisorAction(
            action_type="recommend",
            recommended_decision="no_go",
            reasoning="Negative ROI",
        )
        obs, reward, done, info = env.step(action)
        assert done is True
        assert reward >= 0.60

    def test_wrong_go_on_nogo_task(self, env):
        env.reset(task_id="task_hard")
        action = LoanAdvisorAction(
            action_type="recommend",
            recommended_decision="go",
            recommended_loan_id="loan_A",
            reasoning="Wrong",
        )
        obs, reward, done, info = env.step(action)
        assert done is True
        assert reward <= 0.30


# ---------------------------------------------------------------------------
# Grading — process bonuses
# ---------------------------------------------------------------------------

class TestGradingBonuses:
    def test_full_research_bonus(self, env):
        """Complete research workflow should yield max score."""
        env.reset(task_id="task_easy")
        # 3+ queries
        env.step(LoanAdvisorAction(action_type="query_info", query_field="loan_products"))
        env.step(LoanAdvisorAction(action_type="query_info", query_field="salary_outlook"))
        env.step(LoanAdvisorAction(action_type="query_info", query_field="user_profile"))
        # ROI
        env.step(LoanAdvisorAction(action_type="calculate", calculation_type="roi", loan_id="loan_A"))
        # Compare
        env.step(LoanAdvisorAction(action_type="compare", loan_ids=["loan_A", "loan_B"]))
        # Recommend
        obs, reward, done, info = env.step(LoanAdvisorAction(
            action_type="recommend",
            recommended_decision="go",
            recommended_loan_id="loan_A",
            reasoning="Full research",
        ))
        # 0.60 base + 0.10 compare + 0.10 roi + 0.10 queries(>=3) = 0.90
        # No scholarship for easy task
        assert reward == pytest.approx(0.90, abs=0.01)

    def test_full_research_with_scholarship(self, env):
        """Medium task with scholarship query should yield max 1.0."""
        env.reset(task_id="task_medium")
        env.step(LoanAdvisorAction(action_type="query_info", query_field="loan_products"))
        env.step(LoanAdvisorAction(action_type="query_info", query_field="salary_outlook"))
        env.step(LoanAdvisorAction(action_type="query_info", query_field="user_profile"))
        env.step(LoanAdvisorAction(action_type="query_info", query_field="scholarship_options"))
        env.step(LoanAdvisorAction(action_type="calculate", calculation_type="roi", loan_id="loan_A"))
        env.step(LoanAdvisorAction(action_type="compare", loan_ids=["loan_A", "loan_B"]))
        obs, reward, done, info = env.step(LoanAdvisorAction(
            action_type="recommend",
            recommended_decision="go",
            recommended_loan_id="loan_A",
            reasoning="Full research with scholarship",
        ))
        # 0.60 + 0.10 + 0.10 + 0.10 + 0.10 = 1.0
        assert reward == pytest.approx(1.0, abs=0.01)

    def test_no_research_correct_decision(self, env):
        """Correct decision with no research = base 0.60 only."""
        env.reset(task_id="task_easy")
        obs, reward, done, info = env.step(LoanAdvisorAction(
            action_type="recommend",
            recommended_decision="go",
            recommended_loan_id="loan_A",
            reasoning="Lucky guess",
        ))
        assert reward == pytest.approx(0.60, abs=0.01)


# ---------------------------------------------------------------------------
# Episode Boundaries
# ---------------------------------------------------------------------------

class TestEpisodeBoundaries:
    def test_step_after_done_returns_done(self, env):
        env.reset(task_id="task_easy")
        env.step(LoanAdvisorAction(
            action_type="recommend", recommended_decision="go",
            recommended_loan_id="loan_A", reasoning="Done",
        ))
        obs, reward, done, info = env.step(LoanAdvisorAction(
            action_type="query_info", query_field="loan_products",
        ))
        assert done is True
        assert obs.episode_done is True

    def test_step_limit_reached(self, env):
        env.reset(task_id="task_easy")  # max_steps = 8
        for i in range(8):
            obs, reward, done, info = env.step(LoanAdvisorAction(
                action_type="query_info", query_field="loan_products",
            ))
        assert done is True
        assert "Step limit" in obs.action_result

    def test_step_without_reset(self, env):
        obs, reward, done, info = env.step(LoanAdvisorAction(
            action_type="query_info", query_field="loan_products",
        ))
        assert done is True
        assert "No active episode" in obs.action_result


# ---------------------------------------------------------------------------
# Financial Calculations — Accuracy
# ---------------------------------------------------------------------------

class TestFinancialCalculations:
    def test_emi_calculation(self):
        # 10L at 10% for 5 years → known EMI ≈ ₹21,247/month
        emi = calculate_emi(1_000_000, 10.0, 5)
        assert 21_200 < emi < 21_300

    def test_emi_zero_principal(self):
        assert calculate_emi(0, 10.0, 5) == 0.0

    def test_emi_zero_rate(self):
        emi = calculate_emi(600_000, 0.0, 5)
        assert emi == pytest.approx(10_000, abs=1)  # 6L / 60 months

    def test_total_cost_with_moratorium(self):
        # Moratorium increases effective principal
        cost_no_mor = calculate_total_cost(1_000_000, 10.0, 5, 0, 0)
        cost_with_mor = calculate_total_cost(1_000_000, 10.0, 5, 0, 2)
        assert cost_with_mor > cost_no_mor

    def test_roi_positive(self):
        roi = calculate_roi(
            current_annual_income=400_000,
            post_grad_annual_salary=1_600_000,
            loan_total_cost=1_200_000,
        )
        assert roi["roi_positive"] is True
        assert roi["net_roi_10yr"] > 0

    def test_roi_negative(self):
        roi = calculate_roi(
            current_annual_income=360_000,
            post_grad_annual_salary=420_000,
            loan_total_cost=3_000_000,
        )
        assert roi["roi_positive"] is False

    def test_affordability_affordable(self):
        aff = calculate_affordability(
            emi=15_000, monthly_income=100_000, monthly_expenses=30_000,
        )
        assert aff["affordable"] is True

    def test_affordability_unaffordable(self):
        aff = calculate_affordability(
            emi=30_000, monthly_income=50_000, monthly_expenses=30_000,
        )
        assert aff["affordable"] is False  # 30K EMI > 30% of 20K disposable


# ---------------------------------------------------------------------------
# Grading Determinism
# ---------------------------------------------------------------------------

class TestGradingDeterminism:
    def test_same_actions_same_score(self, env):
        """Running the same sequence twice must produce identical scores."""
        def run_episode():
            env.reset(task_id="task_medium")
            env.step(LoanAdvisorAction(action_type="query_info", query_field="loan_products"))
            env.step(LoanAdvisorAction(action_type="query_info", query_field="salary_outlook"))
            env.step(LoanAdvisorAction(action_type="query_info", query_field="user_profile"))
            env.step(LoanAdvisorAction(action_type="calculate", calculation_type="roi", loan_id="loan_A"))
            env.step(LoanAdvisorAction(action_type="compare", loan_ids=["loan_A", "loan_B"]))
            obs, reward, _, _ = env.step(LoanAdvisorAction(
                action_type="recommend", recommended_decision="go",
                recommended_loan_id="loan_A", reasoning="Test",
            ))
            return reward

        score1 = run_episode()
        score2 = run_episode()
        assert score1 == score2

    def test_scores_in_0_1_range(self, env):
        """All possible outcomes should produce scores in [0.0, 1.0]."""
        for task_id in TASK_ORDER:
            for decision in ["go", "no_go"]:
                for loan_id in [None, "loan_A", "loan_B"]:
                    env.reset(task_id=task_id)
                    obs, reward, done, info = env.step(LoanAdvisorAction(
                        action_type="recommend",
                        recommended_decision=decision,
                        recommended_loan_id=loan_id,
                        reasoning="Test",
                    ))
                    assert 0.0 <= reward <= 1.0, f"Score {reward} out of range for {task_id}/{decision}/{loan_id}"


# ---------------------------------------------------------------------------
# State endpoint
# ---------------------------------------------------------------------------

class TestState:
    def test_state_before_reset(self, env):
        state = env.state()
        assert state["status"] == "not_started"

    def test_state_after_reset(self, env):
        env.reset(task_id="task_easy")
        state = env.state()
        assert state["task_id"] == "task_easy"
        assert state["step_count"] == 0
        assert state["done"] is False

    def test_state_tracks_queries(self, env):
        env.reset(task_id="task_easy")
        env.step(LoanAdvisorAction(action_type="query_info", query_field="loan_products"))
        state = env.state()
        assert "loan_products" in state["queries_made"]


# ---------------------------------------------------------------------------
# Close / cleanup
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_resets_state(self, env):
        env.reset(task_id="task_easy")
        env.close()
        state = env.state()
        assert state["status"] == "not_started"

    def test_close_then_reset_works(self, env):
        env.reset(task_id="task_easy")
        env.step(LoanAdvisorAction(action_type="query_info", query_field="loan_products"))
        env.close()
        obs = env.reset(task_id="task_medium")
        assert obs.task_id == "task_medium"
        assert obs.steps_taken == 0


# ---------------------------------------------------------------------------
# Observation structure
# ---------------------------------------------------------------------------

class TestObservationStructure:
    def test_observation_has_all_fields(self, env):
        obs = env.reset(task_id="task_easy")
        assert hasattr(obs, "task_id")
        assert hasattr(obs, "task_description")
        assert hasattr(obs, "action_result")
        assert hasattr(obs, "student_profile_summary")
        assert hasattr(obs, "course_university")
        assert hasattr(obs, "available_loan_ids")
        assert hasattr(obs, "steps_taken")
        assert hasattr(obs, "max_steps")
        assert hasattr(obs, "episode_done")
        assert hasattr(obs, "final_reward")
        assert hasattr(obs, "correct_answer")

    def test_observation_correct_answer_hidden_during_episode(self, env):
        env.reset(task_id="task_easy")
        obs, _, _, _ = env.step(LoanAdvisorAction(
            action_type="query_info", query_field="loan_products",
        ))
        assert obs.correct_answer is None
        assert obs.final_reward is None

    def test_observation_correct_answer_revealed_at_end(self, env):
        env.reset(task_id="task_easy")
        obs, _, _, _ = env.step(LoanAdvisorAction(
            action_type="recommend", recommended_decision="go",
            recommended_loan_id="loan_A", reasoning="Test",
        ))
        assert obs.correct_answer is not None
        assert obs.final_reward is not None
        assert "go" in obs.correct_answer


# ---------------------------------------------------------------------------
# Medium task with 3 loans
# ---------------------------------------------------------------------------

class TestMediumTaskThreeLoans:
    def test_medium_has_three_loans(self, env):
        obs = env.reset(task_id="task_medium")
        assert len(obs.available_loan_ids) == 3
        assert "loan_C" in obs.available_loan_ids

    def test_compare_any_two_of_three(self, env):
        env.reset(task_id="task_medium")
        obs, reward, _, _ = env.step(LoanAdvisorAction(
            action_type="compare", loan_ids=["loan_A", "loan_C"],
        ))
        assert reward == 0.10
        assert "Cheaper" in obs.action_result


# ---------------------------------------------------------------------------
# Hard task — bond scholarship trap
# ---------------------------------------------------------------------------

class TestHardTaskBondScholarship:
    def test_hard_has_bond_scholarship(self, env):
        env.reset(task_id="task_hard")
        obs, _, _, _ = env.step(LoanAdvisorAction(
            action_type="query_info", query_field="scholarship_options",
        ))
        assert "Bond" in obs.action_result or "bond" in obs.action_result
        assert "opportunity cost" in obs.action_result.lower() or "3.5L" in obs.action_result

    def test_hard_roi_is_negative(self, env):
        env.reset(task_id="task_hard")
        obs, _, _, _ = env.step(LoanAdvisorAction(
            action_type="calculate", calculation_type="roi", loan_id="loan_A",
        ))
        assert "NO" in obs.action_result  # ROI Positive: NO
