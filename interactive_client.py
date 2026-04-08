"""
interactive_client.py - Interactive Loan Advisor Client
========================================================

This script allows users to input their own details and get a personalized
loan recommendation using the Loan Advisor environment.

Usage:
    python3 interactive_client.py

Required environment variables:
    API_BASE_URL   - LLM API endpoint
    MODEL_NAME     - Model identifier
    HF_TOKEN       - API key
    ENV_BASE_URL   - OpenEnv server URL (default: http://localhost:7860)
"""

import json
import os
import sys
import time
import re
from typing import Any, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Check if running in interactive mode or just demo
DEMO_MODE = not (API_BASE_URL and MODEL_NAME and HF_TOKEN)

if DEMO_MODE:
    print("=" * 60)
    print("⚠️  Running in DEMO MODE (no LLM configured)")
    print("Set API_BASE_URL, MODEL_NAME, HF_TOKEN for full functionality")
    print("=" * 60)
    print()

TEMPERATURE = 0.2
MAX_TOKENS = 600

# ---------------------------------------------------------------------------
# User Input Helpers
# ---------------------------------------------------------------------------
def get_input(prompt: str, default: str = "") -> str:
    """Get input with optional default value."""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def get_number(prompt: str, default: float = 0) -> float:
    """Get numeric input with validation."""
    while True:
        try:
            val = get_input(prompt, str(int(default)) if default == int(default) else str(default))
            return float(val.replace(",", "").replace("₹", ""))
        except ValueError:
            print("Please enter a valid number.")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input."""
    default_str = "Y/n" if default else "y/N"
    result = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not result:
        return default
    return result in ("y", "yes", "true", "1")


# ---------------------------------------------------------------------------
# Course Data (from environment.py)
# ---------------------------------------------------------------------------
COURSE_OPTIONS = {
    "1": {
        "key": "btech_cs_iit_bombay",
        "name": "B.Tech Computer Science @ IIT Bombay",
        "fees": 875000,
        "duration": 4,
        "avg_salary": 1600000,
        "field": "Engineering / Technology",
    },
    "2": {
        "key": "mba_private_bschool",
        "name": "MBA @ Private B-School (PGDM)",
        "fees": 1800000,
        "duration": 2,
        "avg_salary": 850000,
        "field": "Management / Business",
    },
    "3": {
        "key": "bfa_symbiosis",
        "name": "BFA + Film Studies @ Symbiosis",
        "fees": 2400000,
        "duration": 4,
        "avg_salary": 420000,
        "field": "Arts / Creative / Film",
    },
    "4": {
        "key": "custom",
        "name": "Enter custom course details",
        "fees": 0,
        "duration": 0,
        "avg_salary": 0,
        "field": "",
    },
}

LOAN_OPTIONS = [
    {
        "id": "loan_A",
        "name": "SBI Education Loan",
        "rate": 8.5,
        "tenure": 5,
        "features": "Government-backed, lowest rate",
    },
    {
        "id": "loan_B",
        "name": "HDFC Credila Loan",
        "rate": 11.5,
        "tenure": 7,
        "features": "Flexible tenure, quick processing",
    },
    {
        "id": "loan_C",
        "name": "Private NBFC Loan",
        "rate": 15.0,
        "tenure": 5,
        "features": "Easy approval, high interest",
    },
]


# ---------------------------------------------------------------------------
# Financial Calculations
# ---------------------------------------------------------------------------
def calculate_emi(principal: float, annual_rate: float, tenure_years: int) -> float:
    """Calculate monthly EMI."""
    if principal <= 0:
        return 0.0
    r = annual_rate / 100.0 / 12.0
    n = tenure_years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def calculate_total_cost(principal: float, annual_rate: float, tenure_years: int) -> float:
    """Calculate total repayment amount."""
    emi = calculate_emi(principal, annual_rate, tenure_years)
    return emi * tenure_years * 12


def calculate_roi(current_income: float, post_grad_salary: float, total_loan_cost: float, years: int = 10) -> dict:
    """Calculate ROI over specified years."""
    annual_increment = post_grad_salary - current_income
    cumulative_increment = annual_increment * years
    net_roi = cumulative_increment - total_loan_cost
    return {
        "annual_increment": annual_increment,
        "cumulative_10yr": cumulative_increment,
        "total_loan_cost": total_loan_cost,
        "net_roi": net_roi,
        "is_positive": net_roi > 0,
    }


# ---------------------------------------------------------------------------
# LLM Advisor
# ---------------------------------------------------------------------------
ADVISOR_PROMPT = """You are an expert financial advisor specializing in education loans in India.

A student is seeking your advice on whether to take an education loan. Analyze their situation and provide a clear recommendation.

## Student Profile
- Name: {name}
- Current Monthly Income: ₹{monthly_income:,}
- Monthly Expenses: ₹{monthly_expenses:,}
- Savings: ₹{savings:,}
- Dependents: {dependents}
- Current Annual Income: ₹{annual_income:,}

## Course Details
- Course: {course_name}
- Total Fees: ₹{total_fees:,}
- Duration: {duration} years
- Field: {field}
- Expected Post-Graduation Salary: ₹{expected_salary:,}/year

## Loan Options
{loan_options}

## Financial Analysis
- Loan Amount Needed: ₹{loan_needed:,}
- Best Loan Option: {best_loan_name} at {best_loan_rate}%
- Monthly EMI (post-graduation): ₹{emi:,.0f}
- Total Repayment: ₹{total_repayment:,.0f}
- 10-Year ROI: ₹{roi:,.0f} ({roi_status})

## Your Task
Based on this analysis, provide:
1. Your recommendation: GO (take the loan) or NO-GO (don't take the loan)
2. If GO, which loan option is best
3. Clear reasoning explaining your decision
4. Any important considerations or risks

Be direct and practical. Consider:
- Is the ROI positive?
- Is the EMI affordable (< 30% of post-grad income)?
- Does the field have good job prospects?
- Does the student have dependents to support?

Provide your advice in a conversational but professional tone.
"""


def get_llm_advice(client: OpenAI, user_data: dict) -> str:
    """Get personalized advice from LLM."""
    
    # Format loan options
    loan_options_str = ""
    for loan in LOAN_OPTIONS:
        loan_options_str += f"- {loan['name']}: {loan['rate']}% for {loan['tenure']} years ({loan['features']})\n"
    
    # Calculate financials
    loan_needed = max(0, user_data["total_fees"] - user_data["savings"])
    best_loan = LOAN_OPTIONS[0]  # Lowest rate
    emi = calculate_emi(loan_needed, best_loan["rate"], best_loan["tenure"])
    total_repayment = calculate_total_cost(loan_needed, best_loan["rate"], best_loan["tenure"])
    roi_data = calculate_roi(user_data["annual_income"], user_data["expected_salary"], total_repayment)
    
    prompt = ADVISOR_PROMPT.format(
        name=user_data["name"],
        monthly_income=user_data["monthly_income"],
        monthly_expenses=user_data["monthly_expenses"],
        savings=user_data["savings"],
        dependents=user_data["dependents"],
        annual_income=user_data["annual_income"],
        course_name=user_data["course_name"],
        total_fees=user_data["total_fees"],
        duration=user_data["duration"],
        field=user_data["field"],
        expected_salary=user_data["expected_salary"],
        loan_options=loan_options_str,
        loan_needed=loan_needed,
        best_loan_name=best_loan["name"],
        best_loan_rate=best_loan["rate"],
        emi=emi,
        total_repayment=total_repayment,
        roi=roi_data["net_roi"],
        roi_status="POSITIVE" if roi_data["is_positive"] else "NEGATIVE",
    )
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful and practical financial advisor."},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content or "Unable to generate advice."
    except Exception as e:
        return f"Error getting advice: {e}"


def get_rule_based_advice(user_data: dict) -> str:
    """Provide rule-based advice when LLM is not available."""
    
    loan_needed = max(0, user_data["total_fees"] - user_data["savings"])
    best_loan = LOAN_OPTIONS[0]
    emi = calculate_emi(loan_needed, best_loan["rate"], best_loan["tenure"])
    total_repayment = calculate_total_cost(loan_needed, best_loan["rate"], best_loan["tenure"])
    roi_data = calculate_roi(user_data["annual_income"], user_data["expected_salary"], total_repayment)
    
    post_grad_monthly = user_data["expected_salary"] / 12
    emi_ratio = (emi / post_grad_monthly * 100) if post_grad_monthly > 0 else 100
    
    # Decision logic
    reasons = []
    decision = "GO"
    
    if roi_data["net_roi"] < 0:
        decision = "NO-GO"
        reasons.append(f"❌ Negative ROI: You'll lose ₹{abs(roi_data['net_roi']):,.0f} over 10 years")
    else:
        reasons.append(f"✅ Positive ROI: Net gain of ₹{roi_data['net_roi']:,.0f} over 10 years")
    
    if user_data["expected_salary"] <= user_data["annual_income"]:
        decision = "NO-GO"
        reasons.append(f"❌ Post-grad salary (₹{user_data['expected_salary']:,}) ≤ current income (₹{user_data['annual_income']:,})")
    else:
        salary_increase = user_data["expected_salary"] - user_data["annual_income"]
        reasons.append(f"✅ Salary increase: +₹{salary_increase:,}/year")
    
    if emi_ratio > 30:
        if decision == "GO":
            decision = "CAUTION"
        reasons.append(f"⚠️ EMI is {emi_ratio:.1f}% of post-grad income (recommended: <30%)")
    else:
        reasons.append(f"✅ EMI is {emi_ratio:.1f}% of post-grad income (affordable)")
    
    if user_data["dependents"] > 0 and decision != "GO":
        reasons.append(f"⚠️ You have {user_data['dependents']} dependent(s) to support")
    
    if "Arts" in user_data["field"] or "Creative" in user_data["field"]:
        reasons.append("⚠️ Creative/Arts fields often have lower and less stable incomes")
    
    # Build advice
    advice = f"""
{'='*60}
📊 LOAN ADVISOR RECOMMENDATION
{'='*60}

👤 {user_data['name']}, here's my analysis of your education loan decision:

📚 Course: {user_data['course_name']}
💰 Loan Needed: ₹{loan_needed:,}
📅 Best Option: {best_loan['name']} at {best_loan['rate']}%

💳 Monthly EMI (after graduation): ₹{emi:,.0f}
💵 Total Repayment: ₹{total_repayment:,.0f}

{'='*60}
🎯 RECOMMENDATION: {decision}
{'='*60}

Analysis:
"""
    for reason in reasons:
        advice += f"\n  {reason}"
    
    if decision == "GO":
        advice += f"""

✅ Based on the numbers, taking this loan makes financial sense.
   The {best_loan['name']} at {best_loan['rate']}% offers the best terms.
   Your expected salary increase will more than cover the loan costs.
"""
    elif decision == "NO-GO":
        advice += """

❌ Based on the numbers, I recommend NOT taking this loan.
   The financial burden outweighs the potential benefits.
   Consider:
   - Alternative courses with better ROI
   - Scholarships or grants
   - Part-time study while working
"""
    else:
        advice += """

⚠️ This is a borderline case. Proceed with caution.
   Consider if you have a backup plan if job prospects don't work out.
"""
    
    advice += f"""
{'='*60}
"""
    return advice


# ---------------------------------------------------------------------------
# Main Interactive Flow
# ---------------------------------------------------------------------------
def main():
    print()
    print("=" * 60)
    print("🎓 EDUCATION LOAN ADVISOR")
    print("=" * 60)
    print()
    print("I'll help you decide whether taking an education loan is")
    print("a good financial decision for your situation.")
    print()
    print("-" * 60)
    
    # Get student info
    print("\n📋 STEP 1: Your Personal Details\n")
    
    name = get_input("Your name", "Student")
    monthly_income = get_number("Your current monthly income (₹)", 0)
    monthly_expenses = get_number("Your monthly expenses (₹)", 20000)
    savings = get_number("Your total savings (₹)", 100000)
    dependents = int(get_number("Number of dependents (0 if none)", 0))
    
    annual_income = monthly_income * 12
    
    # Get course info
    print("\n📚 STEP 2: Course Selection\n")
    print("Select a course option:")
    for key, course in COURSE_OPTIONS.items():
        if key != "4":
            print(f"  {key}. {course['name']}")
            print(f"     Fees: ₹{course['fees']:,} | Duration: {course['duration']}yr | Avg Salary: ₹{course['avg_salary']:,}/yr")
        else:
            print(f"  {key}. {course['name']}")
    print()
    
    course_choice = get_input("Select option (1-4)", "1")
    
    if course_choice not in COURSE_OPTIONS:
        course_choice = "1"
    
    if course_choice == "4":
        # Custom course
        course_name = get_input("Course name")
        university = get_input("University/Institute")
        total_fees = get_number("Total fees (₹)")
        duration = int(get_number("Duration (years)", 4))
        expected_salary = get_number("Expected salary after graduation (₹/year)")
        field = get_input("Field (e.g., Engineering, Arts, Business)", "General")
        course_name = f"{course_name} @ {university}"
    else:
        course = COURSE_OPTIONS[course_choice]
        course_name = course["name"]
        total_fees = course["fees"]
        duration = course["duration"]
        expected_salary = course["avg_salary"]
        field = course["field"]
        
        # Allow customization
        if get_yes_no("\nWant to adjust the expected salary?", False):
            expected_salary = get_number("Expected salary after graduation (₹/year)", expected_salary)
    
    # Compile user data
    user_data = {
        "name": name,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "savings": savings,
        "dependents": dependents,
        "annual_income": annual_income,
        "course_name": course_name,
        "total_fees": total_fees,
        "duration": duration,
        "expected_salary": expected_salary,
        "field": field,
    }
    
    # Generate advice
    print("\n" + "=" * 60)
    print("🔄 Analyzing your situation...")
    print("=" * 60)
    
    if DEMO_MODE:
        advice = get_rule_based_advice(user_data)
    else:
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        advice = get_llm_advice(client, user_data)
    
    print(advice)
    
    # Ask if they want to try another scenario
    print()
    if get_yes_no("Would you like to analyze another scenario?", False):
        main()
    else:
        print("\nThank you for using the Education Loan Advisor! Good luck with your decision! 🎓\n")


if __name__ == "__main__":
    main()
