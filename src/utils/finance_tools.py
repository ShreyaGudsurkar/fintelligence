import logging
from langchain_core.tools import tool
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)

@tool
def calculate_compound_interest(principal: float, monthly_deposit: float, annual_roi: float, years: int) -> str:
    """
    Calculates the future value of an investment with monthly contributions and compound interest.
    All rates are expected in percentage (e.g., 7 for 7%).
    """
    try:
        rate = annual_roi / 100 / 12
        months = years * 12
        
        # Future value of initial principal
        fv_principal = principal * (1 + rate)**months
        
        # Future value of monthly deposits
        fv_deposits = monthly_deposit * ((1 + rate)**months - 1) / rate if rate > 0 else monthly_deposit * months
        
        total_fv = fv_principal + fv_deposits
        total_invested = principal + (monthly_deposit * months)
        total_interest = total_fv - total_invested
        
        return (
            f"Financial Projection for {years} years:\n"
            f"- Total Future Value: ${total_fv:,.2f}\n"
            f"- Total Invested: ${total_invested:,.2f}\n"
            f"- Total Interest Earned: ${total_interest:,.2f}\n"
            f"- Average Monthly Growth: ${ (total_interest/months):,.2f}"
        )
    except Exception as e:
        logger.error(f"Compound interest calculation failed: {e}")
        return f"Error calculating compound interest: {str(e)}"

@tool
def generate_amortization_schedule(loan_amount: float, annual_interest_rate: float, years: int) -> str:
    """
    Generates a high-level amortization schedule for a loan or mortgage.
    Provides monthly payment and total interest paid over the life of the loan.
    """
    try:
        monthly_rate = annual_interest_rate / 100 / 12
        num_payments = years * 12
        
        if monthly_rate > 0:
            monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        else:
            monthly_payment = loan_amount / num_payments
            
        total_paid = monthly_payment * num_payments
        total_interest = total_paid - loan_amount
        
        # Breakdown by year for high-level summary
        summary = (
            f"Loan Summary ({years} years):\n"
            f"- Monthly Payment: ${monthly_payment:,.2f}\n"
            f"- Principal: ${loan_amount:,.2f}\n"
            f"- Total Interest: ${total_interest:,.2f}\n"
            f"- Total Lifetime Cost: ${total_paid:,.2f}\n"
        )
        
        return summary
    except Exception as e:
        logger.error(f"Amortization schedule generation failed: {e}")
        return f"Error generating amortization schedule: {str(e)}"

@tool
def calculate_goal_feasibility(target_amount: float, current_savings: float, years: int, annual_roi: float) -> str:
    """
    Calculates the required monthly savings to reach a target financial goal.
    """
    try:
        rate = annual_roi / 100 / 12
        months = years * 12
        
        # Future value of current savings
        fv_current = current_savings * (1 + rate)**months
        
        # Remaining amount needed
        amount_needed = target_amount - fv_current
        
        if amount_needed <= 0:
            return f"Great news! Your current savings of ${current_savings:,.2f} will grow to over your target of ${target_amount:,.2f} in {years} years at {annual_roi}% ROI without any further contributions."
            
        if rate > 0:
            monthly_needed = amount_needed * rate / ((1 + rate)**months - 1)
        else:
            monthly_needed = amount_needed / months
            
        return (
            f"Goal Feasibility Analysis:\n"
            f"- Target: ${target_amount:,.2f}\n"
            f"- Timeframe: {years} years\n"
            f"- Required Monthly Contribution: ${monthly_needed:,.2f} (at {annual_roi}% annual ROI)"
        )
    except Exception as e:
        logger.error(f"Goal feasibility check failed: {e}")
        return f"Error checking goal feasibility: {str(e)}"
