"""
================================================================================
LOANFLOW - FINTECH LOAN APPROVAL PREDICTOR
Streamlit Web Application
================================================================================

Interactive web app that predicts loan approvals using ML model.
Run with: streamlit run app.py

Author: LoanFlow Team
Date: 2024
================================================================================
"""

# ============================================================================
# IMPORTS: Libraries needed for the app
# ============================================================================
import streamlit as st           # Web app framework
import joblib                   # Load saved ML model
import pandas as pd             # Data handling
import numpy as np              # Numerical operations


# ============================================================================
# PAGE CONFIGURATION: Set up the Streamlit page
# ============================================================================
st.set_page_config(
    page_title="LoanFlow - Loan Approval Predictor",  # Browser tab title
    page_icon="💰",                                   # Favicon
    layout="centered",                                # Centered layout
    initial_sidebar_state="expanded"                  # Show sidebar by default
)

# ============================================================================
# CUSTOM STYLING: CSS for better appearance
# ============================================================================
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD ML MODEL: Cache the model to load only once
# ============================================================================
@st.cache_resource
def load_model():
    """
    Load the trained Random Forest model from pickle file.
    @st.cache_resource ensures it loads only once per session.
    """
    return joblib.load('loan_model.pkl')

# Load the model and handle errors
try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file 'loan_model.pkl' not found. Please ensure it's in the app directory.")
    st.stop()


# ============================================================================
# PAGE HEADER: Title and description
# ============================================================================
st.title("💰 LoanFlow")
st.markdown("**AI-Powered Loan Approval Predictor**")
st.divider()  # Horizontal line separator


# ============================================================================
# SIDEBAR: Input section where users enter applicant information
# ============================================================================
with st.sidebar:
    st.header("📋 Applicant Information")
    st.markdown("Enter loan applicant details below:")
    
    # ========================================================================
    # Row 1: Age and Employment Years (side by side)
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider(
            "Age",
            min_value=18,
            max_value=75,
            value=35,
            step=1,
            help="Applicant's age in years"
        )
    
    with col2:
        employment_years = st.slider(
            "Employment Years",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            help="Years at current job (job stability)"
        )
    
    # ========================================================================
    # Row 2: Annual Income
    # ========================================================================
    annual_income = st.number_input(
        "Annual Income ($)",
        min_value=10000,
        max_value=1000000,
        value=75000,
        step=5000,
        help="Total yearly salary before taxes"
    )
    
    # ========================================================================
    # Row 3: Requested Loan Amount
    # ========================================================================
    loan_amount = st.number_input(
        "Requested Loan Amount ($)",
        min_value=5000,
        max_value=1000000,
        value=200000,
        step=10000,
        help="How much money applicant wants to borrow"
    )
    
    # ========================================================================
    # Row 4: Credit Score
    # ========================================================================
    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=10,
        help="Credit rating (300-850 scale)"
    )
    
    # ========================================================================
    # Row 5: Estimated Monthly Debt
    # ========================================================================
    estimated_monthly_debt = st.number_input(
        "Estimated Monthly Debt ($)",
        min_value=0,
        max_value=50000,
        value=2000,
        step=100,
        help="Current monthly debt payments (car, credit cards, etc.)"
    )
    
    # ========================================================================
    # Calculate Debt-to-Income Ratio (auto-calculated)
    # ========================================================================
    # Formula: Monthly Debt / Monthly Income
    monthly_income = annual_income / 12
    debt_to_income = (estimated_monthly_debt / monthly_income 
                     if monthly_income > 0 else 0)
    
    # Display the calculated ratio
    st.info(f"📊 Debt-to-Income Ratio: {debt_to_income:.2%}")


# ============================================================================
# MAIN CONTENT: Prediction analysis and results
# ============================================================================
col1, col2 = st.columns([2, 1])

# ========================================================================
# Left Column: Prediction Results
# ========================================================================
with col1:
    st.header("📊 Analysis")
    
    # Prepare input data as DataFrame with same features as training
    input_data = pd.DataFrame({
        'Age': [age],
        'Annual_Income': [annual_income],
        'Credit_Score': [credit_score],
        'Loan_Amount': [loan_amount],
        'Employment_Years': [employment_years],
        'Debt_to_Income': [debt_to_income]
    })
    
    # Make prediction using the trained model
    prediction = model.predict(input_data)[0]          # 0=Rejected, 1=Approved
    probability = model.predict_proba(input_data)[0]  # [prob_rejected, prob_approved]
    
    # Display approval decision
    if prediction == 1:
        # APPROVED
        st.success("✅ APPROVED")
        approval_confidence = probability[1]
    else:
        # REJECTED
        st.error("❌ REJECTED")
        approval_confidence = probability[0]
    
    # Display decision confidence score
    st.metric(
        label="Decision Confidence",
        value=f"{max(probability)*100:.1f}%",
        help="How confident the model is in this decision"
    )


# ========================================================================
# Right Column: Quick Statistics
# ========================================================================
with col2:
    st.header("⚡ Quick Stats")
    # Display key applicant metrics
    st.metric(label="Age", value=f"{age} yrs")
    st.metric(label="Credit", value=credit_score)
    st.metric(label="Income", value=f"${annual_income:,.0f}")


# ============================================================================
# KEY RISK FACTORS: Analyze what affects the decision
# ============================================================================
st.divider()
st.header("💡 Key Factors")

# Build list of risk assessment factors
factors = []

# Factor 1: Credit Score
if credit_score < 600:
    factors.append(("❌", "Low Credit Score (< 600) - High Risk"))
else:
    factors.append(("✅", "Good Credit Score"))

# Factor 2: Annual Income
if annual_income < 40000:
    factors.append(("❌", "Low Income (< $40k) - Risk Factor"))
else:
    factors.append(("✅", "Sufficient Annual Income"))

# Factor 3: Debt-to-Income Ratio
if debt_to_income > 0.5:
    factors.append(("❌", "High Debt-to-Income Ratio (> 50%) - Risk"))
else:
    factors.append(("✅", "Healthy Debt-to-Income Ratio"))

# Factor 4: Employment History
if employment_years < 2:
    factors.append(("⚠️", "Limited Employment History"))
else:
    factors.append(("✅", "Stable Employment History"))

# Display all factors
for icon, factor in factors:
    st.write(f"{icon} {factor}")


# ============================================================================
# ADDITIONAL INFORMATION: Expandable sections
# ============================================================================
st.divider()

# ========================================================================
# Section 1: How the Model Works
# ========================================================================
with st.expander("📈 How This Works"):
    st.write("""
    **LoanFlow** uses a **Random Forest Machine Learning Model** trained on 
    500 loan applications.
    
    **What the model analyzes:**
    - **Credit Score** - Your payment history and creditworthiness
    - **Annual Income** - Your ability to repay the loan
    - **Debt-to-Income Ratio** - Your current financial obligations
    - **Employment History** - Your job stability
    - **Age & Loan Amount** - Your risk profile
    
    **How it predicts:**
    The model looks at all 6 factors together to estimate approval probability.
    
    **Important Note:** This is a demo application. Real loan decisions require 
    human review and compliance with banking regulations (Truth in Lending Act, etc.).
    """)

# ========================================================================
# Section 2: Sample Test Profiles
# ========================================================================
with st.expander("💾 Sample Test Data"):
    st.write("Try these profiles to see different outcomes:")
    
    # Define example profiles
    test_cases = {
        "✅ Strong Applicant": {
            "Age": 40, 
            "Annual_Income": 150000, 
            "Credit_Score": 780,
            "Loan_Amount": 200000, 
            "Employment_Years": 15, 
            "Debt_to_Income": 0.25
        },
        "⚠️ Medium Risk": {
            "Age": 28, 
            "Annual_Income": 55000, 
            "Credit_Score": 620,
            "Loan_Amount": 100000, 
            "Employment_Years": 3, 
            "Debt_to_Income": 0.45
        },
        "❌ High Risk": {
            "Age": 35, 
            "Annual_Income": 35000, 
            "Credit_Score": 520,
            "Loan_Amount": 300000, 
            "Employment_Years": 1, 
            "Debt_to_Income": 0.75
        }
    }
    
    # Display each test case
    for case_name, values in test_cases.items():
        st.write(f"\n**{case_name}**")
        st.json(values)


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("🔒 Your data is processed locally and never stored. © 2024 LoanFlow Demo")
