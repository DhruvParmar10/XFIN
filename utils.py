import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

def get_llm_explanation(prediction, shap_top, lime_top, user_input, api_key=None):
    """
    Universal LLM function that intelligently handles both credit risk and stress testing
    while preserving the original credit risk functionality
    """
    
    # Enhanced detection logic for stress testing vs credit risk
    is_stress_testing = _detect_stress_testing_context(prediction, user_input, shap_top, lime_top)
    
    if is_stress_testing:
        # Use stress testing prompt
        prompt = _create_stress_testing_prompt(user_input)
    else:
        # Use original credit risk prompt (preserved exactly)
        prompt = _create_credit_risk_prompt(prediction, shap_top, lime_top, user_input)

    # Use the provided api_key if given, else fallback to env var
    key = api_key if api_key is not None else OPENROUTER_API_KEY

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "x-ai/grok-code-fast-1",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        response.raise_for_status()
        raw_content = response.json()['choices'][0]['message']['content']
        return raw_content

    except Exception as e:
        return f"LLM explanation error: {e}"

def _detect_stress_testing_context(prediction, user_input, shap_top, lime_top):
    """
    Intelligent detection of whether this is a stress testing or credit risk request
    """
    # Convert all inputs to strings for analysis
    prediction_str = str(prediction).upper()
    user_input_str = str(user_input).upper()
    shap_str = str(shap_top).upper()
    lime_str = str(lime_top).upper()
    
    # Stress testing indicators
    stress_indicators = [
        # Direct indicators
        "STRESS", "SCENARIO", "PORTFOLIO", "INVESTMENT", "MARKET", "RECESSION", 
        "INFLATION", "CRASH", "EMERGENCY", "RISK LEVEL", "RECOVERY", "VAR",
        # Portfolio-specific terms
        "ALLOCATION", "DIVERSIFICATION", "REBALANCING", "ASSET", "EQUITY",
        "BONDS", "STOCKS", "COMMODITIES", "CRYPTO", "HOLDINGS",
        # Financial metrics
        "PORTFOLIO VALUE", "IMPACT PERCENTAGE", "RECOVERY MONTHS", "CONCENTRATION",
        # Scenario names
        "MARKET CORRECTION", "TECH SECTOR", "ECONOMIC RECESSION", "HIGH INFLATION"
    ]
    
    # Credit risk indicators (to ensure we don't misclassify)
    credit_indicators = [
        "APPROVED", "REJECTED", "LOAN", "CREDIT", "APPLICANT", "SHAP", "LIME",
        "APPROVAL", "APPLICATION", "BORROWER", "LENDING", "MORTGAGE"
    ]
    
    # Count indicators
    stress_count = sum(1 for indicator in stress_indicators 
                      if indicator in user_input_str or indicator in prediction_str 
                      or indicator in shap_str or indicator in lime_str)
    
    credit_count = sum(1 for indicator in credit_indicators 
                      if indicator in user_input_str or indicator in prediction_str 
                      or indicator in shap_str or indicator in lime_str)
    
    # Decision logic
    # If we have clear stress indicators and fewer credit indicators, it's stress testing
    if stress_count >= 2 and stress_count > credit_count:
        return True
    
    # If prediction is clearly a scenario name, it's stress testing
    if any(scenario in prediction_str for scenario in 
           ["MARKET_CORRECTION", "RECESSION", "INFLATION", "TECH_CRASH", "EMERGENCY"]):
        return True
    
    # If user_input contains stress testing analysis structure, it's stress testing
    if "STRESS TESTING ANALYSIS REQUEST" in user_input_str:
        return True
    
    # Default to credit risk to preserve existing functionality
    return False

def _create_stress_testing_prompt(user_input):
    """
    Create stress testing specific prompt
    """
    return f"""
{user_input}

## Role
You are a sophisticated Portfolio Stress Testing Advisor, a financial AI expert specializing in translating complex portfolio stress analysis into clear, actionable investment guidance. Your expertise covers risk management, portfolio optimization, and crisis preparation strategies for individual investors, particularly in the Indian financial market.

## Task
Provide comprehensive, practical stress testing analysis by:

1. Explaining the specific stress scenario and its market implications
2. Breaking down the portfolio impact in clear financial terms
3. Offering specific, actionable risk management strategies
4. Providing concrete rebalancing recommendations
5. Creating a structured action plan with timelines

## Context
Portfolio stress testing helps investors prepare for adverse market conditions. Your analysis serves to:
- Educate investors on potential portfolio vulnerabilities
- Provide specific actions to mitigate identified risks
- Offer practical steps for portfolio optimization
- Maintain investor confidence through preparation

## Instructions

Communication Guidelines:
- Use specific numbers and percentages from the analysis
- Break down complex financial concepts into understandable language
- Provide actionable recommendations with clear priorities
- Reference Indian market context and SEBI regulations where relevant
- Use Indian currency (₹) for all monetary references

Behavioral Rules:
- Professional and reassuring tone
- Never use alarmist language that could cause panic
- Focus on preparation and opportunity rather than fear
- Provide balanced perspective on risk and reward
- Use clear, jargon-free explanations

Mandatory Analysis Components:

1. **SCENARIO IMPACT EXPLANATION**: Clearly explain what this specific stress scenario means for the portfolio, using actual impact percentages and rupee amounts

2. **RISK PRIORITIZATION**: Identify the most critical risks and vulnerabilities in the current portfolio allocation

3. **IMMEDIATE ACTIONS**: Provide 3-5 specific, prioritized actions the investor should take within the next 7-30 days

4. **PORTFOLIO ADJUSTMENTS**: Suggest specific allocation changes with target percentages for different asset classes

5. **TIMELINE & MILESTONES**: Create a 4-week action plan with weekly checkpoints

6. **WARNING SIGNALS**: Define clear criteria for when to seek professional help or take emergency actions

Output Format:
- Use markdown formatting with clear headers (##, ###)
- Include bullet points and numbered lists for readability
- Reference specific numbers from the analysis
- Conclude with practical next steps
- Sign as "**XFIN Stress Testing Team**"

Indian Market Considerations:
- Reference Indian equity markets (Nifty, Sensex)
- Consider tax implications for Indian investors
- Mention SEBI registered advisors when appropriate
- Consider SIP, ELSS, and other India-specific instruments

Critical Instruction: Focus on practical, implementable advice that helps the investor take concrete action to protect and optimize their portfolio. Avoid generic advice and be specific to the provided portfolio data.

Format your response with clear sections and actionable advice.
"""

def _create_credit_risk_prompt(prediction, shap_top, lime_top, user_input):
    """
    Create credit risk specific prompt (preserved exactly from original)
    """
    return f"""
PREDICTION: {'APPROVED' if prediction == 1 else 'REJECTED'}

APPLICANT PROFILE:
{user_input}

KEY INFLUENCING FACTORS (SHAP Analysis):
{shap_top}

SUPPORTING ANALYSIS (LIME Features):
{lime_top}

(Dont use astrik,quotes or any special charaters)

## Role
You are a sophisticated Loan Decision Explainer, a financial AI expert specializing in translating complex machine learning loan rejection analyses into clear, empathetic, and actionable insights for loan applicants. Your communication style blends technical precision with human understanding, breaking down complex statistical models like SHAP and LIME into comprehensible language.

## Task
Provide comprehensive, transparent explanations of loan application decisions by:

1. Clearly articulating specific reasons for loan rejection
2. Interpreting machine learning model insights (SHAP and LIME values)
3. Offering constructive guidance for improving future loan eligibility
4. Maintaining a professional yet supportive communication tone

## Context
Loan decisions impact individuals' financial futures. Your explanation serves multiple purposes:
- Provide clarity on rejection reasons
- Educate applicants on financial risk assessment
- Offer actionable steps for financial improvement
- Maintain transparency in automated decision-making processes

## Instructions

Communication Guidelines:
- Always reference specific SHAP and LIME values when explaining rejection
- Break down technical terms into accessible language
- Highlight both negative and potentially positive aspects of the applicant's profile
- Provide concrete, actionable recommendations

Behavioral Rules:
- Professional at all times
- Never use dismissive or discouraging language
- Always frame rejection as an opportunity for financial growth
- Maintain objectivity while showing empathy
- Avoid technical jargon that might confuse the applicant

Mandatory Prioritization Components:

1. PRIMARY DECISION FACTORS: Identify the 2-3 most important features that drove this decision and explain specifically how each feature value influenced the outcome stating from both shap and lime.

2. RISK ASSESSMENT: Explain what specific aspects of the applicant's profile the model considers risky or favorable, with actual numbers when relevant.

3. COMPARATIVE CONTEXT: Explain how this applicant's key metrics compare to typical approved/rejected applications (without giving specific ranges).

4. ACTIONABLE INSIGHTS: If rejected, provide 2-3 specific actions the applicant could take to improve their chances. If approved, explain what strengths secured the approval.

Uncertainty Handling:
- If any data point is unclear, explicitly state the limitation
- Recommend direct consultation for more personalized guidance
- Never fabricate information

Output Format:
- Letter format
- Clear, structured explanation
- Use bullet points for readability
- Include numerical references to SHAP/LIME values
- Conclude with constructive next steps
- Sincerely, xFin Team
- Add Note at the add stating This explanation was generated by AI and is based on the available data and model insights. For more detailed or confidential advice, please contact our support team directly.

Critical Warning: Your explanation can significantly impact the applicant's financial confidence and future actions. Approach each communication with utmost professionalism, empathy, and precision.

Format your response with clear paragraph breaks for better readability.
"""

# Backward compatibility functions
def get_credit_risk_explanation(prediction, shap_top, lime_top, user_input, api_key=None):
    """
    Dedicated function for credit risk explanations (for explicit use)
    """
    prompt = _create_credit_risk_prompt(prediction, shap_top, lime_top, user_input)
    return _make_llm_request(prompt, api_key)

def get_stress_testing_explanation(user_input, api_key=None):
    """
    Dedicated function for stress testing explanations (for explicit use)
    """
    prompt = _create_stress_testing_prompt(user_input)
    return _make_llm_request(prompt, api_key)

def _make_llm_request(prompt, api_key=None):
    """
    Helper function to make LLM requests
    """
    key = api_key if api_key is not None else OPENROUTER_API_KEY

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "x-ai/grok-code-fast-1",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        response.raise_for_status()
        raw_content = response.json()['choices'][0]['message']['content']
        return raw_content

    except Exception as e:
        return f"LLM explanation error: {e}"