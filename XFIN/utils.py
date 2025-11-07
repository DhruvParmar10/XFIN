import os
import requests
from dotenv import load_dotenv
from .config import get_config, require_api_key

load_dotenv()

def _get_openrouter_key():
    """Get OpenRouter API key from environment"""
    config = get_config()
    return config.get_api_key('openrouter')

def get_llm_explanation(prediction, shap_top, lime_top, user_input, api_key=None, market_data=None):
    """
    Universal LLM function that intelligently handles both credit risk and stress testing
    while preserving the original credit risk functionality
    Enhanced with market data integration
    """
    
    # Enhanced detection logic for stress testing vs credit risk
    is_stress_testing = _detect_stress_testing_context(prediction, user_input, shap_top, lime_top)
    
    if is_stress_testing:
        # Use enhanced stress testing prompt with market data
        prompt = _create_enhanced_stress_testing_prompt(user_input, market_data)
    else:
        # Use original credit risk prompt (preserved exactly)
        prompt = _create_credit_risk_prompt(prediction, shap_top, lime_top, user_input)

    # Use the provided api_key if given, else fallback to env var
    key = api_key if api_key is not None else _get_openrouter_key()

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "x-ai/grok-code-fast-1",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3500  # Increased for comprehensive analysis without cutoffs
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=20  # Increased timeout for comprehensive responses
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

def _create_enhanced_stress_testing_prompt(user_input, market_data=None):
    """
    Create enhanced stress testing prompt with live market data integration
    """
    # Extract market intelligence from API data
    market_intelligence = ""
    if market_data:
        market_intelligence = _format_market_intelligence(market_data)
    
    base_prompt = f"""
{user_input}

{market_intelligence}

## 🎯 Current Market Context (Indian Markets - October 2025)
**BASELINE REFERENCE VALUES** (Use these for context and calculations):
- **Nifty 50 Index**: ~25,000 levels (Current baseline)
- **USD/INR Exchange Rate**: ₹88-89 per dollar (Current baseline)
- **Indian Market Focus**: All portfolio analysis is in Indian Rupees (₹)

## Role
You are a sophisticated Portfolio Stress Testing Advisor specializing in Indian equity markets and translating complex portfolio stress analysis into clear, actionable investment guidance. Your expertise covers risk management, portfolio optimization, and crisis preparation strategies for individual investors in the Indian financial market.

## CRITICAL CURRENCY & FORMATTING RULES
**MANDATORY - DO NOT DEVIATE:**
1. **ALL monetary values must be in Indian Rupees (₹)** - Never use $, USD, or any other currency
2. **Always use ₹ symbol** before amounts (e.g., ₹1,50,000 or ₹1.5 lakhs)
3. **Use Indian numbering system** where appropriate (Lakhs/Crores)
4. **Portfolio values**: ₹X lakhs or ₹X crores
5. **Stock prices**: ₹XXX per share
6. **Market cap references**: Use ₹ crores or ₹ lakh crores for Indian companies

## Critical Instructions
1. **START IMMEDIATELY** with stress test analysis - NO disclaimers about data availability
2. **USE Nifty 50 ~25,000 and USD/INR ~₹88-89** as reference in your analysis
3. **INTEGRATE ESG insights** from portfolio analysis into recommendations
4. **BE DIRECT** - consumers want actionable insights, not technical caveats
5. **ASSUME data is sufficient** - work with what's provided
6. **USE ONLY INR (₹)** for all monetary references - NO dollar symbols

## ESG Integration (MANDATORY)
If ESG analysis results are provided in the input:
- **Highlight ESG strengths/weaknesses** in portfolio composition
- **Connect ESG ratings to stress resilience** (high ESG = lower downside risk)
- **Recommend ESG improvements** that also reduce stress vulnerability
- **Reference specific holdings** with poor/strong ESG scores
- **Link sector ESG performance** to scenario outcomes

Example: "Your portfolio's IT holdings show strong ESG scores (68/100 avg), which typically correlates with better crisis management. However, the Oil & Gas exposure (45/100 avg) presents dual risk—both from commodity volatility AND environmental transition."

## Task
Provide comprehensive stress testing analysis by:

1. **Scenario Context** - Explain the stress event with current Indian market backdrop (Nifty ~25,000)
2. **Portfolio Impact** - Break down losses/gains by sector and holding IN RUPEES (₹)
3. **ESG-Informed Risk Assessment** - Use ESG scores to identify resilience gaps
4. **Actionable Recommendations** - Specific rebalancing with ESG considerations
5. **Recovery Timeline** - Realistic path forward with milestones

## Output Format Requirements
- **Start with "This stress test..."** - NO preamble about data limitations
- **Use bullet points** for clarity
- **ALL amounts in ₹ (Rupees)** - Never use $ or USD
- **Include specific holdings** by name when discussing risks/opportunities
- **Reference ESG scores** when discussing individual stocks or sectors
- **Provide percentage allocations** for rebalancing suggestions
- **End with concrete next steps**

Output should be investor-ready, not LLM-disclaimer-heavy.
"""
    
    return base_prompt

def _format_market_intelligence(market_data):
    """
    Format live market data into intelligence context for LLM with current market levels
    """
    intelligence = "\n## 📊 Live Market Intelligence\n"
    
    # Get current market data from APIs
    current_market_context = _get_current_market_context()
    intelligence += current_market_context
    
    if market_data:
        intelligence += "\nPortfolio-specific live data from Alpha Vantage & Polygon.io APIs:\n\n"
        
        for symbol, data in market_data.items():
            if 'error' not in data:
                market_cap = data.get('market_cap', 0)
                sector = data.get('sector', 'Unknown')
                
                if market_cap > 0:
                    # Format market cap
                    if market_cap >= 1_000_000_000_000:
                        cap_str = f"${market_cap / 1_000_000_000_000:.2f}T"
                    elif market_cap >= 1_000_000_000:
                        cap_str = f"${market_cap / 1_000_000_000:.2f}B"
                    else:
                        cap_str = f"${market_cap / 1_000_000:.2f}M"
                    
                    intelligence += f"• **{symbol}**: {cap_str} market cap, {sector} sector\n"
    
    intelligence += "\nUse this live data to enhance your analysis with current market context.\n"
    return intelligence

def _get_current_market_context():
    """
    Provide current Indian market levels and USD/INR as hardcoded baseline values
    Nifty 50: ~25,000 levels | USD/INR: ~₹88-89
    """
    try:
        from datetime import datetime
        
        # HARDCODED BASELINE VALUES - Do not change without business approval
        nifty_50_level = 25000
        usd_inr_rate = 88.50
        
        context = "**Current Market Context (Indian Markets):**\n"
        context += f"- **Nifty 50**: ~{nifty_50_level:,} levels\n"
        context += f"- **USD/INR**: ₹{usd_inr_rate:.2f} per dollar\n"
        context += f"- **Market**: Indian equity markets (NSE/BSE)\n"
        context += f"- **Currency**: All values in Indian Rupees (₹)\n"
        context += f"- **Updated**: {datetime.now().strftime('%B %Y')}\n"
        
        return context
        
    except Exception as e:
        # Fallback with default baseline values
        from datetime import datetime
        return f"""**Current Market Context (Indian Markets):**
- **Nifty 50**: ~25,000 levels
- **USD/INR**: ₹88.50 per dollar
- **Market**: Indian equity markets (NSE/BSE)
- **Currency**: All values in Indian Rupees (₹)
- **Updated**: {datetime.now().strftime('%B %Y')}
"""

def _create_stress_testing_prompt(user_input):
    """
    Create stress testing specific prompt (legacy version)
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
    key = api_key if api_key is not None else _get_openrouter_key()
    
    # Check if API key is available
    if not key or key.strip() == "":
        return "❌ **API Key Required**: Please provide an OpenRouter API key in the sidebar to enable AI-powered recommendations. You can get a free API key from [OpenRouter.ai](https://openrouter.ai/)."

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "x-ai/grok-code-fast-1",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300  # Limit response length for speed
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=3  # Ultra-fast 3-second timeout for dashboard
        )
        
        response.raise_for_status()
        raw_content = response.json()['choices'][0]['message']['content']
        return raw_content

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "❌ **Authentication Error**: Invalid API key. Please check your OpenRouter API key and try again."
        elif e.response.status_code == 429:
            return "❌ **Rate Limit Exceeded**: Too many requests. Please try again later."
        else:
            return f"❌ **HTTP Error {e.response.status_code}**: {e.response.text}"
    except requests.exceptions.Timeout:
        return "❌ **Timeout Error**: Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error**: Unable to connect to OpenRouter API. Please check your internet connection."
    except Exception as e:
        return f"❌ **LLM Error**: {str(e)}"