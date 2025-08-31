import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

def get_llm_explanation(prediction, shap_top, lime_top, user_input):
    prompt = f"""You are a financial analyst explaining credit approval decisions to loan applicants. 

PREDICTION: {'APPROVED' if prediction == 1 else 'REJECTED'}

APPLICANT PROFILE:
{user_input}

KEY INFLUENCING FACTORS (SHAP Analysis):
{shap_top}

SUPPORTING ANALYSIS (LIME Features):
{lime_top}

Provide a detailed, specific explanation (shouldn't include astrik,quotos or speacial characters) in plain text that includes:

1. PRIMARY DECISION FACTORS: Identify the 2-3 most important features that drove this decision and explain specifically how each feature value influenced the outcome.

2. RISK ASSESSMENT: Explain what specific aspects of the applicant's profile the model considers risky or favorable, with actual numbers when relevant.

3. COMPARATIVE CONTEXT: Explain how this applicant's key metrics compare to typical approved/rejected applications (without giving specific ranges).

4. ACTIONABLE INSIGHTS: If rejected, provide 2-3 specific actions the applicant could take to improve their chances. If approved, explain what strengths secured the approval.

Keep the explanation conversational but professional, avoiding vague statements. Be specific about why each feature matters for credit risk assessment. Keep it max 3 Paragraph"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "google/gemini-2.5-flash",
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
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"LLM explanation error: {e}"
