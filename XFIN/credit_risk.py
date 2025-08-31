from .explainer import PrivacyPreservingExplainer
from .utils import get_llm_explanation

class CreditRiskModule(PrivacyPreservingExplainer):
    def generate_recommendations(self, sample):
        explanation = self.explain_prediction(sample)
        prediction = explanation['prediction']
        shap_top_str = ", ".join([f"{k}: {v:.3f}" for k, v in explanation['shap_top']])
        lime_top_str = ", ".join([f"{k}: {v:.3f}" for k, v in explanation['lime_top']])
        user_input_str = sample.iloc[0].to_dict()  # Get first row as dict
        
        # Generate LLM-based recommendations
        llm_rec = get_llm_explanation(prediction, shap_top_str, lime_top_str, user_input_str)
        return llm_rec
