from .explainer import PrivacyPreservingExplainer
from .utils import get_llm_explanation
from .compliance import ComplianceEngine


class CreditRiskModule(PrivacyPreservingExplainer):
    def __init__(self, model_interface, domain, compliance_level="GDPR_ECOA", api_key=None):
        super().__init__(model_interface, domain, compliance_level)
        self.compliance_engine = ComplianceEngine()
        self.api_key = api_key

    def get_top_features(self, sample):
        explanation = self.explain_prediction(sample)
        shap_top_str = ", ".join([f"{k}: {v:.3f}" for k, v in explanation['shap_top']])
        lime_top_str = ", ".join([f"{k}: {v:.3f}" for k, v in explanation['lime_top']])
        return shap_top_str, lime_top_str

    def generate_recommendations(self, sample):
        explanation = self.explain_prediction(sample)
        prediction = explanation['prediction']
        shap_top_str = ", ".join([f"{k}: {v:.3f}" for k, v in explanation['shap_top']])
        lime_top_str = ", ".join([f"{k}: {v:.3f}" for k, v in explanation['lime_top']])
        user_input_str = sample.iloc[0].to_dict()  # Get first row as dict
        # Pass self.api_key to get_llm_explanation
        llm_rec = get_llm_explanation(prediction, shap_top_str, lime_top_str, user_input_str, api_key=self.api_key)
        return llm_rec

    def get_compliance_notice(self, explanation):
        """Generate compliance notice using the integrated ComplianceEngine"""
        return self.compliance_engine.generate_adverse_action_notice(explanation)

    def full_analysis(self, sample):
        """Perform complete analysis including explanations and compliance"""
        explanation = self.explain_prediction(sample)
        recommendations = self.generate_recommendations(sample)
        compliance_notice = self.get_compliance_notice(explanation)
        return {
            'explanation': explanation,
            'recommendations': recommendations,
            'compliance_notice': compliance_notice
        }
