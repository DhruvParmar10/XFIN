class ComplianceEngine:
    def generate_adverse_action_notice(self, explanation):
        if explanation['prediction'] == 0:  # Not Approved
            notice = (
                "Adverse Action Notice (Compliant with ECOA/GDPR):\n"
                f"Reason: Top factors - {', '.join([f'{k} (impact: {v:.3f})' for k,v in explanation['shap_top']])}\n"
                "You have the right to request more details within 60 days."
            )
            return notice
        return "Loan Approved - No adverse action required."
