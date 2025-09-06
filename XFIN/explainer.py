import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

class PrivacyPreservingExplainer:
    def __init__(self, model_interface, domain, compliance_level="GDPR_ECOA"):
        self.model = model_interface  # Black-box: only predict/predict_proba exposed
        self.domain = domain
        self.compliance_level = compliance_level

    def explain_prediction(self, sample):
        prediction = self.model.predict(sample)
        
        background_data = np.repeat(sample.values, 50, axis=0)
        for i in range(len(sample.columns)):
            col_mean = sample.values[0, i]
            col_std = abs(col_mean * 0.2) if col_mean != 0 else 0.1
            noise = np.random.normal(0, col_std, background_data.shape[0])
            background_data[:, i] = background_data[:, i] + noise
        
        explainer = shap.KernelExplainer(
            self.model.predict_proba, 
            background_data,
            link="logit"
        )
        shap_values = explainer.shap_values(sample, nsamples=100)
        
        # For binary classification, shap_values is a list with 2 elements (one for each class)
        # We'll use the positive class (index 1) for explanations
        if isinstance(shap_values, list):
            shap_values_class = shap_values[1]  # Use positive class
        else:
            shap_values_class = shap_values
        
        # Flatten the shap values if they're 2D
        if len(shap_values_class.shape) > 1:
            shap_values_flat = shap_values_class.flatten()
        else:
            shap_values_flat = shap_values_class
            
        shap_top = self._get_top_features(sample.columns, shap_values_flat)
        
        # LIME
        # Create a more realistic background dataset for LIME
        # Use multiple variations of the input to create better training data
    
        # Create synthetic background data based on the sample
        n_background = 500
        background_samples = []
        
        for _ in range(n_background):
            # Create variations by modifying each feature
            new_sample = sample.values.flatten().copy()
            
            # Randomly modify some features
            n_features_to_modify = np.random.randint(1, min(5, len(new_sample)))
            features_to_modify = np.random.choice(len(new_sample), n_features_to_modify, replace=False)
            
            for feat_idx in features_to_modify:
                current_val = new_sample[feat_idx]
                if current_val == 0:  # Binary or categorical feature
                    new_sample[feat_idx] = np.random.choice([0, 1])
                else:  # Continuous feature
                    # Add noise proportional to the value
                    noise_scale = abs(current_val * 0.5) if current_val != 0 else 0.5
                    new_sample[feat_idx] = current_val + np.random.normal(0, noise_scale)
                    new_sample[feat_idx] = max(0, new_sample[feat_idx])  # Keep non-negative
            
            background_samples.append(new_sample)
        
        synthetic_training = np.array(background_samples)
        
        lime_explainer = LimeTabularExplainer(
            training_data=synthetic_training,
            feature_names=sample.columns.tolist(),
            class_names=['Not Approved', 'Approved'],
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
        
        # Get LIME explanation with more samples for better stability
        lime_exp = lime_explainer.explain_instance(
            sample.values.flatten(), 
            self.model.predict_proba,
            num_features=min(10, len(sample.columns)),
            num_samples=2000
        )
        
        # Get all features and their importance scores
        lime_features = lime_exp.as_list()
        # Sort by absolute importance and take top 3
        lime_sorted = sorted(lime_features, key=lambda x: abs(x[1]), reverse=True)[:3]
        lime_top = lime_sorted
        
        return {
            'prediction': prediction,
            'shap_top': shap_top,
            'lime_top': lime_top
        }

    def _get_top_features(self, columns, values):
        # Ensure values are scalar numbers, not arrays
        scalar_values = [float(v) if hasattr(v, 'item') else float(v) for v in values]
        return sorted(zip(columns, scalar_values), key=lambda x: abs(x[1]), reverse=True)[:3]

    def create_shap_plot(self, sample, explanation):
        """Create SHAP visualization plot"""
        try:
            # Get top features for plotting
            shap_top = explanation['shap_top']
            features = [item[0] for item in shap_top]
            values = [item[1] for item in shap_top]
            
            # Check if we have meaningful values
            if all(abs(v) < 1e-6 for v in values):
                # If values are too small, regenerate with different parameters
                print("SHAP values too small, regenerating...")
                explanation_new = self.explain_prediction(sample)
                shap_top = explanation_new['shap_top']
                features = [item[0] for item in shap_top]
                values = [item[1] for item in shap_top]
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(features))
            colors = ['#ff4444' if v < 0 else '#44ff44' for v in values]
            
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
            ax.set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                label_x = value + (max(values) * 0.02 if value >= 0 else min(values) * 0.02)
                ax.text(label_x, i, f'{value:.4f}', 
                       ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
            
            # Add grid for better readability
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating SHAP plot: {e}")
            # Create a simple fallback plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"SHAP plot generation failed: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('SHAP Feature Importance (Error)')
            return fig

    def create_lime_plot(self, explanation):
        """Create LIME visualization plot"""
        try:
            lime_top = explanation['lime_top']
            
            # Extract features and values from LIME explanation
            features = [item[0] for item in lime_top]
            values = [item[1] for item in lime_top]
            
            # Check if we have meaningful values
            if all(abs(v) < 1e-6 for v in values):
                print("Warning: LIME values are very small, results may not be meaningful")
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(features))
            colors = ['#ff4444' if v < 0 else '#44ff44' for v in values]
            
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('LIME Value (Impact on Prediction)', fontsize=12)
            ax.set_title('LIME Feature Importance', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                label_x = value + (max(values) * 0.02 if value >= 0 else min(values) * 0.02)
                ax.text(label_x, i, f'{value:.4f}', 
                       ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
            
            # Add grid for better readability
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating LIME plot: {e}")
            # Create a simple fallback plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"LIME plot generation failed: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('LIME Feature Importance (Error)')
            return fig
