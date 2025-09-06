import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import tempfile
from .credit_risk import CreditRiskModule

class StreamlitApp:
    def __init__(self, model=None, data=None):
        self.model = model
        self.data = data
    
    def create_app_content(self):
        """Create the Streamlit app content"""
        return f'''
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from XFIN import CreditRiskModule, ComplianceEngine
import os

# Set page config
st.set_page_config(
    page_title="XFIN - Credit Risk Explainer",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 XFIN Credit Risk Explainer")
st.markdown("**Explainable AI for Credit Risk Assessment with Privacy-Preserving Features**")

# Sidebar for file upload and configuration
st.sidebar.header("Configuration")

# File uploads
model_file = st.sidebar.file_uploader("Upload Model File (.pl)", type=['pl'])
data_file = st.sidebar.file_uploader("Upload Dataset (.csv)", type=['csv'])

# Load model and data
@st.cache_data
def load_model(model_path=None):
    if model_path:
        return joblib.load(model_path)
    else:
        return None

@st.cache_data
def load_data(data_path=None):
    if data_path:
        return pd.read_csv(data_path)
    else:
        return None

# Initialize model and data
try:
    if model_file:
        model = joblib.load(model_file)
    else:
        model = None
    
    if data_file:
        data = pd.read_csv(data_file)
    else:
        data = None
    
    if model is None or data is None:
        st.info("👈 Use the file uploaders in the sidebar to get started")
        st.stop()
    
    # Prepare data - make it dynamic
    # Auto-detect target column or default to 'Approval_Status'
    potential_targets = ['Approval_Status', 'target', 'label', 'class', 'y']
    target_column = None
    
    for col in potential_targets:
        if col in data.columns:
            target_column = col
            break
    
    if target_column is None:
        # Use last column as target if no common target found
        target_column = data.columns[-1]
    
    # Display target info
    st.sidebar.write(f"🎯 **Target Column**: {{target_column}}")
    
    # Prepare features
    X = data.drop(target_column, axis=1)
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        st.sidebar.write(f"🔤 **Categorical Columns**: {{len(categorical_columns)}}")
        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=False)
    else:
        X_encoded = X.copy()
    
    st.sidebar.write(f"📊 **Total Features**: {{len(X_encoded.columns)}}")
    
    # Model wrapper class
    class UniversalModel:
        def __init__(self, model):
            self.model = model
            
        def predict(self, X): 
            return self.model.predict(X)
            
        def predict_proba(self, X): 
            return self.model.predict_proba(X)
    
    # Initialize explainer
    universal_model = UniversalModel(model)
    explainer = CreditRiskModule(universal_model, domain="credit_risk")
    
    st.success("✅ Model and data loaded successfully!")
    
except Exception as e:
    st.error(f"Error loading model or data: {{str(e)}}")
    st.stop()

# Main interface
st.header("📊 Credit Application Analysis")

# Create two tabs
tab1, tab2 = st.tabs(["Manual Input", "Sample from Dataset"])

with tab1:
    st.subheader("Enter Applicant Information")
    
    # Create input form based on original features (before encoding)
    col1, col2 = st.columns(2)
    
    # Get original columns (before dummy encoding)
    input_values = {{}}
    
    # Create inputs for each original feature
    for i, column in enumerate(X.columns):
        with col1 if i % 2 == 0 else col2:
            if column in categorical_columns:
                # Categorical column
                unique_values = X[column].dropna().unique()
                input_values[column] = st.selectbox(
                    f"📂 {{column}}", 
                    unique_values,
                    help=f"Categorical feature with {{len(unique_values)}} options"
                )
            else:
                # Numerical column
                min_val = float(X[column].min())
                max_val = float(X[column].max())
                mean_val = float(X[column].mean())
                
                # Determine step size
                range_val = max_val - min_val
                if range_val > 1000:
                    step = 10.0
                elif range_val > 100:
                    step = 1.0
                elif range_val > 10:
                    step = 0.1
                else:
                    step = 0.01
                
                input_values[column] = st.number_input(
                    f"🔢 {{column}}", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=mean_val,
                    step=step,
                    help=f"Numeric feature (range: {{min_val:.2f}} - {{max_val:.2f}})"
                )
    
    # Create sample dataframe and encode it properly
    sample_df = pd.DataFrame([input_values])
    
    # Apply same encoding as training data
    if categorical_columns:
        sample_encoded = pd.get_dummies(sample_df, columns=categorical_columns, drop_first=False)
    else:
        sample_encoded = sample_df.copy()
    
    # Ensure all columns are present and in correct order
    for col in X_encoded.columns:
        if col not in sample_encoded.columns:
            sample_encoded[col] = 0
    
    # Remove any extra columns that shouldn't be there
    sample_encoded = sample_encoded[[col for col in sample_encoded.columns if col in X_encoded.columns]]
    
    # Reorder columns to match training data
    sample_encoded = sample_encoded[X_encoded.columns]

with tab2:
    st.subheader("Select from Existing Applications")
    
    # Sample selection
    sample_idx = st.selectbox("Select a sample", range(min(50, len(X_encoded))))
    sample_encoded = X_encoded.iloc[[sample_idx]]
    
    # Display selected sample info
    st.write("Selected Application:")
    st.dataframe(sample_encoded.head())

# Analysis button
if st.button("🔍 Analyze Application", type="primary"):
    with st.spinner("Analyzing application..."):
        try:
            # Get comprehensive analysis including compliance
            full_analysis = explainer.full_analysis(sample_encoded)
            explanation = full_analysis['explanation']
            recommendations = full_analysis['recommendations']
            compliance_notice = full_analysis['compliance_notice']
            
            # Display results
            st.header("📋 Analysis Results")
            
            # Prediction result
            prediction = explanation['prediction'][0] if hasattr(explanation['prediction'], '__len__') else explanation['prediction']
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if prediction == 1:
                    st.success("✅ **APPROVED**")
                else:
                    st.error("❌ **REJECTED**")
            
            with col2:
                # Prediction probability if available
                try:
                    proba = universal_model.predict_proba(sample_encoded)[0]
                    st.metric("Approval Probability", f"{{proba[1]:.2%}}")
                except:
                    pass
            
            # Show compliance notice for rejected applications
            if prediction == 0:  # Rejected
                st.warning("⚖️ **Regulatory Compliance Notice**")
                st.text(compliance_notice)
                st.info("💡 This notice is generated in compliance with ECOA and GDPR regulations.")
                st.markdown("---")
            
            # Create two columns for visualizations
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("SHAP Analysis")
                shap_fig = explainer.create_shap_plot(sample_encoded, explanation)
                if shap_fig:
                    st.pyplot(shap_fig)
                else:
                    st.error("Could not generate SHAP plot")
                
                # SHAP explanation
                st.write("**Top SHAP Features:**")
                for feature, value in explanation['shap_top']:
                    direction = "🔴" if value < 0 else "🟢"
                    st.write(f"{{direction}} {{feature}}: {{value:.3f}}")
            
            with col2:
                st.subheader("LIME Analysis")
                lime_fig = explainer.create_lime_plot(explanation)
                if lime_fig:
                    st.pyplot(lime_fig)
                else:
                    st.error("Could not generate LIME plot")
                
                # LIME explanation
                st.write("**Top LIME Features:**")
                for feature, value in explanation['lime_top']:
                    direction = "🔴" if value < 0 else "🟢"
                    st.write(f"{{direction}} {{feature}}: {{value:.3f}}")
            
            st.markdown("---")

            st.subheader("🤖 AI Explanation")
            st.markdown(recommendations)
        
        except Exception as e:
            st.error(f"Error during analysis: {{str(e)}}")
            st.write("Debug info:")
            st.write(f"Sample shape: {{sample_encoded.shape}}")
            st.write(f"Sample columns: {{list(sample_encoded.columns)}}")

# Footer
st.markdown("---")
st.markdown("**XFIN Library** - Privacy-Preserving Explainable AI for Financial Services")
'''

    def launch_app(self, port=8501, host="localhost", auto_open=True):
        """Launch the Streamlit app"""
        try:
            # Create temporary app file
            app_content = self.create_app_content()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(app_content)
                temp_app_path = f.name
            
            # Launch streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run", temp_app_path,
                "--server.port", str(port),
                "--server.address", host
            ]
            
            if auto_open:
                cmd.extend(["--server.headless", "false"])
            else:
                cmd.extend(["--server.headless", "true"])
            
            print(f"🚀 Launching XFIN Streamlit app on http://{{host}}:{{port}}")
            print("Press Ctrl+C to stop the server")
            
            # Run the command
            process = subprocess.run(cmd)
            
        except KeyboardInterrupt:
            print("\\n✋ Streamlit app stopped by user")
        except Exception as e:
            print(f"❌ Error launching app: {{e}}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_app_path)
            except:
                pass

def launch_streamlit_app(port=8501, host="localhost", auto_open=True):
    """
    Launch the XFIN Streamlit app directly
    
    Parameters:
    -----------
    port : int, default 8501
        Port number for the Streamlit server
    host : str, default "localhost"  
        Host address for the server
    auto_open : bool, default True
        Whether to automatically open the browser
        
    Example:
    --------
    >>> from XFIN import launch_streamlit_app
    >>> launch_streamlit_app(port=8502)
    """
    app = StreamlitApp()
    app.launch_app(port=port, host=host, auto_open=auto_open) 