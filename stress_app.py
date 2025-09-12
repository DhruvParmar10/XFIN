import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from XFIN import StressTestingModule, StressPlotGenerator
import io

def parse_broker_csv(file_content):
    """Parse broker CSV files with dynamic header detection"""
    try:
        # Read the entire file content
        if isinstance(file_content, bytes):
            content = file_content.decode('utf-8')
        else:
            content = file_content
        
        lines = content.strip().split('\n')
        
        # Find the header row (look for Stock Name or similar)
        header_indicators = ['Stock Name', 'stock name', 'STOCK NAME', 'Symbol', 'Name of Security', 'Security Name', 'Company Name', 'Scrip/Contract']
        header_row_idx = None
        
        for i, line in enumerate(lines):
            if any(indicator in line for indicator in header_indicators):
                header_row_idx = i
                break
        
        if header_row_idx is None:
            st.error("Could not find stock data headers in the file")
            return None
        
        # Extract headers and data
        header_line = lines[header_row_idx]
        headers = [col.strip() for col in header_line.split(',')]
        
        # Get data rows (skip empty rows)
        data_rows = []
        for line in lines[header_row_idx + 1:]:
            if line.strip() and not line.startswith(',,,'):
                row = [col.strip() for col in line.split(',')]
                if len(row) >= len(headers) and any(row):  # Ensure row has data
                    data_rows.append(row[:len(headers)])  # Trim to header length
        
        if not data_rows:
            st.error("No stock data found in the file")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Clean up numeric columns
        numeric_columns = ['Quantity', 'Average buy price', 'Buy value', 'Closing price', 'Closing value', 'Unrealised P&L', 'Prev closing Price', 'Holding Weightage', 'Avg Trading Price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate invested value using Average buy price * Quantity
        if 'Average buy price' in df.columns and 'Quantity' in df.columns:
            df['Invested Value'] = df['Average buy price'] * df['Quantity']
        
        # Calculate current value if closing price is available
        if 'Closing price' in df.columns and 'Quantity' in df.columns:
            df['Current Value'] = df['Closing price'] * df['Quantity']
        
        # Remove rows with all NaN values in numeric columns
        df = df.dropna(subset=[col for col in numeric_columns if col in df.columns], how='all')
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing CSV file: {str(e)}")
        return None

def get_value_column(df):
    """Find the appropriate value column from the dataframe, prioritizing calculated values"""
    possible_columns = [
        'Invested Value',  # Our calculated column gets priority
        'Current Value',   # Another calculated column
        'Closing value', 'Closing Value', 'closing value', 'CLOSING VALUE',
        'Market Value', 'Market value', 'market value', 'MARKET VALUE',
        'Buy value', 'Buy Value', 'buy value', 'BUY VALUE',
        'Value', 'value', 'VALUE'
    ]
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    return None

def get_stock_name_column(df):
    """Find the appropriate stock name column"""
    possible_columns = [
        'Stock Name', 'stock name', 'Stock name', 'STOCK NAME',
        'Security Name', 'security name', 'Security name', 'SECURITY NAME',
        'Name', 'name', 'NAME', 'Symbol', 'symbol', 'SYMBOL'
    ]
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    return None

def create_stress_dashboard():
    # Streamlit page config
    st.set_page_config(page_title="XFIN Portfolio Stress Testing", page_icon="📈", layout="wide")
    
    st.title("📈 XFIN Portfolio Stress Testing")
    st.markdown("**AI-Powered Portfolio Analysis for Real-World Scenarios**")
    
    # Sidebar: Portfolio upload and display
    st.sidebar.header("📂 Portfolio Configuration")
    portfolio_file = st.sidebar.file_uploader("Upload Portfolio CSV", type=['csv'])
    
    # LLM Configuration section
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AI Analysis Settings")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenRouter API Key (Optional)",
        type="password",
        help="Enter your OpenRouter API key for enhanced AI explanations. Leave blank to use environment variable."
    )
    
    # LLM Model selection
    llm_model = st.sidebar.selectbox(
        "AI Model for Analysis",
        ["x-ai/grok-code-fast-1", "anthropic/claude-3-haiku", "openai/gpt-4o-mini"],
        help="Select the AI model for generating explanations"
    )
    
    # Enable/disable LLM
    use_llm = st.sidebar.checkbox(
        "Enable AI-Powered Recommendations",
        value=True,
        help="Get detailed AI-generated analysis and recommendations"
    )
    
    if use_llm:
        st.sidebar.success("✅ AI recommendations enabled")
    else:
        st.sidebar.info("ℹ️ Using basic analysis only")
    
    # Portfolio processing
    if portfolio_file:
        try:
            # Read file content
            file_content = portfolio_file.read()
            portfolio_data = parse_broker_csv(file_content)
            
            if portfolio_data is not None:
                st.sidebar.success("✅ Portfolio loaded!")
                
                # Debug: Show column names
                st.sidebar.write("**Columns detected:**")
                for col in portfolio_data.columns:
                    st.sidebar.write(f"- {col}")
                    
            else:
                st.sidebar.error("Failed to parse portfolio file")
        except Exception as e:
            st.sidebar.error(f"Error loading portfolio: {e}")
            portfolio_data = None
    else:
        portfolio_data = None
        st.sidebar.info("⬇️ Please upload your portfolio CSV")
    
    if portfolio_data is not None:
        # Find the appropriate columns
        value_col = get_value_column(portfolio_data)
        name_col = get_stock_name_column(portfolio_data)
        
        if value_col and name_col:
            # Calculate total portfolio value
            total_value = portfolio_data[value_col].sum()
            
            st.sidebar.markdown("**Your Portfolio Summary:**")
            st.sidebar.write(f"**Total Portfolio Value:** ₹{total_value:,.0f}")
            st.sidebar.write(f"**Value Source:** {value_col}")
            st.sidebar.write(f"**Number of Holdings:** {len(portfolio_data)}")
            
            # Show investment vs current value if both are available
            if 'Invested Value' in portfolio_data.columns and 'Current Value' in portfolio_data.columns:
                total_invested = portfolio_data['Invested Value'].sum()
                total_current = portfolio_data['Current Value'].sum()
                total_pnl = total_current - total_invested
                pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
                
                st.sidebar.write(f"**Total Invested:** ₹{total_invested:,.0f}")
                st.sidebar.write(f"**Current Value:** ₹{total_current:,.0f}")
                
                color = "🟢" if total_pnl >= 0 else "🔴"
                st.sidebar.write(f"**Total P&L:** {color} ₹{total_pnl:,.0f} ({pnl_pct:+.1f}%)")
            
            # Show portfolio stats from existing columns if available
            elif 'Unrealised P&L' in portfolio_data.columns:
                total_pnl = portfolio_data['Unrealised P&L'].sum()
                pnl_pct = (total_pnl / total_value) * 100 if total_value > 0 else 0
                color = "🟢" if total_pnl >= 0 else "🔴"
                st.sidebar.write(f"**Total P&L:** {color} ₹{total_pnl:,.0f} ({pnl_pct:+.1f}%)")
            
            # Show top holdings
            st.sidebar.markdown("**Top Holdings:**")
            try:
                top_holdings = portfolio_data.nlargest(5, value_col)
                for _, row in top_holdings.iterrows():
                    holding_val = row[value_col]
                    weight = (holding_val / total_value) * 100
                    stock_name = row[name_col]
                    
                    # Show quantity and average price if available
                    extra_info = ""
                    if 'Quantity' in row and 'Average buy price' in row:
                        qty = row['Quantity']
                        avg_price = row['Average buy price']
                        if pd.notna(qty) and pd.notna(avg_price):
                            extra_info = f" (Qty: {qty:.0f} @ ₹{avg_price:.2f})"
                    
                    st.sidebar.write(f"- {stock_name}: ₹{holding_val:,.0f} ({weight:.1f}%){extra_info}")
            except Exception as e:
                st.sidebar.write("Unable to display top holdings")
            
            # Show portfolio allocation pie chart
            try:
                analyzer = StressTestingModule(None, api_key=api_key if api_key else None).portfolio_analyzer
                composition = analyzer.analyze_portfolio(portfolio_data)['composition']
                plot_gen = StressPlotGenerator()
                fig_pie = plot_gen.create_allocation_pie(composition)
                st.sidebar.pyplot(fig_pie)
                plt.close(fig_pie)
            except Exception as e:
                st.sidebar.info("Portfolio allocation chart unavailable")
                
        else:
            st.sidebar.warning("⚠️ Required columns not found. Please check your CSV format.")
    
    # Initialize modules with API key
    model = type("MockModel", (), {"predict_stress_impact": lambda self, d, f: None})()
    explainer = StressTestingModule(model, api_key=api_key if api_key else None)
    plot_gen = StressPlotGenerator()
    
    # Scenario selection
    scenarios = explainer.scenario_generator.list_scenarios()
    scenario_names = {s: explainer.scenario_generator.get_scenario(s)['name'] for s in scenarios}
    
    st.subheader("🎯 Choose Your Scenario")
    selected_scenario = st.selectbox("Select Scenario:", scenarios, format_func=lambda x: scenario_names[x])
    
    # Show scenario description
    scenario_info = explainer.scenario_generator.get_scenario(selected_scenario)
    st.info(f"**{scenario_info['name']}**: {scenario_info['description']}")
    
    # Analysis type
    st.subheader("⚙️ Analysis Options")
    analysis_type = st.radio("What would you like to see?", ["Single Scenario Impact", "Compare Multiple Scenarios"])
    
    compare_keys = []
    if analysis_type == "Compare Multiple Scenarios" and portfolio_data is not None:
        # Fix for "No results" issue - ensure we don't filter out the selected scenario
        st.write("**Select Additional Scenarios:**")
        for scenario_key in scenarios:
            if st.checkbox(scenario_names[scenario_key], value=(scenario_key == selected_scenario), key=scenario_key):
                if scenario_key not in compare_keys:
                    compare_keys.append(scenario_key)
    
    # Run analysis
    if st.button("🚀 Analyze My Portfolio", type="primary"):
        if portfolio_data is None:
            st.error("Please upload a portfolio CSV first.")
            st.stop()
        
        if analysis_type == "Single Scenario Impact":
            # Single scenario
            with st.spinner("Analyzing portfolio stress impact..."):
                result = explainer.explain_stress_impact(portfolio_data, selected_scenario)
                impact = result['stress_impact']
                
                st.subheader("📋 Your Portfolio Analysis Results")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Portfolio Impact", f"{impact['impact_percentage']:.1f}%")
                
                with col2:
                    st.metric("Risk Assessment", f"{impact['risk_level']}")
                
                with col3:
                    st.metric("Expected Recovery", f"{impact['recovery_months']:.0f} months")
                
                with col4:
                    st.metric("Worst Case (VaR)", f"{impact['var_95']*100:.1f}%")
                
                # Show value calculation details
                if portfolio_data is not None and value_col:
                    total_portfolio_value = portfolio_data[value_col].sum()
                    impact_amount = total_portfolio_value * abs(impact['impact_percentage']) / 100
                    
                    st.info(f"💰 **Value Calculation**: Using '{value_col}' column | Portfolio Value: ₹{total_portfolio_value:,.0f} | Potential Impact: ₹{impact_amount:,.0f}")
                
                # Improved Plot
                try:
                    fig = plot_gen.create_stress_impact_plot(result)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Visualization unavailable: {e}")
                
                # AI-Powered Recommendations
                st.markdown("### 💡 What Should You Do?")
                if use_llm:
                    with st.spinner("🤖 Generating AI-powered recommendations..."):
                        try:
                            recs = explainer.generate_recommendations(portfolio_data, result)
                            st.markdown(recs)
                        except Exception as e:
                            st.error(f"AI analysis failed: {e}")
                            st.markdown(f"""
                            **Basic Analysis:**
                            - Portfolio Impact: {impact['impact_percentage']:.1f}%
                            - Risk Level: {impact['risk_level']}
                            - Recovery Time: {impact['recovery_months']:.0f} months
                            
                            For detailed recommendations, please check your API key or try again later.
                            """)
                else:
                    st.markdown(f"""
                    **Basic Analysis Summary:**
                    - Portfolio Impact: {impact['impact_percentage']:.1f}%
                    - Risk Level: {impact['risk_level']}
                    - Recovery Time: {impact['recovery_months']:.0f} months
                    
                    Enable AI-powered recommendations in the sidebar for detailed analysis and actionable advice.
                    """)
        
        else:
            # Comparison - fix for when compare_keys is empty
            if not compare_keys:
                st.warning("Please select at least one scenario to compare.")
            else:
                with st.spinner("Comparing scenarios..."):
                    comp_df = explainer.compare_scenarios(portfolio_data, compare_keys)
                    
                    st.subheader("📊 Multi-Scenario Comparison")
                    st.dataframe(comp_df.style.format({'impact_percentage': '{:.1f}%'}), use_container_width=True)
                    
                    try:
                        fig2 = plot_gen.create_scenario_comparison(comp_df)
                        st.pyplot(fig2)
                        plt.close(fig2)
                    except Exception as e:
                        st.warning(f"Comparison chart unavailable: {e}")
                    
                    # AI Summary for comparison
                    if use_llm and len(compare_keys) > 1:
                        st.markdown("### 🤖 AI Analysis Summary")
                        with st.spinner("Generating comparison insights..."):
                            try:
                                # Create a summary for multiple scenarios
                                summary_prompt = f"""
                                MULTI-SCENARIO COMPARISON ANALYSIS
                                
                                Portfolio Value: ₹{portfolio_data[value_col].sum():,.0f}
                                Value Source: {value_col}
                                Scenarios Compared: {', '.join([scenario_names[k] for k in compare_keys])}
                                
                                Results Summary:
                                {comp_df.to_string()}
                                
                                Provide a brief comparative analysis highlighting:
                                1. Which scenario poses the highest risk
                                2. Key differences between scenarios  
                                3. Overall portfolio resilience assessment
                                4. Top 2 recommendations for risk mitigation
                                """
                                
                                comparison_analysis = explainer.generate_recommendations(
                                    portfolio_data,
                                    {'stress_impact': {'impact_percentage': 0, 'risk_level': 'Comparison'},
                                     'scenario': {'name': 'Multi-Scenario Analysis'},
                                     'portfolio_analysis': explainer.portfolio_analyzer.analyze_portfolio(portfolio_data)}
                                )
                                
                                st.markdown(comparison_analysis)
                            except Exception as e:
                                st.error(f"AI comparison analysis failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**XFIN Portfolio Stress Testing** - AI-Powered Risk Analysis for Smarter Investment Decisions")
    if use_llm:
        st.caption("🤖 Powered by Advanced AI for Personalized Investment Insights")
    else:
        st.caption("💡 Enable AI recommendations for personalized insights")

def launch_stress_dashboard():
    create_stress_dashboard()

if __name__ == "__main__":
    launch_stress_dashboard()