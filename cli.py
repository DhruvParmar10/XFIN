#!/usr/bin/env python3
"""
XFIN Command-Line Interface (CLI)
==================================

A comprehensive CLI for XFIN's three main modules:
- Credit Risk & XAI
- Stress Testing
- ESG Analysis

Author: Dhruv Parmar & Rishabh Bhangale
Version: 0.1.0
"""

import argparse
import sys
import os
from pathlib import Path


def print_banner():
    """Print XFIN banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██╗  ██╗███████╗██╗███╗   ██╗                         ║
║   ╚██╗██╔╝██╔════╝██║████╗  ██║                         ║
║    ╚███╔╝ █████╗  ██║██╔██╗ ██║                         ║
║    ██╔██╗ ██╔══╝  ██║██║╚██╗██║                         ║
║   ██╔╝ ██╗██║     ██║██║ ╚████║                         ║
║   ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═══╝                         ║
║                                                           ║
║   Financial Risk Analysis & Explainable AI                ║
║   v0.1.0                                                  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def launch_credit_dashboard(args):
    """Launch Credit Risk & XAI dashboard."""
    print("🚀 Launching Credit Risk & XAI Dashboard...")
    print("-" * 60)
    
    try:
        import subprocess
        
        # Build streamlit command
        cmd = ['streamlit', 'run', 'XFIN/app.py']
        
        if args.port:
            cmd.extend(['--server.port', str(args.port)])
        
        if args.host:
            cmd.extend(['--server.address', args.host])
        
        print(f"📊 Dashboard URL: http://{args.host or 'localhost'}:{args.port or 8501}")
        print("💡 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n💡 Make sure Streamlit is installed: pip install streamlit")
        sys.exit(1)


def launch_stress_dashboard(args):
    """Launch Stress Testing dashboard."""
    print("🚀 Launching Stress Testing Dashboard...")
    print("-" * 60)
    
    try:
        import subprocess
        
        # Build streamlit command
        cmd = ['streamlit', 'run', 'XFIN/stress_app.py']
        
        if args.port:
            cmd.extend(['--server.port', str(args.port)])
        
        if args.host:
            cmd.extend(['--server.address', args.host])
        
        print(f"📊 Dashboard URL: http://{args.host or 'localhost'}:{args.port or 8502}")
        print("💡 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n💡 Make sure Streamlit is installed: pip install streamlit")
        sys.exit(1)


def launch_unified_dashboard(args):
    """Launch Unified Dashboard (all modules)."""
    print("🚀 Launching Unified Dashboard (Credit + Stress + ESG)...")
    print("-" * 60)
    
    try:
        import subprocess
        
        # Build streamlit command
        cmd = ['streamlit', 'run', 'unified_dashboard.py']
        
        if args.port:
            cmd.extend(['--server.port', str(args.port)])
        
        if args.host:
            cmd.extend(['--server.address', args.host])
        
        print(f"📊 Dashboard URL: http://{args.host or 'localhost'}:{args.port or 8503}")
        print("💡 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n💡 Make sure Streamlit is installed: pip install streamlit")
        sys.exit(1)


def run_stress_analysis(args):
    """Run stress testing analysis from command line."""
    print("🧪 Running Stress Testing Analysis...")
    print("-" * 60)
    
    if not args.portfolio:
        print("❌ Error: --portfolio argument is required")
        print("💡 Usage: xfin stress-analyze --portfolio my_portfolio.csv")
        sys.exit(1)
    
    if not os.path.exists(args.portfolio):
        print(f"❌ Error: Portfolio file not found: {args.portfolio}")
        sys.exit(1)
    
    try:
        import pandas as pd
        # Add parent directory to path so XFIN can be imported
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from XFIN import StressAnalyzer
        
        # Load portfolio
        print(f"📂 Loading portfolio: {args.portfolio}")
        if args.portfolio.endswith('.csv'):
            portfolio = pd.read_csv(args.portfolio)
        elif args.portfolio.endswith(('.xlsx', '.xls')):
            portfolio = pd.read_excel(args.portfolio)
        else:
            print("❌ Error: Unsupported file format. Use .csv, .xlsx, or .xls")
            sys.exit(1)
        
        print(f"✅ Loaded {len(portfolio)} holdings")
        
        # Initialize analyzer
        stress_analyzer = StressAnalyzer()
        
        # Get scenario
        scenario_name = args.scenario or 'market_correction'
        print(f"🎯 Testing scenario: {scenario_name}")
        
        # Run stress test analysis
        results = stress_analyzer.explain_stress_impact(portfolio, scenario_name)
        
        # Display results
        print("\n📊 Stress Test Results:")
        print(f"   Current Portfolio Value: ₹{results.get('portfolio_value', 0):,.2f}")
        print(f"   Stressed Portfolio Value: ₹{results.get('stressed_value', 0):,.2f}")
        print(f"   Estimated Loss: ₹{results.get('dollar_loss', 0):,.2f}")
        print(f"   Impact: {results.get('impact_percent', 0):.2f}%")
        
        # Export if requested
        if args.output:
            output_file = args.output
            # Create results DataFrame
            results_df = pd.DataFrame([{
                'Scenario': scenario_name,
                'Portfolio_Value': results.get('portfolio_value', 0),
                'Stressed_Value': results.get('stressed_value', 0),
                'Dollar_Loss': results.get('dollar_loss', 0),
                'Impact_Percent': results.get('impact_percent', 0)
            }])
            results_df.to_csv(output_file, index=False)
            print(f"\n✅ Results exported to: {output_file}")
        
        print("\n✅ Analysis complete!")
        
    except ImportError as e:
        print(f"❌ Error: Missing required package: {e}")
        print("💡 Install XFIN: pip install xfin-xai")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_esg_analysis(args):
    """Run ESG analysis from command line."""
    print("🌱 Running ESG Analysis...")
    print("-" * 60)
    
    if not args.portfolio:
        print("❌ Error: --portfolio argument is required")
        print("💡 Usage: xfin esg-analyze --portfolio my_portfolio.csv")
        sys.exit(1)
    
    if not os.path.exists(args.portfolio):
        print(f"❌ Error: Portfolio file not found: {args.portfolio}")
        sys.exit(1)
    
    try:
        import pandas as pd
        from XFIN.esg import ESGScoringEngine
        
        # Load portfolio
        print(f"📂 Loading portfolio: {args.portfolio}")
        if args.portfolio.endswith('.csv'):
            portfolio = pd.read_csv(args.portfolio)
        elif args.portfolio.endswith(('.xlsx', '.xls')):
            portfolio = pd.read_excel(args.portfolio)
        else:
            print("❌ Error: Unsupported file format. Use .csv, .xlsx, or .xls")
            sys.exit(1)
        
        print(f"✅ Loaded {len(portfolio)} holdings")
        
        # Initialize ESG engine
        api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
        if api_key:
            esg_engine = ESGScoringEngine(api_key=api_key)
            print("🔑 Using OpenRouter API for enhanced analysis")
        else:
            esg_engine = ESGScoringEngine()
            print("ℹ️  Using basic ESG analysis (add --api-key for AI insights)")
        
        print("\n📊 Scoring holdings...")
        
        # Score each holding
        esg_scores = []
        for idx, row in portfolio.iterrows():
            ticker = row['Ticker']
            sector = row['Sector']
            company = row.get('Company_Name', ticker.split('.')[0])
            
            score = esg_engine.score_security(
                ticker=ticker,
                company_name=company,
                sector=sector
            )
            
            esg_scores.append({
                'Ticker': ticker,
                'ESG_Score': score.get('overall_score', 50.0),
                'E_Score': score.get('environmental_score', 50.0),
                'S_Score': score.get('social_score', 50.0),
                'G_Score': score.get('governance_score', 50.0)
            })
            
            print(f"   ✅ {ticker}: {score.get('overall_score', 50.0):.2f}/100")
        
        # Merge with portfolio
        esg_df = pd.DataFrame(esg_scores)
        portfolio_esg = portfolio.merge(esg_df, on='Ticker')
        
        # Calculate portfolio ESG
        if 'Current_Value' not in portfolio_esg.columns:
            portfolio_esg['Current_Value'] = portfolio_esg['Quantity'] * portfolio_esg['Current_Price']
        
        portfolio_esg['Weight'] = portfolio_esg['Current_Value'] / portfolio_esg['Current_Value'].sum()
        portfolio_score = (portfolio_esg['ESG_Score'] * portfolio_esg['Weight']).sum()
        
        # Rating
        def get_rating(score):
            if score >= 80: return 'AAA (Excellent)'
            elif score >= 70: return 'AA (Very Good)'
            elif score >= 60: return 'A (Good)'
            elif score >= 50: return 'BBB (Average)'
            elif score >= 40: return 'BB (Below Average)'
            else: return 'B (Poor)'
        
        # Display results
        print(f"\n📊 Portfolio ESG Score: {portfolio_score:.2f}/100")
        print(f"   Rating: {get_rating(portfolio_score)}")
        
        # Component scores
        e_score = (portfolio_esg['E_Score'] * portfolio_esg['Weight']).sum()
        s_score = (portfolio_esg['S_Score'] * portfolio_esg['Weight']).sum()
        g_score = (portfolio_esg['G_Score'] * portfolio_esg['Weight']).sum()
        
        print(f"\n   Environmental (E): {e_score:.2f}/100")
        print(f"   Social (S): {s_score:.2f}/100")
        print(f"   Governance (G): {g_score:.2f}/100")
        
        # Export if requested
        if args.output:
            output_file = args.output
            portfolio_esg.to_csv(output_file, index=False)
            print(f"\n✅ Results exported to: {output_file}")
        
        print("\n✅ Analysis complete!")
        
    except ImportError as e:
        print(f"❌ Error: Missing required package: {e}")
        print("💡 Install XFIN: pip install xfin-xai")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)


def show_info(args):
    """Show XFIN library information."""
    print_banner()
    
    info_text = """
📦 XFIN v0.1.0 - Comprehensive Financial Risk Analysis

📚 Three Powerful Modules:
   1. 💳 Credit Risk & XAI - Explainable credit decisions
   2. 🧪 Stress Testing - Portfolio resilience analysis
   3. 🌱 ESG Analysis - Sustainable investment scoring

🚀 Quick Commands:
   xfin credit           # Launch credit risk dashboard
   xfin stress           # Launch stress testing dashboard
   xfin unified          # Launch all modules together
   xfin info             # Show this information

📖 Documentation:
   • GitHub: https://github.com/dhruvparmar10/XFIN
   • Docs: https://xfin-xai.readthedocs.io
   • PyPI: https://pypi.org/project/xfin-xai

📞 Support:
   • Email: dhruv.jparmar0@gmail.com
   • Issues: https://github.com/dhruvparmar10/XFIN/issues

💡 Examples:
   # Launch dashboards
   xfin credit --port 8501
   xfin stress --host 0.0.0.0 --port 8502
   
   # Command-line analysis
   xfin stress-analyze --portfolio data.csv --scenario recession
   xfin esg-analyze --portfolio data.csv --output results.csv

📄 License: MIT
👥 Authors: Dhruv Parmar & Rishabh Bhangale
"""
    print(info_text)


def list_scenarios(args):
    """List all available stress testing scenarios."""
    print("🎯 Available Stress Testing Scenarios")
    print("=" * 60)
    
    try:
        from XFIN.stress_testing import ScenarioGenerator
        
        scenario_gen = ScenarioGenerator()
        scenarios = scenario_gen.list_scenarios()
        
        print(f"\nFound {len(scenarios)} scenarios:\n")
        
        for i, scenario_name in enumerate(scenarios, 1):
            scenario = scenario_gen.get_scenario(scenario_name)
            print(f"{i}. {scenario.get('name', scenario_name.replace('_', ' ').title())}")
            print(f"   Key: {scenario_name}")
            print(f"   Description: {scenario.get('description', 'No description available')}")
            
            # Show sector impacts if available
            if scenario.get('factors'):
                factors = scenario['factors']
                print(f"   Sector Impacts: {len(factors)} sectors")
                
                # Show impact range
                impacts = list(factors.values())
                if impacts:
                    print(f"   Impact Range: {min(impacts):+.0f}% to {max(impacts):+.0f}%")
            
            # Show probability if available
            if 'probability' in scenario:
                prob = scenario['probability'] * 100
                print(f"   Probability: {prob:.0f}%")
            
            print()
        
        print(f"💡 Usage: xfin stress-analyze --portfolio data.csv --scenario <scenario_key>")
        print(f"   Example: xfin stress-analyze --portfolio data.csv --scenario {scenarios[0]}")
        
    except ImportError:
        print("❌ Error: XFIN not installed properly")
        print("💡 Install: pip install xfin-xai")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error listing scenarios: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    
    # Create main parser
    parser = argparse.ArgumentParser(
        prog='xfin',
        description='XFIN - Comprehensive Financial Risk Analysis & XAI Library',
        epilog='For more information, visit: https://github.com/dhruvparmar10/XFIN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add version
    parser.add_argument(
        '--version',
        action='version',
        version='XFIN v0.1.0'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available XFIN commands',
        dest='command',
        help='Command to execute'
    )
    
    # =========================================================================
    # CREDIT RISK COMMAND
    # =========================================================================
    credit_parser = subparsers.add_parser(
        'credit',
        help='Launch Credit Risk & XAI dashboard',
        description='Launch the interactive Credit Risk & XAI dashboard'
    )
    credit_parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port number for the dashboard (default: 8501)'
    )
    credit_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address (default: localhost, use 0.0.0.0 for all interfaces)'
    )
    credit_parser.set_defaults(func=launch_credit_dashboard)
    
    # =========================================================================
    # STRESS TESTING COMMAND
    # =========================================================================
    stress_parser = subparsers.add_parser(
        'stress',
        help='Launch Stress Testing dashboard',
        description='Launch the interactive Stress Testing dashboard'
    )
    stress_parser.add_argument(
        '--port',
        type=int,
        default=8502,
        help='Port number for the dashboard (default: 8502)'
    )
    stress_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address (default: localhost, use 0.0.0.0 for all interfaces)'
    )
    stress_parser.set_defaults(func=launch_stress_dashboard)
    
    # =========================================================================
    # UNIFIED DASHBOARD COMMAND
    # =========================================================================
    unified_parser = subparsers.add_parser(
        'unified',
        help='Launch Unified Dashboard (all modules)',
        description='Launch the unified dashboard with Credit Risk, Stress Testing, and ESG'
    )
    unified_parser.add_argument(
        '--port',
        type=int,
        default=8503,
        help='Port number for the dashboard (default: 8503)'
    )
    unified_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address (default: localhost, use 0.0.0.0 for all interfaces)'
    )
    unified_parser.set_defaults(func=launch_unified_dashboard)
    
    # =========================================================================
    # STRESS ANALYZE COMMAND (CLI Analysis)
    # =========================================================================
    stress_analyze_parser = subparsers.add_parser(
        'stress-analyze',
        help='Run stress testing analysis from command line',
        description='Analyze portfolio stress without launching the dashboard'
    )
    stress_analyze_parser.add_argument(
        '--portfolio',
        type=str,
        required=True,
        help='Path to portfolio CSV or Excel file (required)'
    )
    stress_analyze_parser.add_argument(
        '--scenario',
        type=str,
        default='market_correction',
        help='Stress scenario to test (default: market_correction)'
    )
    stress_analyze_parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results (CSV format)'
    )
    stress_analyze_parser.set_defaults(func=run_stress_analysis)
    
    # =========================================================================
    # ESG ANALYZE COMMAND (CLI Analysis)
    # =========================================================================
    esg_analyze_parser = subparsers.add_parser(
        'esg-analyze',
        help='Run ESG analysis from command line',
        description='Analyze portfolio ESG scores without launching the dashboard'
    )
    esg_analyze_parser.add_argument(
        '--portfolio',
        type=str,
        required=True,
        help='Path to portfolio CSV or Excel file (required)'
    )
    esg_analyze_parser.add_argument(
        '--api-key',
        type=str,
        help='OpenRouter API key for enhanced analysis (optional)'
    )
    esg_analyze_parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results (CSV format)'
    )
    esg_analyze_parser.set_defaults(func=run_esg_analysis)
    
    # =========================================================================
    # INFO COMMAND
    # =========================================================================
    info_parser = subparsers.add_parser(
        'info',
        help='Show XFIN library information',
        description='Display XFIN version, modules, and usage information'
    )
    info_parser.set_defaults(func=show_info)
    
    # =========================================================================
    # SCENARIOS COMMAND
    # =========================================================================
    scenarios_parser = subparsers.add_parser(
        'scenarios',
        help='List all available stress testing scenarios',
        description='Display all pre-configured stress testing scenarios'
    )
    scenarios_parser.set_defaults(func=list_scenarios)
    
    # =========================================================================
    # PARSE AND EXECUTE
    # =========================================================================
    
    # If no arguments, show banner and help
    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        print("\n💡 Quick start: xfin info")
        sys.exit(0)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


# Make the file runnable
if __name__ == "__main__":
    main()
