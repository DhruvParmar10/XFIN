#!/usr/bin/env python3
"""
XFIN Command Line Interface
"""
import argparse
import sys
from .app import launch_streamlit_app

def credit_command(args):
    """Handle the credit subcommand"""
    try:
        print("Starting XFIN Credit Risk Explainer...")
        print(f"Server will run on http://{args.host}:{args.port}")
        print("Upload your model (.pl) and dataset (.csv) files in the sidebar")
        print("Press Ctrl+C to stop the server")
        print("-" * 60)
        
        launch_streamlit_app(
            port=args.port,
            host=args.host,
            auto_open=True
        )
        
    except KeyboardInterrupt:
        print("\nXFIN Credit app stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting XFIN Credit app: {e}")
        sys.exit(1)

def main():
    """Main CLI entry point for XFIN"""
    parser = argparse.ArgumentParser(
        prog='xfin',
        description="XFIN - Privacy-Preserving Explainable AI for Financial Services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xfin credit                       # Launch Credit Risk Explainer
  xfin credit --port 8502           # Launch on custom port
  xfin credit --host 0.0.0.0        # Launch on all interfaces
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='XFIN 0.1.1'
    )
    
    # Create subparsers for different modules
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Credit Risk subcommand
    credit_parser = subparsers.add_parser(
        'credit',
        help='Launch Credit Risk Explainer',
        description='Privacy-preserving credit risk analysis with explainable AI'
    )
    
    credit_parser.add_argument(
        '--port', '-p',
        type=int,
        default=8501,
        help='Port number for the Streamlit server (default: 8501)'
    )
    
    credit_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address for the server (default: localhost)'
    )

    credit_parser.set_defaults(func=credit_command)    # Parse arguments
    args = parser.parse_args()
    
    # If no command provided, show help
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate command
    args.func(args)

def xfin_cli():
    """Entry point for xfin command"""
    main()

# Legacy entry point for backward compatibility
def credit_risk_cli():
    sys.argv = ['xfin', 'credit'] + sys.argv[1:]
    main()

if __name__ == "__main__":
    main()
