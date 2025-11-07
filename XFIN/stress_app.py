import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import os

# Handle imports whether running as module or standalone
try:
    from XFIN.stress_testing import StressTestingEngine
    from XFIN.stress_plots import StressPlotGenerator
    from XFIN.esg import ESGScoringEngine
    from XFIN.esg_plots import ESGPlotGenerator
except ImportError:
    # Running from within XFIN directory
    from stress_testing import StressTestingEngine
    from stress_plots import StressPlotGenerator
    from esg import ESGScoringEngine
    from esg_plots import ESGPlotGenerator

class UniversalBrokerCSVParser:
    """Universal parser that can handle CSV files from any broker with intelligent format detection"""
    
    def __init__(self):
        # Comprehensive mapping of broker terminologies to standard names
        self.column_mappings = {
            # Stock/Security Name variations
            'stock_name': [
                'Stock Name', 'stock name', 'STOCK NAME', 'StockName',
                'Security Name', 'security name', 'SECURITY NAME', 'SecurityName',
                'Scrip/Contract', 'Scrip', 'Contract', 'Symbol', 'SYMBOL',
                'Company Name', 'company name', 'COMPANY NAME', 'CompanyName',
                'Name of Security', 'Security', 'Instrument', 'ISIN',
                'Script Name', 'Name', 'NAME'
            ],
            
            # Quantity variations
            'quantity': [
                'Quantity', 'quantity', 'QUANTITY', 'Qty', 'QTY', 'qty',
                'Holdings', 'holdings', 'HOLDINGS', 'Units', 'UNITS',
                'Shares', 'shares', 'SHARES', 'No. of Shares'
            ],
            
            # Average/Buy Price variations
            'avg_price': [
                'Average buy price', 'Avg buy price', 'AVG BUY PRICE',
                'Avg Trading Price', 'Average Price', 'Buy Price', 'Purchase Price',
                'Avg Rate', 'Average Rate', 'Rate', 'Price', 'Unit Price',
                'Cost Price', 'Acquisition Price', 'Buy Rate'
            ],
            
            # Current/Market Price variations
            'current_price': [
                'Closing price', 'Closing Price', 'CLOSING PRICE', 'Close Price',
                'Current Price', 'Market Price', 'LTP', 'Last Traded Price',
                'Current Rate', 'Market Rate', 'Live Price', 'CMP'
            ],
            
            # Current/Market Value variations
            'current_value': [
                'Closing value', 'Closing Value', 'CLOSING VALUE',
                'Current Value', 'Market Value', 'Present Value',
                'Market Value as of last trading day', 'Current Market Value',
                'Total Value', 'Value', 'Holdings Value', 'Investment Value'
            ],
            
            # Invested/Buy Value variations
            'invested_value': [
                'Buy value', 'Buy Value', 'BUY VALUE', 'Purchase Value',
                'Invested Value', 'Investment Amount', 'Cost Value',
                'Total Cost', 'Acquisition Value', 'Book Value'
            ],
            
            # P&L variations
            'pnl': [
                'Unrealised P&L', 'P&L', 'PnL', 'Profit/Loss', 'Gain/Loss',
                'Unrealized P&L', 'Total P&L', 'Net P&L', 'Day P&L'
            ]
        }
        
        # Broker format fingerprints for detection
        self.broker_signatures = {
            'Zerodha': ['ISIN', 'Exchange', 'Last traded price'],
            'Upstox': ['Scrip/Contract', 'Market Value as of last trading day'],
            'Angel Broking': ['Symbol', 'Exchange', 'Net Qty'],
            'HDFC Securities': ['Security Name', 'Market Value', 'Unrealised P&L'],
            'ICICI Direct': ['Stock Name', 'Holdings', 'Current Value'],
            'Kotak Securities': ['Scrip Name', 'Quantity', 'Market Price'],
            'Groww': ['Stock Name', 'Qty', 'Current Price'],
            'Generic': []  # Fallback
        }
        
        # Financial indicators for header detection
        self.financial_indicators = [
            'price', 'value', 'quantity', 'qty', 'amount', 'rate', 'cost',
            'market', 'current', 'total', 'investment', 'holding', 'shares',
            'units', 'stock', 'security', 'scrip', 'symbol', 'isin'
        ]
    
    def detect_broker_format(self, headers):
        """Detect broker format based on column signatures"""
        header_text = ' '.join(headers).lower()
        
        for broker, signatures in self.broker_signatures.items():
            if broker == 'Generic':
                continue
            
            matches = sum(1 for sig in signatures if sig.lower() in header_text)
            if matches >= len(signatures) * 0.6:  # 60% match threshold
                return broker
        
        return 'Generic'
    
    def find_header_row(self, lines):
        """Smart header detection using pattern-based approach"""
        for i, line in enumerate(lines):
            if not line.strip():  # Skip empty lines
                continue
                
            # Skip obvious metadata rows
            line_lower = line.lower()
            if any(skip_term in line_lower for skip_term in 
                   ['client', 'account', 'report', 'date', 'total portfolio', 'grand total']):
                continue
            
            # Split and clean the line
            cells = [cell.strip() for cell in line.split(',')]
            if len(cells) < 3:  # Need at least 3 columns
                continue
            
            # Count financial indicators in this row
            financial_count = 0
            for cell in cells:
                cell_lower = cell.lower()
                if any(indicator in cell_lower for indicator in self.financial_indicators):
                    financial_count += 1
            
            # If this row has 3+ financial indicators, it's likely the header
            if financial_count >= 3:
                return i, self.detect_broker_format(cells)
        
        raise ValueError("Could not detect header row - no row found with sufficient financial indicators")
    
    def map_columns(self, headers):
        """Map broker-specific column names to standard names"""
        mapped = {}
        
        for standard_name, variations in self.column_mappings.items():
            for header in headers:
                if header in variations:
                    mapped[standard_name] = header
                    break
        
        return mapped
    
    def clean_numeric_value(self, value):
        """Clean various number formats from different brokers"""
        if pd.isna(value) or value == '':
            return 0.0
        
        value_str = str(value)
        
        # Handle negative values in brackets: (1000) -> -1000
        if value_str.startswith('(') and value_str.endswith(')'):
            value_str = '-' + value_str[1:-1]
        
        # Remove currency symbols and commas
        value_str = value_str.replace('₹', '').replace('$', '').replace(',', '')
        
        # Remove percentage signs if present
        value_str = value_str.replace('%', '')
        
        # Extract just numbers, decimal points, and minus signs
        import re
        clean_value = re.sub(r'[^\d.-]', '', value_str)
        
        try:
            return float(clean_value) if clean_value else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def extract_data_rows(self, lines, header_idx, headers):
        """Extract valid data rows, skipping metadata and footer rows"""
        data_rows = []
        
        for line in lines[header_idx + 1:]:
            if not line.strip():  # Skip empty lines
                continue
            
            # Skip summary/total rows
            line_lower = line.lower()
            if any(skip_term in line_lower for skip_term in 
                   ['total', 'grand total', 'portfolio total', 'summary', 'net amount']):
                continue
            
            row_data = [cell.strip() for cell in line.split(',')]
            
            # Ensure row has enough columns
            if len(row_data) < len(headers):
                row_data.extend([''] * (len(headers) - len(row_data)))
            elif len(row_data) > len(headers):
                row_data = row_data[:len(headers)]
            
            # Skip rows without a stock name (first financial column should have value)
            if not row_data[0] or row_data[0].lower() in ['', 'na', 'n/a', 'null']:
                continue
            
            data_rows.append(row_data)
        
        return data_rows

def clean_and_handle_missing_data(df, column_mapping):
    """
    Handle missing data in portfolio CSV files:
    1. Remove rows where stock name itself is missing
    2. Calculate missing buy/sell prices from quantity and P&L
    3. Track and report all changes made
    """
    changes = {
        "Removed Rows (Missing Stock Name)": [],
        "Calculated Missing Buy Prices": [],
        "Calculated Missing Sell Prices": [],
        "Calculated Missing Current Values": [],
        "Calculated Missing Invested Values": [],
        "Rows Removed (Insufficient Data)": []
    }
    
    original_count = len(df)
    
    # Get column names from mapping
    stock_col = column_mapping.get('stock_name')
    qty_col = column_mapping.get('quantity')
    avg_price_col = column_mapping.get('avg_price')
    current_price_col = column_mapping.get('current_price')
    current_value_col = column_mapping.get('current_value')
    invested_value_col = column_mapping.get('invested_value')
    pnl_col = column_mapping.get('pnl')
    
    # Find actual column names in DataFrame
    stock_col_actual = None
    for col in df.columns:
        if stock_col and stock_col in str(col):
            stock_col_actual = col
            break
    
    if not stock_col_actual:
        # Try to find any column that looks like a stock name
        for col in df.columns:
            if 'stock' in col.lower() or 'security' in col.lower() or 'name' in col.lower():
                stock_col_actual = col
                break
    
    # Step 1: Remove rows with missing stock names
    if stock_col_actual and stock_col_actual in df.columns:
        rows_to_remove = df[stock_col_actual].isna() | (df[stock_col_actual] == '') | (df[stock_col_actual].astype(str).str.strip() == '')
        removed_indices = df[rows_to_remove].index.tolist()
        
        if len(removed_indices) > 0:
            df = df[~rows_to_remove].copy()
            changes["Removed Rows (Missing Stock Name)"].append(
                f"Removed {len(removed_indices)} rows with missing stock names"
            )
    
    # Step 2: Calculate missing prices from quantity and P&L
    # Find P&L column in actual DataFrame
    pnl_col_actual = None
    for col in df.columns:
        if any(term in col.lower() for term in ['p&l', 'pnl', 'profit', 'loss', 'gain']):
            pnl_col_actual = col
            break
    
    qty_col_actual = None
    for col in df.columns:
        if any(term in col.lower() for term in ['quantity', 'qty', 'units', 'shares', 'holdings']):
            qty_col_actual = col
            break
    
    # Find current/market value column (flexible matching)
    current_value_col_actual = None
    for col in df.columns:
        if any(term in col.lower() for term in ['current value', 'market value', 'closing value', 'present value']):
            current_value_col_actual = col
            break
    
    # Find invested/buy value column (flexible matching)
    invested_value_col_actual = None
    for col in df.columns:
        if any(term in col.lower() for term in ['invested value', 'buy value', 'cost value', 'investment value', 'purchase value']):
            invested_value_col_actual = col
            break
    
    # Find previous closing price column (for port1 scenario)
    prev_closing_price_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['previous closing', 'prev closing', 'last closing', 'previous close']):
            prev_closing_price_col = col
            break
    
    # Find current/closing price column
    current_price_col_actual = None
    for col in df.columns:
        if any(term in col.lower() for term in ['closing price', 'current price', 'market price', 'ltp', 'last traded']):
            current_price_col_actual = col
            break
    
    # Find buy/average price column
    avg_price_col_actual = None
    for col in df.columns:
        if any(term in col.lower() for term in ['average buy', 'avg buy', 'buy price', 'purchase price', 'avg price']):
            avg_price_col_actual = col
            break
    
    # Calculate missing values row by row
    for idx in df.index:
        row = df.loc[idx]
        stock_name = row.get(stock_col_actual, 'Unknown') if stock_col_actual else 'Unknown'
        
        # Get values (convert to numeric, handle NaN)
        qty = pd.to_numeric(row.get(qty_col_actual, 0), errors='coerce') if qty_col_actual else 0
        pnl = pd.to_numeric(row.get(pnl_col_actual, 0), errors='coerce') if pnl_col_actual else 0
        current_val = pd.to_numeric(row.get(current_value_col_actual, None), errors='coerce') if current_value_col_actual else None
        invested_val = pd.to_numeric(row.get(invested_value_col_actual, None), errors='coerce') if invested_value_col_actual else None
        
        # Get previous closing price if available
        prev_closing_price = pd.to_numeric(row.get(prev_closing_price_col, None), errors='coerce') if prev_closing_price_col else None
        
        # Skip if quantity is 0 or NaN
        if pd.isna(qty) or qty == 0:
            continue
        
        # SPECIAL CASE: If no current price but have previous closing price, use it as current price
        if prev_closing_price_col and not pd.isna(prev_closing_price) and prev_closing_price > 0:
            # Check if current price is missing
            current_price_value = None
            if current_price_col_actual:
                current_price_value = pd.to_numeric(row.get(current_price_col_actual, None), errors='coerce')
            
            # If current price is missing, use previous closing price
            if pd.isna(current_price_value):
                if current_price_col_actual:
                    df.at[idx, current_price_col_actual] = prev_closing_price
                else:
                    # Create new column if it doesn't exist
                    if 'Current Price' not in df.columns:
                        df['Current Price'] = None
                    df.at[idx, 'Current Price'] = prev_closing_price
                    current_price_col_actual = 'Current Price'
                
                changes["Calculated Missing Sell Prices"].append(
                    f"{stock_name}: Used Previous Closing Price = ₹{prev_closing_price:,.2f} as Current Price"
                )
                
                # Calculate current value from previous closing price
                if pd.isna(current_val):
                    calculated_current_val = prev_closing_price * qty
                    if current_value_col_actual:
                        df.at[idx, current_value_col_actual] = calculated_current_val
                    else:
                        if 'Current Value' not in df.columns:
                            df['Current Value'] = None
                        df.at[idx, 'Current Value'] = calculated_current_val
                        current_value_col_actual = 'Current Value'
                    
                    changes["Calculated Missing Current Values"].append(
                        f"{stock_name}: Calculated Current Value = ₹{calculated_current_val:,.2f} (Previous Closing Price × Quantity)"
                    )
                    current_val = calculated_current_val
                    
                    # Now calculate P&L if we have buy price or invested value
                    if not pd.isna(invested_val) and invested_val > 0:
                        calculated_pnl = calculated_current_val - invested_val
                        if pnl_col_actual:
                            df.at[idx, pnl_col_actual] = calculated_pnl
                        else:
                            if 'P&L' not in df.columns:
                                df['P&L'] = None
                            df.at[idx, 'P&L'] = calculated_pnl
                            pnl_col_actual = 'P&L'
                        
                        changes["Calculated Missing Invested Values"].append(
                            f"{stock_name}: Calculated P&L = ₹{calculated_pnl:,.2f} (Current Value - Invested Value)"
                        )
                        pnl = calculated_pnl
        
        # Calculate missing invested value from current value and P&L
        if pd.isna(invested_val) and not pd.isna(current_val) and not pd.isna(pnl):
            calculated_invested = current_val - pnl
            if calculated_invested > 0:
                df.at[idx, invested_value_col_actual or 'Invested Value'] = calculated_invested
                changes["Calculated Missing Invested Values"].append(
                    f"{stock_name}: Calculated Invested Value = ₹{calculated_invested:,.2f} (from Current Value - P&L)"
                )
                invested_val = calculated_invested
        
        # Calculate missing current value from invested value and P&L
        if pd.isna(current_val) and not pd.isna(invested_val) and not pd.isna(pnl):
            calculated_current = invested_val + pnl
            if calculated_current > 0:
                df.at[idx, current_value_col_actual or 'Current Value'] = calculated_current
                changes["Calculated Missing Current Values"].append(
                    f"{stock_name}: Calculated Current Value = ₹{calculated_current:,.2f} (from Invested Value + P&L)"
                )
                current_val = calculated_current
        
        # Calculate buy price if missing but we have invested value and quantity
        if invested_value_col_actual and not pd.isna(invested_val) and invested_val > 0:
            buy_price = invested_val / qty
            # Check if buy price column exists and is NaN
            for col in df.columns:
                if any(term in col.lower() for term in ['average buy', 'avg buy', 'buy price', 'purchase price']):
                    if pd.isna(row.get(col, None)):
                        df.at[idx, col] = buy_price
                        changes["Calculated Missing Buy Prices"].append(
                            f"{stock_name}: Buy Price = ₹{buy_price:,.2f} (Invested Value ÷ Quantity)"
                        )
                    break
        
        # Calculate sell/current price if missing but we have current value and quantity
        if current_value_col_actual and not pd.isna(current_val) and current_val > 0:
            current_price = current_val / qty
            # Check if current price column exists and is NaN
            for col in df.columns:
                if any(term in col.lower() for term in ['closing price', 'current price', 'market price', 'ltp']):
                    if pd.isna(row.get(col, None)):
                        df.at[idx, col] = current_price
                        changes["Calculated Missing Sell Prices"].append(
                            f"{stock_name}: Current Price = ₹{current_price:,.2f} (Current Value ÷ Quantity)"
                        )
                    break
    
    # Step 3: Final check - remove rows that still don't have essential data
    essential_cols = [current_value_col_actual, invested_value_col_actual]
    rows_to_remove_final = []
    
    for idx in df.index:
        # Must have at least one value column
        has_value = False
        for col in essential_cols:
            if col and col in df.columns:
                val = pd.to_numeric(df.at[idx, col], errors='coerce')
                if not pd.isna(val) and val > 0:
                    has_value = True
                    break
        
        if not has_value:
            rows_to_remove_final.append(idx)
    
    if rows_to_remove_final:
        stock_names_removed = []
        for idx in rows_to_remove_final:
            stock = df.at[idx, stock_col_actual] if stock_col_actual else f"Row {idx}"
            stock_names_removed.append(str(stock))
        
        df = df.drop(rows_to_remove_final)
        changes["Rows Removed (Insufficient Data)"].append(
            f"Removed {len(rows_to_remove_final)} rows lacking essential value data: {', '.join(stock_names_removed[:5])}" +
            (f" and {len(stock_names_removed) - 5} more" if len(stock_names_removed) > 5 else "")
        )
    
    # Reset index
    df = df.reset_index(drop=True)
    
    final_count = len(df)
    total_removed = original_count - final_count
    
    if total_removed > 0:
        st.info(f"ℹ️ Processed {original_count} rows → Kept {final_count} valid rows (removed {total_removed})")
    
    # Remove empty change categories
    changes = {k: v for k, v in changes.items() if v}
    
    return df, changes

def parse_broker_csv(file_content):
    """Enhanced universal broker CSV parser with intelligent format detection"""
    parser = UniversalBrokerCSVParser()
    
    try:
        # Read the entire file content
        if isinstance(file_content, bytes):
            content = file_content.decode('utf-8')
        else:
            content = str(file_content)
        
        lines = content.strip().split('\n')
        if len(lines) < 2:
            st.error("File appears to be empty or has insufficient data")
            return None
        
        # Smart header detection
        header_idx, broker_format = parser.find_header_row(lines)
        
        # Extract headers and map to standard names
        header_line = lines[header_idx]
        raw_headers = [col.strip() for col in header_line.split(',')]
        column_mapping = parser.map_columns(raw_headers)
        
        # Extract data rows
        data_rows = parser.extract_data_rows(lines, header_idx, raw_headers)
        
        if not data_rows:
            st.error("No valid data rows found after header")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=raw_headers)
        
        # Apply universal column cleaning
        for standard_name, broker_column in column_mapping.items():
            if broker_column in df.columns:
                if standard_name in ['quantity', 'avg_price', 'current_price', 'current_value', 'invested_value', 'pnl']:
                    df[broker_column] = df[broker_column].apply(parser.clean_numeric_value)
        
        # Calculate missing values using available data
        if 'current_value' not in column_mapping:
            # Try to calculate current value
            current_price_col = column_mapping.get('current_price')
            quantity_col = column_mapping.get('quantity')
            if current_price_col and quantity_col and current_price_col in df.columns and quantity_col in df.columns:
                df['Current Value'] = df[current_price_col] * df[quantity_col]
                
        if 'invested_value' not in column_mapping:
            # Try to calculate invested value
            avg_price_col = column_mapping.get('avg_price')
            quantity_col = column_mapping.get('quantity')
            if avg_price_col and quantity_col and avg_price_col in df.columns and quantity_col in df.columns:
                df['Invested Value'] = df[avg_price_col] * df[quantity_col]
        
        # Enhanced missing data handling
        df, data_changes = clean_and_handle_missing_data(df, column_mapping)
        
        # Simple success message for users
        st.success("✅ Portfolio loaded successfully!")
        
        # Display data cleaning summary if changes were made
        if data_changes:
            with st.expander("📋 Data Cleaning Summary - Click to see what was processed"):
                st.markdown("### Changes Made to Your Portfolio Data")
                for change_type, changes in data_changes.items():
                    if changes:
                        st.markdown(f"**{change_type}:**")
                        for change in changes:
                            st.write(f"- {change}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        st.info("💡 **Tip**: Make sure your CSV has columns for stock names, quantities, and prices/values")
        return None

def get_value_column(df):
    """Find the appropriate value column from the dataframe - Universal approach for multiple brokers"""
    if df.empty:
        return None
        
    # Priority order: Current/Market Value > Invested Value > Buy Value > others
    possible_columns = [
        # Format 1: Your current CSV format
        'Invested Value',  # Our calculated column gets priority
        'Current Value',   # Another calculated column
        'Closing value', 'Closing Value', 'closing value', 'CLOSING VALUE',
        'Buy value', 'Buy Value', 'buy value', 'BUY VALUE',
        # Format 2: Advanced broker format  
        'Market Value as of last trading day', 'Overall Gain/Loss',
        # Format 3: Trading format
        'Sell Value',
        # Common variations
        'Market Value', 'Market value', 'market value', 'MARKET VALUE',
        'Investment Value', 'Cur. val', 'Investment', 'Holdings Value', 
        'Total Value', 'Present Value', 'Value', 'value', 'VALUE',
        'Amount', 'Current Market Value'
    ]
    
    for col in possible_columns:
        if col in df.columns and not df[col].isna().all():
            # Check if it's actually numeric
            try:
                pd.to_numeric(df[col], errors='coerce')
                return col
            except:
                continue
    
    # If no perfect match, look for any numeric column with value-related keywords
    for col in df.columns:
        col_lower = col.lower()
        if (('val' in col_lower or 'investment' in col_lower or 'amount' in col_lower or 'price' in col_lower) 
            and pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all()):
            return col
            
    return None

def get_stock_name_column(df):
    """Find the appropriate stock name column - Universal approach for multiple brokers"""
    if df.empty:
        return None
        
    # Priority order based on multiple broker formats
    possible_columns = [
        # Format 1: Your current CSV format
        'Stock Name', 'stock name', 'Stock name', 'STOCK NAME',
        # Format 2: Advanced broker format
        'Scrip/Contract', 'Company Name', 'Scrip', 'Contract',
        # Format 3: Trading format  
        'Symbol', 'symbol', 'SYMBOL',
        # Common variations
        'Security Name', 'security name', 'Security name', 'SECURITY NAME',
        'Name', 'name', 'NAME', 'ISIN',
        'Name of Security', 'Script Name', 'CompanyName', 'StockName', 
        'SecurityName', 'Instrument', 'Ticker'
    ]
    
    for col in possible_columns:
        if col in df.columns and not df[col].isna().all():
            # Verify it contains actual stock names, not just numbers or empty values
            sample_values = df[col].dropna().head(5).astype(str)
            if any(len(val) > 1 and not val.replace('.', '').isdigit() for val in sample_values):
                return col
    
    # If no perfect match, use the first non-numeric column that looks like names
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
            # Check if it contains stock-like names (not dates or other metadata)
            sample_values = df[col].dropna().head(3).astype(str)
            if any(len(val) > 2 and not val.isdigit() and 'date' not in val.lower() for val in sample_values):
                return col
            
    return None

def calculate_enhanced_value(row, df, col_map, date_price_columns):
    """
    Calculate current value for a holding using multiple methods.
    Mirrors the logic from esg.py for consistency.
    """
    from XFIN.data_utils import find_column_case_insensitive, safe_float_conversion
    
    # Method 1: Check existing "Current Value" column (skip if zero)
    current_value_col = find_column_case_insensitive(
        col_map, 
        ['current value', 'market value']
    )
    if current_value_col:
        val = safe_float_conversion(row.get(current_value_col, 0))
        if val > 0:
            return val
    
    # Method 2: Qty × Price
    qty_col = find_column_case_insensitive(
        col_map, 
        ['qty', 'quantity', 'portfolio holdings', 'shares', 'units']
    )
    price_col = find_column_case_insensitive(
        col_map, 
        ['ltp', 'cmp', 'current price', 'price']
    )
    
    if qty_col and price_col:
        qty = safe_float_conversion(row.get(qty_col, 0))
        price = safe_float_conversion(row.get(price_col, 0))
        if qty > 0 and price > 0:
            return qty * price
    
    # Method 3: Date columns
    if date_price_columns and qty_col:
        qty = safe_float_conversion(row.get(qty_col, 0))
        for date_col in date_price_columns:
            date_price = safe_float_conversion(row.get(date_col, 0))
            if date_price > 0 and qty > 0:
                return qty * date_price
    
    # Method 4: Invested + P&L
    invested_col = find_column_case_insensitive(
        col_map, 
        ['invested value', 'invested', 'cost']
    )
    pnl_col = find_column_case_insensitive(
        col_map, 
        ['p&l', 'pnl', 'profit/loss', 'unrealized profit/loss', 
         'unrealized profit / loss', 'unrealised profit/loss']
    )
    
    if invested_col and pnl_col:
        invested = safe_float_conversion(row.get(invested_col, 0))
        pnl = safe_float_conversion(row.get(pnl_col, 0))
        if invested > 0:
            return invested + pnl
    
    # Method 5: Invested × (1 + Change%)
    change_col = find_column_case_insensitive(
        col_map, 
        ['% chg', '% change', 'change%']
    )
    
    if invested_col and change_col:
        invested = safe_float_conversion(row.get(invested_col, 0))
        change_pct = safe_float_conversion(row.get(change_col, 0))
        if invested > 0:
            if abs(change_pct) > 1:
                change_pct = change_pct / 100.0
            return invested * (1 + change_pct)
    
    # Method 6: Qty × Avg Cost
    avg_cost_col = find_column_case_insensitive(
        col_map, 
        ['avg cost', 'average cost', 'average cost value']
    )
    
    if qty_col and avg_cost_col:
        qty = safe_float_conversion(row.get(qty_col, 0))
        avg_cost = safe_float_conversion(row.get(avg_cost_col, 0))
        if qty > 0 and avg_cost > 0:
            return qty * avg_cost
    
    # Method 7: Invested fallback
    if invested_col:
        invested = safe_float_conversion(row.get(invested_col, 0))
        if invested > 0:
            return invested
    
    return 0.0

def create_stress_dashboard():
    # Streamlit page config
    st.set_page_config(page_title="XFIN Portfolio Stress Testing", page_icon="📈", layout="wide")
    
    st.title("📈 XFIN Portfolio Stress Testing")
    st.markdown("**AI-Powered Portfolio Analysis for Real-World Scenarios**")
    
    # Initialize ESG engine early (before any usage)
    try:
        esg_engine = ESGScoringEngine()
        esg_plot_gen = ESGPlotGenerator()
        esg_available = True
    except Exception as e:
        esg_available = False
        esg_engine = None
        esg_plot_gen = None
    
    # Sidebar: Portfolio upload and display
    st.sidebar.header("📂 Portfolio Configuration")
    portfolio_file = st.sidebar.file_uploader("Upload Portfolio CSV", type=['csv'])
    
    # LLM Configuration section
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AI Analysis Settings")
    
    # API Key input
    st.sidebar.markdown("### 🔑 AI Configuration")
    api_key = st.sidebar.text_input(
        "OpenRouter API Key (Optional)",
        type="password",
        help="Enter your OpenRouter API key for enhanced AI explanations. Leave blank to use environment variable."
    )
    
    if not api_key:
        st.sidebar.info("💡 **Get a free API key**: Visit [OpenRouter.ai](https://openrouter.ai/) to enable AI-powered recommendations")
    
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
    
    # Fast Mode toggle for backend performance (not for charts)
    fast_mode = st.sidebar.checkbox(
        "⚡ Fast Mode (Skip Market Data)",
        value=True,
        help="Skip live market data fetching for faster backend analysis. Charts will still be generated."
    )
    
    if use_llm:
        st.sidebar.success("✅ AI recommendations enabled")
    else:
        st.sidebar.info("ℹ️ Using basic analysis only")
        
    if fast_mode:
        st.sidebar.success("⚡ Fast mode: Quick backend analysis")
        st.sidebar.info("📊 All charts generated + fast core analysis")
    else:
        st.sidebar.warning("🐌 Full mode: With live market data")
        st.sidebar.info("📈 All charts + detailed market analysis")
    
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
        # Import enhanced value calculation utilities from consolidated data_utils
        from XFIN.data_utils import (
            detect_date_price_columns,
            find_column_case_insensitive, 
            safe_float_conversion
        )
        
        # Create column mapping
        col_map = {col.lower(): col for col in portfolio_data.columns}
        date_price_columns = detect_date_price_columns(portfolio_data)
        
        # Calculate enhanced values for each holding
        enhanced_values = []
        for idx, row in portfolio_data.iterrows():
            value = calculate_enhanced_value(row, portfolio_data, col_map, date_price_columns)
            enhanced_values.append(value)
        
        # Add enhanced values to dataframe
        portfolio_data['_enhanced_value'] = enhanced_values
        
        # Find the appropriate columns
        value_col = '_enhanced_value'  # Use our calculated values
        name_col = get_stock_name_column(portfolio_data)
        
        if value_col and name_col:
            # Calculate total portfolio value using enhanced calculation
            total_value = portfolio_data[value_col].sum()
            
            st.sidebar.markdown("**Your Portfolio Summary:**")
            st.sidebar.write(f"**Total Portfolio Value:** ₹{total_value:,.0f}")
            st.sidebar.write(f"**Value Source:** Enhanced Calculation")
            st.sidebar.write(f"**Number of Holdings:** {len(portfolio_data)}")
            
            # Show investment vs current value using enhanced calculation
            invested_col = find_column_case_insensitive(
                col_map, 
                ['invested value', 'invested', 'cost']
            )
            
            if invested_col:
                # Ensure numeric conversion (some broker CSVs keep currency symbols/commas)
                total_invested = portfolio_data[invested_col].apply(lambda x: safe_float_conversion(x)).sum()
                total_current = total_value  # This is our enhanced calculation
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
            
            # Show portfolio composition chart in sidebar
            st.sidebar.markdown("### 📊 Portfolio Overview")
            try:
                temp_plot_gen = StressPlotGenerator()
                fig_composition = temp_plot_gen.create_portfolio_composition_pie(portfolio_data, "Your Portfolio")
                st.sidebar.pyplot(fig_composition)
                plt.close(fig_composition)
            except Exception as e:
                st.sidebar.info("💡 Chart temporarily unavailable")
            
            # ESG Analysis will run when "Analyze My Portfolio" is clicked
            # Display placeholder in sidebar if ESG was previously analyzed
            if esg_available and 'esg_result' in st.session_state:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🌱 ESG Score")
                esg_result = st.session_state['esg_result']
                
                if esg_result and 'portfolio_esg_scores' in esg_result:
                    overall_score = esg_result['portfolio_esg_scores']['overall']
                    star_rating = esg_result['star_rating_text']
                    rating_label = esg_result['rating_label']
                    coverage = esg_result.get('coverage_percentage', 0)
                    risk_mult = esg_result.get('risk_multiplier', 1.0)
                    
                    # Display ESG metrics
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        st.metric("ESG Score", f"{overall_score:.0f}/100")
                    with col2:
                        st.metric("Rating", star_rating)
                    
                    # Risk multiplier indicator
                    mult_color = "🟢" if risk_mult < 1.0 else "🟡" if risk_mult == 1.0 else "🔴"
                    st.sidebar.write(f"**{rating_label}** | Risk: {mult_color} {risk_mult:.2f}x")
                    st.sidebar.caption(f"📊 Data Coverage: {coverage:.0f}%")
            elif esg_available:
                st.sidebar.markdown("---")
                st.sidebar.info("🌱 ESG analysis will run when you click 'Analyze My Portfolio'")
                            
        else:
            st.sidebar.warning("⚠️ Required columns not found. Please check your CSV format.")
    
    # Initialize modules
    model = type("MockModel", (), {"predict_stress_impact": lambda self, d, f: None})()
    
    # Initialize StressTestingEngine - use defaults first, then set API key
    try:
        explainer = StressTestingEngine()
        if use_llm and api_key:
            explainer.api_key = api_key  # Set API key after initialization
        st.success("✅ StressTestingEngine initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing StressTestingEngine: {e}")
        st.info("Using mock explainer as fallback...")
        # Fallback: create a minimal mock explainer
        explainer = type("MockExplainer", (), {
            "scenario_generator": type("MockGen", (), {
                "list_scenarios": lambda self: ["market_correction", "recession_scenario", "inflation_spike", "tech_sector_crash", "us_bond_yields_impact"],
                "get_scenario": lambda self, x: {
                    "name": x.replace("_", " ").title(), 
                    "description": f"Mock description for {x.replace('_', ' ')}", 
                    "factors": {}
                }
            })(),
            "portfolio_analyzer": type("MockAnalyzer", (), {
                "analyze_stress_impact": lambda self, data, scenario: {
                    "total_impact": -15.5,
                    "impact_percentage": -15.5,
                    "recovery_months": 12,
                    "risk_level": "Medium"
                }
            })(),
            "explain_stress_impact": lambda self, data, scenario: {
                "explanation": f"Mock explanation for {scenario} scenario with portfolio impact analysis.",
                "risk_level": "Medium",
                "recommendations": ["Diversify portfolio", "Consider hedging strategies"]
            },
            "api_key": None
        })()
    
    plot_gen = StressPlotGenerator()
    
    # Update explainer with user's API key if provided
    if api_key and use_llm:
        explainer.api_key = api_key
    
    # Scenario selection
    scenarios = explainer.scenario_generator.list_scenarios()
    scenario_names = {}
    for s in scenarios:
        try:
            scenario_info = explainer.scenario_generator.get_scenario(s)
            scenario_names[s] = scenario_info.get('name', s.replace('_', ' ').title())
        except (KeyError, AttributeError):
            scenario_names[s] = s.replace('_', ' ').title()
    
    st.subheader("🎯 Choose Your Scenario")
    selected_scenario = st.selectbox("Select Scenario:", scenarios, format_func=lambda x: scenario_names[x])
    
    # Show scenario description
    scenario_info = explainer.scenario_generator.get_scenario(selected_scenario)
    scenario_name = scenario_info.get('name', selected_scenario.replace('_', ' ').title())
    scenario_desc = scenario_info.get('description', 'Stress testing scenario')
    st.info(f"**{scenario_name}**: {scenario_desc}")
    
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
    
    # Add debug section
    with st.expander("🔧 Debug & Test", expanded=False):
        st.write("**Debug Information:**")
        if portfolio_data is not None:
            st.write(f"Portfolio shape: {portfolio_data.shape}")
            st.write(f"Portfolio columns: {list(portfolio_data.columns)}")
            
            # Test analysis button
            if st.button("🧪 Test Analysis (Debug Mode)", key="debug_test"):
                with st.spinner("Testing analysis..."):
                    try:
                        test_result = explainer.explain_stress_impact(portfolio_data, selected_scenario)
                        st.success("✅ Analysis test successful!")
                        st.json(test_result)
                    except Exception as e:
                        st.error(f"❌ Analysis test failed: {e}")
                        st.write("Error details:", str(e))
        else:
            st.info("Upload a portfolio first to enable debug testing")
    
    # Run analysis
    if st.button("🚀 Analyze My Portfolio", type="primary"):
        if portfolio_data is None:
            st.error("Please upload a portfolio CSV first.")
            st.stop()
        
        # Run ESG analysis first (only in non-fast mode for Single Scenario)
        if analysis_type == "Single Scenario Impact" and not fast_mode and esg_available and esg_engine:
            with st.spinner("🌱 Analyzing ESG scores..."):
                try:
                    esg_result = esg_engine.score_portfolio(portfolio_data)
                    st.session_state['esg_result'] = esg_result
                except Exception as e:
                    st.warning(f"ESG analysis skipped: {e}")
                    st.session_state['esg_result'] = None
        
        if analysis_type == "Single Scenario Impact":
            # PROGRESSIVE LOADING: Charts first, then AI
            
            # STEP 1: Generate core analysis quickly
            with st.spinner("⚡ Analyzing portfolio... (generating charts)"):
                try:
                    result = explainer.explain_stress_impact(portfolio_data, selected_scenario)
                    
                    # Handle different result structures
                    impact = None
                    portfolio_analysis = None
                    
                    if result and isinstance(result, dict):
                        # Try different result structures
                        if 'stress_impact' in result:
                            impact = result['stress_impact']
                            portfolio_analysis = result.get('portfolio_analysis')
                        elif 'total_impact' in result:
                            # Direct impact format
                            impact = result
                        
                        # Debug info (always show for troubleshooting)
                        with st.expander("🔍 Analysis Details", expanded=False):
                            st.write("**Raw Result Structure:**")
                            st.json(result)
                            if impact:
                                st.write("**Impact Data:**")
                                st.json(impact)
                    
                    if not impact:
                        st.error("❌ Could not extract stress impact from analysis")
                        st.info("💡 This might be due to:")
                        st.info("- CSV format issues (check column names)")
                        st.info("- Missing required columns (Symbol, Quantity, Price)")
                        st.info("- Data processing errors")
                        if result:
                            st.write("**Available result keys:**", list(result.keys()) if isinstance(result, dict) else "Not a dictionary")
                        st.stop()
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please check your CSV format and try again.")
                    st.stop()
                
                st.subheader("📋 Your Portfolio Analysis Results")
                
                # Calculate values for display
                if portfolio_data is not None and value_col:
                    total_portfolio_value = portfolio_data[value_col].sum()
                    # Ensure impact_percentage is converted to float
                    impact_percentage = float(impact['impact_percentage'])
                    impact_amount = total_portfolio_value * abs(impact_percentage) / 100
                else:
                    total_portfolio_value = 0
                    impact_percentage = float(impact['impact_percentage'])
                    impact_amount = 0
                
                # STEP 2: Display results based on fast_mode
                if fast_mode:
                    # FAST MODE: Basic text analysis only (no charts)
                    st.success("⚡ Fast Analysis Complete!")
                    st.markdown("### ⚡ Quick Impact Assessment")
                    
                    # Show basic metrics without charts
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Impact", f"{impact_percentage:.1f}%", 
                                f"₹{-impact_amount:,.0f}",
                                delta_color="inverse")
                    with col2:
                        st.metric("Risk Level", impact.get('risk_level', 'Medium'))
                    with col3:
                        st.metric("Recovery Time", f"{impact.get('recovery_months', 12):.0f} months")
                    with col4:
                        st.metric("VaR (95%)", f"{impact.get('var_95', 0)*100:.1f}%")
                    
                    # Display dynamic calculation details (NEW)
                    if 'dynamic_calculation' in impact:
                        dynamic = impact['dynamic_calculation']
                        st.markdown("---")
                        st.markdown("### 🎯 Portfolio-Specific Impact Calculation")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Base Impact", f"{dynamic.get('base_impact', 0):.2f}%",
                                    help="Sector-weighted baseline impact before diversification adjustment")
                        with col2:
                            hhi = dynamic.get('hhi', 0)
                            div_factor = dynamic.get('diversification_factor', 0)
                            div_label = "Well Diversified" if div_factor > 0.7 else "Concentrated" if div_factor < 0.4 else "Moderate"
                            st.metric("Diversification", f"{div_factor:.2f}",
                                    help=f"HHI: {hhi:.0f} - {div_label} portfolio")
                        with col3:
                            penalty = dynamic.get('concentration_penalty', 0)
                            st.metric("Concentration Penalty", f"+{penalty*100:.1f}%",
                                    help="Additional risk from portfolio concentration")
                        
                        # Show sector breakdown
                        with st.expander("📊 Sector Composition & Impact Breakdown"):
                            sector_comp = dynamic.get('sector_composition', {})
                            if sector_comp:
                                st.write("**Your Portfolio by Sector:**")
                                for sector, weight in sorted(sector_comp.items(), key=lambda x: x[1], reverse=True):
                                    st.write(f"• {sector}: {weight:.1f}%")
                            
                            # Show the detailed explanation
                            details_text = dynamic.get('details', '')
                            if details_text:
                                st.markdown(details_text)
                    
                    # Basic portfolio info without charts
                    st.markdown("**📋 Basic Portfolio Information:**")
                    st.write(f"• **Portfolio Value**: ₹{total_portfolio_value:,.0f}")
                    st.write(f"• **Potential Loss**: ₹{impact_amount:,.0f}")
                    st.write(f"• **Risk Assessment**: {impact.get('risk_level', 'Medium')} risk scenario")
                    st.write(f"• **Recovery Expectation**: {impact.get('recovery_months', 12):.0f} months")
                    
                else:
                    # FULL MODE: Generate all charts + comprehensive analysis
                    st.success("✅ Analysis Complete! Charts generated successfully.")
                    st.markdown("### 📊 Portfolio Analysis & Risk Assessment")
                
                # Chart generation (only for non-fast mode)  
                if not fast_mode:
                    try:
                        # Create tabs for different chart types - these should be fast
                        if esg_available and 'esg_result' in st.session_state:
                            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Risk Assessment", "📊 Portfolio Analysis", "⏰ Recovery Timeline", "🔍 Risk Breakdown", "🌱 ESG Analysis"])
                        else:
                            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Assessment", "📊 Portfolio Analysis", "⏰ Recovery Timeline", "🔍 Risk Breakdown"])
                        
                        with tab1:
                            st.markdown("### Risk Gauge Assessment")
                            try:
                                fig_gauge = plot_gen.create_risk_gauge_chart(impact, f"{selected_scenario.replace('_', ' ').title()} Risk Assessment")
                                st.pyplot(fig_gauge)
                                plt.close(fig_gauge)
                            except Exception as e:
                                st.error(f"Could not create risk gauge: {e}")
                                st.info("Using basic risk display instead")
                                st.write(f"**Risk Level**: {impact.get('risk_level', 'Unknown')}")
                        
                        with tab2:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### Portfolio Composition")
                                try:
                                    fig_composition = plot_gen.create_portfolio_composition_pie(portfolio_data, "Current Portfolio")
                                    st.pyplot(fig_composition)
                                    plt.close(fig_composition)
                                except Exception as e:
                                    st.error(f"Could not create portfolio composition chart: {e}")
                                    # Show basic portfolio info instead
                                    if result.get('portfolio_analysis'):
                                        composition = result['portfolio_analysis'].get('composition', {})
                                        st.write("**Portfolio Composition:**")
                                        for sector, weight in composition.items():
                                            st.write(f"- {sector}: {weight*100:.1f}%")
                            
                            with col2:
                                st.markdown("### Sector Risk Analysis")
                                try:
                                    fig_sector = plot_gen.create_sector_exposure_vs_impact_scatter(
                                        portfolio_data, impact, "Sector Risk vs Exposure"
                                    )
                                    st.pyplot(fig_sector)
                                    plt.close(fig_sector)
                                except Exception as e:
                                    st.error(f"Could not create sector analysis chart: {e}")
                                    st.info("Check portfolio data format - ensure it has Symbol and sector information")
                        
                        with tab3:
                            st.markdown("### Recovery Timeline")
                            try:
                                fig_recovery = plot_gen.create_recovery_timeline_chart(impact, "Expected Recovery Path")
                                st.pyplot(fig_recovery)
                                plt.close(fig_recovery)
                            except Exception as e:
                                st.error(f"Could not create recovery timeline: {e}")
                                # Show basic recovery info
                                recovery_months = impact.get('recovery_months', 12)
                                st.write(f"**Expected Recovery Time**: {recovery_months:.0f} months")
                                st.write("Recovery assumes 12.5% annual market return")
                        
                        with tab4:
                            st.markdown("### Risk Impact Breakdown")
                            try:
                                fig_waterfall = plot_gen.create_risk_impact_waterfall(impact, portfolio_data, "Risk Factor Analysis")
                                st.pyplot(fig_waterfall)
                                plt.close(fig_waterfall)
                            except Exception as e:
                                st.error(f"Could not create risk breakdown chart: {e}")
                                # Show basic risk breakdown
                                st.write("**Risk Summary:**")
                                st.write(f"- Total Impact: {impact.get('impact_percentage', 0):.1f}%")
                                st.write(f"- Risk Level: {impact.get('risk_level', 'Unknown')}")
                                st.write(f"- VaR 95%: {impact.get('var_95', 0)*100:.1f}%")
                        
                        # ESG Analysis Tab (only if ESG is available)
                        if esg_available and 'esg_result' in st.session_state:
                            with tab5:
                                st.markdown("### 🌱 ESG Analysis")
                                esg_result = st.session_state['esg_result']
                                
                                # Check if ESG result is valid (not None)
                                if esg_result is None or not isinstance(esg_result, dict):
                                    st.warning("⚠️ ESG analysis data is not available. This might happen if:")
                                    st.markdown("""
                                    - Portfolio has no stocks with valid ESG data
                                    - All stocks have zero values (no holdings)
                                    - ESG analysis encountered an error during processing
                                    
                                    **To fix**: Ensure your portfolio CSV contains stocks with non-zero values.
                                    """)
                                    st.info("💡 **Tip**: Check the console output for detailed error messages about ESG data fetching.")
                                else:
                                    try:
                                        # Row 1: ESG Gauge and Breakdown
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("#### Overall ESG Score")
                                            fig_gauge = esg_plot_gen.create_esg_gauge(esg_result)
                                            st.pyplot(fig_gauge)
                                            plt.close(fig_gauge)
                                        
                                        with col2:
                                            st.markdown("#### E-S-G Breakdown")
                                            fig_breakdown = esg_plot_gen.create_esg_breakdown(esg_result)
                                            st.pyplot(fig_breakdown)
                                            plt.close(fig_breakdown)
                                        
                                        # Row 2: Sector Heatmap
                                        st.markdown("#### ESG by Sector")
                                        fig_heatmap = esg_plot_gen.create_sector_esg_heatmap(esg_result)
                                        st.pyplot(fig_heatmap)
                                        plt.close(fig_heatmap)
                                        
                                        # Row 3: Coverage and Star Distribution
                                        col3, col4 = st.columns(2)
                                        with col3:
                                            st.markdown("#### Data Coverage")
                                            fig_coverage = esg_plot_gen.create_coverage_donut(esg_result)
                                            st.pyplot(fig_coverage)
                                            plt.close(fig_coverage)
                                        
                                        with col4:
                                            st.markdown("#### Star Rating Distribution")
                                            fig_stars = esg_plot_gen.create_esg_star_distribution(esg_result)
                                            st.pyplot(fig_stars)
                                            plt.close(fig_stars)
                                        
                                        # Holdings Detail Table
                                        st.markdown("#### Holdings ESG Detail")
                                        holdings_table = esg_plot_gen.create_holdings_esg_table(esg_result)
                                        st.dataframe(holdings_table, width='stretch')
                                        
                                        # ESG Risk Impact
                                        st.markdown("#### ESG Risk Adjustment")
                                        risk_mult = esg_result.get('risk_multiplier', 1.0)
                                        overall_score = esg_result['portfolio_esg_scores']['overall']
                                        
                                        if risk_mult < 1.0:
                                            st.success(f"✅ **Strong ESG performance** (Score: {overall_score:.0f}/100) reduces stress risk by {(1-risk_mult)*100:.0f}%")
                                        elif risk_mult > 1.0:
                                            st.warning(f"⚠️ **Weak ESG performance** (Score: {overall_score:.0f}/100) increases stress risk by {(risk_mult-1)*100:.0f}%")
                                        else:
                                            st.info(f"ℹ️ **Average ESG performance** (Score: {overall_score:.0f}/100) has neutral risk impact")
                                        
                                        # Show ESG-adjusted stress impact
                                        if impact:
                                            base_impact = impact.get('impact_percentage', 0)
                                            esg_adjusted_impact = base_impact * risk_mult
                                            impact_diff = esg_adjusted_impact - base_impact
                                            
                                            st.markdown("**ESG-Adjusted Stress Impact:**")
                                            col_a, col_b, col_c = st.columns(3)
                                            with col_a:
                                                st.metric("Base Impact", f"{base_impact:.1f}%")
                                            with col_b:
                                                st.metric("ESG Multiplier", f"{risk_mult:.2f}x")
                                            with col_c:
                                                st.metric("Adjusted Impact", f"{esg_adjusted_impact:.1f}%", 
                                                         delta=f"{impact_diff:+.1f}%", delta_color="inverse")
                                        
                                        # SHAP Waterfall Chart - ML Explainability (NEW!)
                                        st.markdown("---")
                                        st.markdown("#### 🔍 ML Model Explainability - What Drives Your ESG Score?")
                                        
                                        # Check if SHAP analysis is available
                                        shap_available = (esg_result.get('portfolio_shap_analysis') is not None and 
                                                        esg_result['portfolio_shap_analysis'] is not None)
                                        
                                        if shap_available:
                                            st.info("💡 This chart shows which factors most influenced the ML model's ESG predictions for your portfolio. Green bars increase the score, red bars decrease it.")
                                            
                                            try:
                                                fig_shap = esg_plot_gen.create_portfolio_shap_waterfall(esg_result, top_n=15)
                                                st.pyplot(fig_shap)
                                                plt.close(fig_shap)
                                                
                                                # Show SHAP data sources
                                                shap_data = esg_result['portfolio_shap_analysis']
                                                if 'data_sources' in esg_result:
                                                    sources = esg_result['data_sources']
                                                    ml_count = sources.get('ml_predictions', 0)
                                                    api_count = sources.get('api_data', 0)
                                                    proxy_count = sources.get('sector_proxy', 0)
                                                    
                                                    st.caption(f"📊 Analysis based on: {ml_count} ML predictions, {api_count} API data, {proxy_count} sector proxies")
                                                
                                                # Optional: Show grouped feature contributions
                                                with st.expander("📈 View Feature Category Breakdown"):
                                                    fig_grouped = esg_plot_gen.create_shap_grouped_bars(esg_result)
                                                    st.pyplot(fig_grouped)
                                                    plt.close(fig_grouped)
                                            
                                            except Exception as e:
                                                st.error(f"Could not create SHAP visualization: {e}")
                                                st.caption("💡 SHAP analysis is only available when ML predictions are used for ESG scoring.")
                                        else:
                                            # Show why SHAP is not available
                                            st.info("💡 **ML Explainability Not Available for This Portfolio**")
                                            
                                            # Show data sources breakdown
                                            if 'data_sources' in esg_result:
                                                sources = esg_result['data_sources']
                                                ml_count = sources.get('ml_predictions', 0)
                                                api_count = sources.get('api_data', 0)
                                                proxy_count = sources.get('sector_proxy', 0)
                                                total_stocks = api_count + ml_count + proxy_count
                                                
                                                # Explain based on actual data composition and 3-tier fallback
                                                if ml_count == 0 and api_count > 0 and proxy_count == 0:
                                                    st.write("✅ All stocks in your portfolio have **direct ESG ratings** from data providers (Yahoo Finance, BRSR).")
                                                    st.write("**No ML predictions were needed** - all stocks had real ESG data available.")
                                                elif ml_count == 0 and proxy_count > 0:
                                                    st.write(f"**3-Tier Fallback System:** API Data → ML Prediction → Sector Proxy")
                                                    st.write(f"For your portfolio: {api_count} stocks have real ESG data, but {proxy_count} stocks fell back to **sector proxy** (tier 3) because ML predictions failed (tier 2 - requires market data from yfinance).")
                                                    st.write("💡 ML predictions require valid ticker symbols and market data. Stocks with incorrect tickers or delisted stocks skip ML and go straight to sector proxy.")
                                                else:
                                                    st.write(f"**3-Tier Fallback System Active:** API Data → ML Prediction → Sector Proxy")
                                                
                                                st.write("**Your portfolio's ESG data sources:**")
                                                col_x, col_y, col_z = st.columns(3)
                                                with col_x:
                                                    st.metric("🌐 Tier 1: Real ESG", api_count, 
                                                             help="Stocks with actual ESG ratings from Yahoo Finance, BRSR, etc. (Highest quality)",
                                                             delta=f"{(api_count/total_stocks*100):.1f}% coverage" if total_stocks > 0 else None)
                                                with col_y:
                                                    st.metric("🤖 Tier 2: ML Predictions", ml_count, 
                                                             help="Stocks where ML model predicted ESG from market features (SHAP explainable)",
                                                             delta="SHAP available ✓" if ml_count > 0 else "Not used")
                                                with col_z:
                                                    st.metric("📊 Tier 3: Sector Proxy", proxy_count, 
                                                             help="Fallback: Stocks using sector average when tier 1 & 2 unavailable",
                                                             delta=f"{(proxy_count/total_stocks*100):.1f}% estimated" if total_stocks > 0 else None)
                                                
                                                st.markdown("---")
                                                if ml_count == 0:
                                                    st.write("**💡 Why no ML predictions?**")
                                                    st.write("ML predictions (tier 2) require:")
                                                    st.write("• Valid NSE ticker symbol (e.g., RELIANCE.NS)")
                                                    st.write("• Available market data on Yahoo Finance")
                                                    st.write("• Recent price history for volatility/returns calculation")
                                                    st.caption("🔍 If stocks have real ESG data (tier 1), ML is skipped. If ML can't fetch features, system falls back to sector proxy (tier 3).")
                                            else:
                                                # Fallback if data_sources not tracked
                                                st.warning("⚠️ Data source tracking not available for this analysis.")
                                                st.write("**XFIN uses 3-tier fallback:** API Data → ML Prediction → Sector Proxy")
                                                st.caption("💡 ML predictions are used when stocks lack public ESG ratings AND have valid market data available.")
                                    
                                    except Exception as e:
                                        st.error(f"Could not create ESG visualizations: {e}")
                                        st.write(f"Error details: {str(e)}")
                                        import traceback
                                        st.code(traceback.format_exc())
                
                    except Exception as e:
                        st.warning(f"Enhanced visualizations unavailable: {e}")
                        # Fallback to old chart
                        try:
                            fig = plot_gen.create_stress_impact_plot(result)
                            st.pyplot(fig)
                            plt.close(fig)
                        except:
                            st.error("All visualizations failed to load")
                
                # ============================================================================
                # STEP 3: AI-Powered Recommendations (separate phase, may take 3-5 seconds)
                # ============================================================================
                
                st.markdown("---")  # Visual separator
                st.markdown("### 💡 What Should You Do?")
                
                # Show analysis summary (for both modes)
                if not fast_mode:
                    # Full mode gets detailed summary with horizontal metrics
                    st.markdown("**📊 Comprehensive Analysis Summary:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Impact", f"{impact['impact_percentage']:.1f}%", 
                                f"₹{-impact_amount:,.0f}",
                                delta_color="inverse")
                    with col2:
                        st.metric("Risk Level", impact['risk_level'])
                    with col3:
                        st.metric("Recovery Time", f"{impact.get('recovery_months', 12):.0f} months")
                    with col4:
                        st.metric("VaR (95%)", f"{impact.get('var_95', 0)*100:.1f}%")
                # Fast mode already showed basic metrics above
                
                if use_llm:
                    # STEP 4: Generate AI recommendations (this may take 3-5 seconds)
                    st.info("🤖 Generating personalized AI recommendations... (this may take a few seconds)")
                    with st.spinner("🤖 AI is analyzing your portfolio..."):
                        try:
                            # Get ESG result if available
                            esg_data = st.session_state.get('esg_result') if esg_available else None
                            
                            # Generate recommendations with ESG integration
                            recs = explainer.generate_recommendations(portfolio_data, result, fast_mode=fast_mode, esg_result=esg_data)
                            
                            # Check if the response contains an error message
                            if recs.startswith("❌"):
                                st.warning(recs)
                                st.info("💡 **Tip**: You can get a free API key from [OpenRouter.ai](https://openrouter.ai/) to enable AI-powered recommendations.")
                            else:
                                st.success("✅ AI recommendations generated!")
                                st.markdown("### 🤖 AI-Powered Recommendations")
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
                    # If AI is disabled, show encouraging message
                    st.info("💡 Enable 'AI-Powered Recommendations' in the sidebar for personalized advice!")
                    st.markdown("""
                    **📋 Basic Recommendations:**
                    - Review your risk tolerance based on the analysis above
                    - Consider diversification if risk level is High/Extreme  
                    - Maintain 6-12 months emergency fund
                    - Consult a financial advisor for personalized advice
                    """)
        
        else:
            # Comparison - fix for when compare_keys is empty
            if not compare_keys:
                st.warning("Please select at least one scenario to compare.")
            else:
                with st.spinner("Comparing scenarios..."):
                    comp_df = explainer.compare_scenarios(portfolio_data, compare_keys)
                    
                    st.subheader("📊 Multi-Scenario Comparison")
                    st.dataframe(comp_df.style.format({'impact_percentage': '{:.1f}%'}), width='stretch')
                    
                    try:
                        # Use the new radar chart for multi-scenario comparison
                        fig_radar = plot_gen.create_multi_scenario_radar(comp_df, "Multi-Scenario Risk Comparison")
                        st.pyplot(fig_radar)
                        plt.close(fig_radar)
                    except Exception as e:
                        st.warning(f"Enhanced comparison chart unavailable: {e}")
                        # Fallback to old comparison chart
                        try:
                            fig2 = plot_gen.create_scenario_comparison(comp_df)
                            st.pyplot(fig2)
                            plt.close(fig2)
                        except:
                            st.error("All comparison visualizations failed")
                    
                    # AI Summary for comparison
                    if use_llm and len(compare_keys) > 1:
                        st.markdown("---")
                        st.markdown("### 🤖 AI Multi-Scenario Analysis")
                        with st.spinner("Generating comparative insights..."):
                            try:
                                # Create a detailed summary for multiple scenarios
                                portfolio_value = portfolio_data[value_col].sum()
                                scenario_names_list = [scenario_names[k] for k in compare_keys]
                                
                                # Find worst, best, and median scenarios
                                worst_scenario = comp_df.loc[comp_df['impact_percentage'].idxmin()]
                                best_scenario = comp_df.loc[comp_df['impact_percentage'].idxmax()]
                                median_impact = comp_df['impact_percentage'].median()
                                
                                # Calculate financial impacts
                                worst_loss = portfolio_value * abs(worst_scenario['impact_percentage']) / 100
                                best_loss = portfolio_value * abs(best_scenario['impact_percentage']) / 100
                                impact_spread = abs(worst_scenario['impact_percentage'] - best_scenario['impact_percentage'])
                                
                                # Direct LLM call for cleaner output
                                try:
                                    from XFIN.utils import _get_openrouter_key
                                except ImportError:
                                    from utils import _get_openrouter_key
                                    
                                import requests
                                
                                comparison_prompt = f"""You are analyzing a portfolio stress test comparing {len(compare_keys)} different scenarios.

PORTFOLIO OVERVIEW:
• Total Value: ₹{portfolio_value:,.0f}
• Scenarios Analyzed: {', '.join(scenario_names_list)}

STRESS TEST RESULTS (sorted by severity):
{comp_df.sort_values('impact_percentage').to_string(index=False)}

KEY FINDINGS:
• Worst Case: {worst_scenario['scenario']} → {worst_scenario['impact_percentage']:.1f}% loss (₹{worst_loss:,.0f})
• Best Case: {best_scenario['scenario']} → {best_scenario['impact_percentage']:.1f}% loss (₹{best_loss:,.0f})
• Impact Range: {impact_spread:.1f}% difference between scenarios
• Median Impact: {median_impact:.1f}%
• Longest Recovery: {comp_df['recovery_months'].max():.0f} months

Provide a focused comparative analysis with these sections (use markdown headers):

## 🎯 SCENARIO IMPACT RANKING
Rank all {len(compare_keys)} scenarios from most to least severe. For EACH scenario, explain:
- Why this scenario affects the portfolio differently
- Specific vulnerabilities it exposes
- Financial impact in rupees

## ⚠️ BIGGEST THREAT
Identify THE ONE most dangerous scenario and explain:
- Why it's the worst case for this portfolio
- Which sectors/assets would be hit hardest
- When this scenario might occur

## 💪 PORTFOLIO RESILIENCE
Based on {impact_spread:.1f}% impact range:
- Is diversification effective?
- Which scenarios expose similar weaknesses?
- What recovery patterns emerge?

## 🔄 COMPARATIVE INSIGHTS
- Why is there a {impact_spread:.1f}% difference between best and worst?
- Which scenarios might occur together?
- Which is most likely in current Indian market conditions?

## 🛡️ STRATEGIC RECOMMENDATIONS
Provide THREE specific actions:
1. Primary hedge against worst scenario
2. Portfolio adjustment to reduce impact spread
3. Key monitoring indicator

## ⚡ PRIORITY ACTION
THE SINGLE MOST IMPORTANT action to take now.

Keep analysis comparative and specific to Indian markets. DO NOT add introductory text before the first header."""

                                # Use user-provided API key if available, otherwise get from environment
                                final_api_key = api_key if api_key else _get_openrouter_key()
                                
                                if not final_api_key:
                                    st.error("❌ No API key configured!")
                                    st.info("""
                                    **To enable AI analysis, you need an OpenRouter API key:**
                                    
                                    1. Get a free key from [OpenRouter.ai](https://openrouter.ai/)
                                    2. Either:
                                       - Enter it in the sidebar under "OpenRouter API Key"
                                       - Or create a `.env` file with: `OPENROUTER_API_KEY=your_key_here`
                                    3. Refresh the page
                                    """)
                                else:
                                    headers = {
                                        "Authorization": f"Bearer {final_api_key}",
                                        "Content-Type": "application/json"
                                    }
                                    data = {
                                        "model": "x-ai/grok-code-fast-1",
                                        "messages": [{"role": "user", "content": comparison_prompt}],
                                        "max_tokens": 3500,  # Increased for complete analysis
                                        "temperature": 0.7
                                    }
                                    
                                    response = requests.post(
                                        "https://openrouter.ai/api/v1/chat/completions",
                                        headers=headers,
                                        json=data,
                                        timeout=20  # Increased timeout for complete responses
                                    )
                                    
                                    response.raise_for_status()  # Raise exception for bad status codes
                                    
                                    if response.status_code == 200:
                                        comparison_analysis = response.json()['choices'][0]['message']['content']
                                        st.success("✅ Multi-scenario analysis complete!")
                                        st.markdown(comparison_analysis)
                                    else:
                                        st.warning(f"AI analysis returned status {response.status_code} - showing comparison table above")
                                    
                            except requests.exceptions.RequestException as req_err:
                                st.error(f"AI API connection failed: {str(req_err)}")
                                st.info("💡 Check your internet connection and API key configuration.")
                            except KeyError as key_err:
                                st.error(f"AI response format error: {str(key_err)}")
                                st.info("💡 The API response was unexpected. Please try again.")
                            except Exception as e:
                                st.error(f"AI comparison analysis failed: {str(e)}")
                                st.info("The scenario comparison data above shows the key metrics for manual analysis.")
    
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