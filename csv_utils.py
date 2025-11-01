"""
CSV Parsing Utilities for Portfolio Data
Handles broker-specific formats with date columns, prev close, and missing identifiers
"""

import re
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Optional


# Regex for detecting date-like column headers
DATE_COL_REGEX = re.compile(
    r'^\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\w{3,9}\s+\d{1,2},?\s*\d{4})\s*$'
)


def detect_date_price_columns(df: pd.DataFrame) -> List[str]:
    """
    Return list of column names that look like date headers (YYYY-MM-DD, DD-MM-YYYY, 'Oct 25, 2025')
    or columns named 'Prev Close', 'Previous Close', etc.
    Sorted by date if possible (old->new). If parsing fails, keep order discovered.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The portfolio DataFrame
        
    Returns:
    --------
    list of str : Column names that contain price data (date columns or prev close)
    """
    date_cols = []
    
    for col in df.columns:
        s = str(col).strip()
        low = s.lower()
        
        # Check if it's a date-like column header
        if DATE_COL_REGEX.match(s):
            date_cols.append((col, s))
            continue
        
        # Check for 'prev close', 'previous close', 'prev price', etc.
        if 'prev' in low and ('close' in low or 'price' in low):
            date_cols.append((col, s))
            continue
        
        if 'previous' in low and ('close' in low or 'price' in low):
            date_cols.append((col, s))
            continue
    
    if not date_cols:
        return []
    
    # Try to parse and sort by date
    def parse_col_date(s):
        """Try multiple date formats"""
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
                    "%b %d, %Y", "%d %b %Y", "%d-%b-%Y", "%d %B %Y"]:
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        return None
    
    parsed = []
    unparsed = []
    
    for col, s in date_cols:
        p = parse_col_date(s)
        if p:
            parsed.append((p, col))
        else:
            unparsed.append(col)
    
    # Sort parsed by date (oldest first, so we can reverse later for newest)
    parsed_sorted = [c for _, c in sorted(parsed, key=lambda x: x[0])]
    
    # Return newest first for practical use
    return list(reversed(parsed_sorted)) + unparsed


def find_column_case_insensitive(column_map_lower: Dict[str, str], patterns: List[str]) -> Optional[str]:
    """
    Find column in DataFrame using case-insensitive pattern matching
    
    Parameters:
    -----------
    column_map_lower : dict
        Mapping of lowercase column names to original column names
    patterns : list of str
        List of patterns to search for (case-insensitive)
        
    Returns:
    --------
    str or None : Original column name if found, else None
    """
    for pattern in patterns:
        pattern_lower = pattern.lower()
        if pattern_lower in column_map_lower:
            return column_map_lower[pattern_lower]
        
        # Partial match fallback
        for col_lower, col_original in column_map_lower.items():
            if pattern_lower in col_lower or col_lower in pattern_lower:
                return col_original
    
    return None


def extract_holding_from_row(row: pd.Series, df: pd.DataFrame, column_map_lower: Dict[str, str]) -> Optional[Dict]:
    """
    Extract holding information from a CSV row, salvaging rows where Company is missing
    by using Symbol or ISIN as fallback
    
    Parameters:
    -----------
    row : pd.Series
        Single row from the DataFrame
    df : pd.DataFrame
        Full DataFrame (for context)
    column_map_lower : dict
        Lowercase column name mapping
        
    Returns:
    --------
    dict or None : Holding information with salvage metadata, or None if unsalvageable
    """
    # Try to find stock name
    stock_name_col = find_column_case_insensitive(
        column_map_lower, 
        ['company', 'stock name', 'security name', 'name', 'scrip', 'security']
    )
    
    # Try to find symbol
    symbol_col = find_column_case_insensitive(
        column_map_lower,
        ['symbol', 'ticker', 'scrip code', 'nse symbol', 'bse symbol', 'trading symbol']
    )
    
    # Try to find ISIN
    isin_col = find_column_case_insensitive(
        column_map_lower,
        ['isin', 'isin code', 'isin number']
    )
    
    # Extract values
    stock_name = row[stock_name_col] if stock_name_col and pd.notna(row[stock_name_col]) else None
    symbol = row[symbol_col] if symbol_col and pd.notna(row[symbol_col]) else None
    isin = row[isin_col] if isin_col and pd.notna(row[isin_col]) else None
    
    # Clean stock name
    if stock_name:
        stock_name = str(stock_name).strip()
        # Skip if empty after stripping
        if not stock_name or stock_name.lower() in ['nan', 'none', '', 'null', 'na']:
            stock_name = None
    
    # Salvage logic
    salvage_reason = None
    
    # Salvage by symbol
    if not stock_name and symbol:
        stock_name = str(symbol).strip()
        salvage_reason = 'used_symbol'
        print(f"   📍 Salvaged row using Symbol: {stock_name}")
    
    # Salvage by ISIN
    if not stock_name and isin:
        stock_name = str(isin).strip()
        salvage_reason = 'used_isin'
        print(f"   📍 Salvaged row using ISIN: {stock_name}")
    
    # Last resort: check if row has meaningful numeric data
    if not stock_name:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        meaningful = False
        
        for c in numeric_cols:
            val = row.get(c)
            if pd.notna(val):
                try:
                    if float(val) != 0:
                        meaningful = True
                        break
                except (ValueError, TypeError):
                    pass
        
        if meaningful:
            # Generate a name from available data
            if symbol:
                stock_name = f"Unknown-{symbol}"
            elif isin:
                stock_name = f"Unknown-{isin[:8]}"
            else:
                stock_name = f"Unknown (salvaged row)"
            salvage_reason = 'salvaged_numeric'
            print(f"   ⚠️  Salvaged row with numeric data: {stock_name}")
    
    # If still no stock name, cannot salvage
    if not stock_name:
        return None
    
    return {
        'stock_name': stock_name,
        'symbol': symbol,
        'isin': isin,
        'salvage_reason': salvage_reason,
        'raw_row': row
    }


def safe_float_conversion(value) -> float:
    """
    Safely convert various formats to float
    Handles currency symbols, commas, quotes, etc.
    
    Parameters:
    -----------
    value : any
        Value to convert to float
        
    Returns:
    --------
    float : Converted value, or 0.0 if conversion fails
    """
    if pd.isna(value):
        return 0.0
    
    try:
        # Handle string values
        if isinstance(value, str):
            # Remove currency symbols, commas, spaces, quotes
            value = value.replace('₹', '').replace('$', '').replace('€', '').replace('£', '')
            value = value.replace(',', '').replace(' ', '').strip()
            value = value.replace('"', '').replace("'", '').strip()
            # Handle parentheses for negative numbers (accounting format)
            if '(' in value and ')' in value:
                value = '-' + value.replace('(', '').replace(')', '')
        
        # Convert to float
        result = float(value)
        return result
    
    except (ValueError, TypeError, AttributeError):
        return 0.0


def create_diagnostics_report(holdings: List[Dict], df_original: pd.DataFrame, 
                              skipped_rows: List[Tuple[int, str]]) -> Dict:
    """
    Create a diagnostics report for CSV upload
    
    Parameters:
    -----------
    holdings : list of dict
        Successfully processed holdings
    df_original : pd.DataFrame
        Original uploaded DataFrame
    skipped_rows : list of tuple
        List of (row_index, reason) for skipped rows
        
    Returns:
    --------
    dict : Diagnostics information
    """
    rows_read = len(df_original)
    rows_processed = len(holdings)
    rows_skipped = len(skipped_rows)
    rows_salvaged = sum(1 for h in holdings if h.get('salvage_reason'))
    
    # Group by salvage reason
    salvage_breakdown = {}
    for h in holdings:
        reason = h.get('salvage_reason', 'normal')
        if reason not in salvage_breakdown:
            salvage_breakdown[reason] = 0
        salvage_breakdown[reason] += 1
    
    # Group by value source
    value_source_breakdown = {}
    for h in holdings:
        source = h.get('value_source', 'unknown')
        if source not in value_source_breakdown:
            value_source_breakdown[source] = 0
        value_source_breakdown[source] += 1
    
    return {
        'rows_read': rows_read,
        'rows_processed': rows_processed,
        'rows_salvaged': rows_salvaged,
        'rows_skipped': rows_skipped,
        'salvage_breakdown': salvage_breakdown,
        'value_source_breakdown': value_source_breakdown,
        'skipped_details': skipped_rows,
        'salvaged_holdings': [h for h in holdings if h.get('salvage_reason')]
    }
