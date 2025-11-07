"""
XFIN Data Utilities
===================

Consolidated data manipulation utilities for portfolio analysis.
Combines CSV parsing, sector classification, and ticker mapping.

This module provides:
- CSV parsing with broker-specific format handling
- Sector classification using keyword matching and API lookup
- Indian stock ticker mapping (NSE/BSE to Yahoo Finance)
"""

import re
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Optional


# ============================================================================
# CSV PARSING UTILITIES
# ============================================================================

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


# ============================================================================
# SECTOR CLASSIFICATION UTILITIES
# ============================================================================

def infer_sector_from_name(security_name: str) -> str:
    """
    Infer sector from security name using comprehensive keyword matching
    
    Parameters:
    -----------
    security_name : str
        Company or security name
    
    Returns:
    --------
    str
        Sector name
    """
    name_upper = security_name.upper()
    
    # PRIORITY: Company-specific exact matches (overrides keyword matching)
    exact_company_mapping = {
        'TATA MOTORS': 'Automobiles',
        'TATAMOTORS': 'Automobiles',
        'TATA MOTOR': 'Automobiles',
        'TATA MOTORS PASS VEH': 'Automobiles',
        'TATA MOTORS LTD': 'Automobiles',
        'TATA CAPITAL': 'Financial Services',
        'TATA CAPITAL LIMITED': 'Financial Services',
        'TATAAML-TATSILV': 'Financial Services',  # Tata Asset Management
        'TATA AML': 'Financial Services',
        'TATA ASSET': 'Financial Services',
        'REC LIMITED': 'Power',  # Rural Electrification Corporation
        'REC LTD': 'Power',
        'POWER FIN CORP': 'Power',  # Power Finance Corporation
        'PFC': 'Power',
        'SUZLON ENERGY': 'Power',
        'SUZLON': 'Power',
        'GRAPHITE INDIA': 'Metals & Mining',
        'GRAPHITE': 'Metals & Mining',
        'RITES LIMITED': 'Infrastructure',  # Engineering consultancy
        'RITES LTD': 'Infrastructure',
        'GAIL': 'Oil & Gas',
        'GAIL INDIA': 'Oil & Gas',
        'COAL INDIA': 'Power',  # Coal production for power generation
        'COAL INDIA LTD': 'Power',
        'VEDANTA': 'Metals & Mining',
        'VEDANTA LIMITED': 'Metals & Mining',
        'NMDC': 'Metals & Mining',
    }
    
    # Check for exact company name match first
    for company_key, sector in exact_company_mapping.items():
        if company_key in name_upper:
            return sector
    
    # Comprehensive sector keywords with weighted scoring
    sector_patterns = {
        'Banking': {
            'high_priority': ['BANK OF', 'STATE BANK', 'SBI ', 'HDFC BANK', 'ICICI BANK', 'AXIS BANK', 'KOTAK BANK'],
            'medium_priority': ['BANK', 'BANKS', 'BANKING'],
            'low_priority': []
        },
        'Financial Services': {
            'high_priority': ['BAJAJ FIN', 'HOUSING FINANCE', 'BAJAJ HOUSING', 'DEPOSITORY', 'CDSL', 'NSDL', 'MUTUAL FUND', 'ASSET MANAGEMENT'],
            'medium_priority': ['FINANCIAL', 'FINANCE', 'CREDIT', 'INSURANCE', 'SECURITIES', 'INVESTMENT'],
            'low_priority': ['CAPITAL', 'FUND']  # Low priority as these are common in names
        },
        'IT Services': {
            'high_priority': ['TCS', 'TATA CONSULTANCY', 'INFOSYS', 'INFY', 'WIPRO', 'TECH MAHINDRA', 'HCL TECH'],
            'medium_priority': ['SOFTWARE', 'INFOTECH', 'TECHNOLOGY', 'IT SERVICES', 'DIGITAL', 'CYBER'],
            'low_priority': ['TECH', 'DATA', 'SYSTEM']
        },
        'Pharmaceuticals': {
            'high_priority': ['CIPLA', 'LUPIN', 'DR REDDY', 'SUN PHARMA', 'BIOCON', 'PHARMACEUTICAL'],
            'medium_priority': ['PHARMA', 'HEALTHCARE', 'MEDICAL', 'MEDICINE', 'DRUG'],
            'low_priority': ['HEALTH', 'BIO', 'LIFE']
        },
        'Oil & Gas': {
            'high_priority': ['INDIAN OIL', 'BPCL', 'HPCL', 'HINDUSTAN PETROLEUM', 'BHARAT PETROLEUM', 'ONGC', 'OIL AND NATURAL GAS'],
            'medium_priority': ['PETROLEUM', 'REFINERY', 'OIL CORP'],
            'low_priority': ['OIL', 'GAS']  # Can be ambiguous (e.g., Graphite)
        },
        'Power': {
            'high_priority': ['NTPC', 'POWERGRID', 'TATA POWER', 'ADANI POWER', 'JSW ENERGY', 'TORRENT POWER', 'NLC INDIA', 'NHPC', 'REC LIMITED', 'POWER FIN', 'SUZLON'],
            'medium_priority': ['POWER CORP', 'ELECTRICITY', 'ELECTRIC UTILITIES', 'RENEWABLE ENERGY', 'SOLAR ENERGY', 'WIND ENERGY'],
            'low_priority': ['POWER', 'ELECTRIC', 'ENERGY', 'RENEWABLE']
        },
        'Automobiles': {
            'high_priority': ['TATA MOTORS', 'MARUTI', 'MAHINDRA', 'BAJAJ AUTO', 'HERO MOTO', 'TVS MOTOR', 'EICHER MOTORS', 'ASHOK LEYLAND'],
            'medium_priority': ['AUTOMOBILE', 'AUTOMOTIVE', 'VEHICLES'],
            'low_priority': ['AUTO', 'MOTOR', 'CAR']
        },
        'FMCG': {
            'high_priority': ['ITC LTD', 'HUL', 'HINDUSTAN UNILEVER', 'BRITANNIA', 'NESTLE', 'DABUR', 'MARICO', 'GODREJ CONSUMER'],
            'medium_priority': ['CONSUMER GOODS', 'CONSUMER PRODUCTS', 'FOODS', 'BEVERAGE'],
            'low_priority': ['CONSUMER', 'FOOD', 'PRODUCTS']
        },
        'Telecom': {
            'high_priority': ['BHARTI AIRTEL', 'VODAFONE', 'IDEA', 'JIO', 'RELIANCE JIO'],
            'medium_priority': ['TELECOM', 'TELECOMMUNICATION', 'COMMUNICATION SERVICES'],
            'low_priority': ['MOBILE', 'NETWORK']
        },
        'Metals & Mining': {
            'high_priority': ['TATA STEEL', 'JSW STEEL', 'HINDALCO', 'VEDANTA', 'NALCO', 'NMDC', 'SAIL', 'JINDAL STEEL', 'GRAPHITE INDIA'],
            'medium_priority': ['STEEL CORP', 'ALUMINIUM', 'ALUMINUM', 'MINING CORP'],
            'low_priority': ['STEEL', 'METAL', 'MINING', 'COPPER', 'ZINC']
        },
        'Infrastructure': {
            'high_priority': ['L&T', 'LARSEN TOUBRO', 'LARSEN & TOUBRO', 'GMR INFRA', 'GVK', 'IRB INFRA', 'NCC LTD', 'HCC', 'RITES'],
            'medium_priority': ['INFRASTRUCTURE', 'CONSTRUCTION', 'ENGINEERING SERVICES', 'PROJECTS LTD'],
            'low_priority': ['INFRA', 'ENGINEERING', 'BUILDERS', 'DEVELOPERS']
        },
        'Cement': {
            'high_priority': ['ULTRATECH', 'ACC LTD', 'AMBUJA', 'SHREE CEMENT', 'RAMCO CEMENT', 'DALMIA CEMENT'],
            'medium_priority': ['CEMENT CORP', 'CEMENT LTD'],
            'low_priority': ['CEMENT']
        },
        'Real Estate': {
            'high_priority': ['DLF', 'OBEROI REALTY', 'GODREJ PROPERTIES', 'PRESTIGE ESTATES', 'BRIGADE', 'SOBHA'],
            'medium_priority': ['REAL ESTATE', 'PROPERTIES', 'REALTY', 'HOUSING DEVELOPMENT'],
            'low_priority': ['PROPERTY', 'HOUSING']
        },
        'Media & Entertainment': {
            'high_priority': ['ZEE ENTERTAINMENT', 'SUN TV', 'TV18', 'NETWORK18', 'PVR', 'INOX'],
            'medium_priority': ['MEDIA', 'ENTERTAINMENT', 'BROADCASTING'],
            'low_priority': ['FILM', 'CINEMA']
        }
    }
    
    # Weighted scoring: high=10, medium=3, low=1
    sector_scores = {}
    for sector, priority_keywords in sector_patterns.items():
        score = 0
        
        # High priority matches (specific company/product names)
        for keyword in priority_keywords.get('high_priority', []):
            if keyword in name_upper:
                score += 10
        
        # Medium priority matches (sector-specific terms)
        for keyword in priority_keywords.get('medium_priority', []):
            if keyword in name_upper:
                score += 3
        
        # Low priority matches (generic terms)
        for keyword in priority_keywords.get('low_priority', []):
            if keyword in name_upper:
                score += 1
        
        if score > 0:
            sector_scores[sector] = score
    
    # Return sector with highest score
    if sector_scores:
        best_sector = max(sector_scores.items(), key=lambda x: x[1])
        return best_sector[0]
    
    # Default fallback
    return 'Other'


def fetch_sector_from_api(ticker: str) -> Optional[str]:
    """
    Fetch sector from Yahoo Finance API
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    
    Returns:
    --------
    str or None
        Sector name if found
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return None
        
        # Get sector
        sector = info.get('sector')
        if sector:
            # Map Yahoo Finance sectors to our categories
            sector_mapping = {
                'Financial Services': 'Financial Services',
                'Banks': 'Banking',
                'Technology': 'IT Services',
                'Communication Services': 'Telecom',
                'Energy': 'Oil & Gas',
                'Basic Materials': 'Metals & Mining',
                'Utilities': 'Power',
                'Industrials': 'Infrastructure',
                'Consumer Cyclical': 'FMCG',
                'Consumer Defensive': 'FMCG',
                'Healthcare': 'Pharmaceuticals',
                'Real Estate': 'Real Estate'
            }
            
            return sector_mapping.get(sector, sector)
        
        # Try industry as fallback
        industry = info.get('industry')
        if industry:
            industry_lower = industry.lower()
            if 'bank' in industry_lower:
                return 'Banking'
            elif 'software' in industry_lower or 'technology' in industry_lower:
                return 'IT Services'
            elif 'oil' in industry_lower or 'gas' in industry_lower:
                return 'Oil & Gas'
            elif 'pharma' in industry_lower or 'drug' in industry_lower:
                return 'Pharmaceuticals'
            elif 'auto' in industry_lower:
                return 'Automobiles'
            elif 'telecom' in industry_lower:
                return 'Telecom'
            elif 'power' in industry_lower or 'electric' in industry_lower:
                return 'Power'
            elif 'steel' in industry_lower or 'metal' in industry_lower:
                return 'Metals & Mining'
        
        return None
        
    except Exception:
        return None


def get_sector(security_name: str, ticker: str = None, isin: str = None, 
               prefer_api: bool = True) -> str:
    """
    Get sector using multiple methods with intelligent fallback
    
    Parameters:
    -----------
    security_name : str
        Company name
    ticker : str, optional
        Stock ticker for API lookup
    isin : str, optional
        ISIN code
    prefer_api : bool, default True
        Try API first before name-based inference
    
    Returns:
    --------
    str
        Sector name (never returns None, always has fallback)
    """
    # Try API first if ticker available and preferred
    if prefer_api and ticker:
        api_sector = fetch_sector_from_api(ticker)
        if api_sector:
            return api_sector
    
    # Fallback to name-based inference
    return infer_sector_from_name(security_name)


# ============================================================================
# TICKER MAPPING UTILITIES
# ============================================================================

# Common problematic stock name patterns -> Correct ticker
STOCK_NAME_TO_TICKER = {
    # PSU Banks
    'BANK OF BARODA': 'BANKBARODA.NS',
    'BANK OF MAHARASHTRA': 'MAHABANK.NS',
    'CENTRAL BANK OF INDIA': 'CENTRALBK.NS',
    'CENTRAL BANK': 'CENTRALBK.NS',
    'INDIAN BANK': 'INDIANB.NS',
    'PUNJAB NATIONAL BANK': 'PNB.NS',
    'STATE BANK OF INDIA': 'SBIN.NS',
    'UNION BANK OF INDIA': 'UNIONBANK.NS',
    'CANARA BANK': 'CANBK.NS',
    
    # Energy/Power PSUs
    'POWER GRID CORPORATION': 'POWERGRID.NS',
    'POWER GRID': 'POWERGRID.NS',
    'POWER FIN CORP': 'PFC.NS',
    'POWER FINANCE CORPORATION': 'PFC.NS',
    'NTPC': 'NTPC.NS',
    'COAL INDIA': 'COALINDIA.NS',
    'OIL AND NATURAL GAS CORP': 'ONGC.NS',
    'OIL AND NATURAL GAS CORPORATION': 'ONGC.NS',
    'OIL INDIA': 'OIL.NS',
    'INDIAN OIL CORP': 'IOC.NS',
    'INDIAN OIL CORPORATION': 'IOC.NS',
    'BHARAT PETROLEUM': 'BPCL.NS',
    'HINDUSTAN PETROLEUM CORP': 'HINDPETRO.NS',
    'HINDUSTAN PETROLEUM': 'HINDPETRO.NS',
    'GAIL (INDIA)': 'GAIL.NS',
    'GAIL': 'GAIL.NS',
    'NLC INDIA': 'NLCINDIA.NS',
    
    # Telecom
    'BHARTI AIRTEL': 'BHARTIARTL.NS',
    'RELIANCE JIO': 'RELIANCE.NS',  # Jio is part of Reliance
    'JIO FIN SERVICES': 'JIOFIN.NS',
    'JIO': 'JIOFIN.NS',
    
    # Manufacturing/Defense
    'HINDUSTAN AERONAUTICS': 'HAL.NS',
    'BHARAT ELECTRONICS': 'BEL.NS',
    'MAZAGON DOCK SHIPBUILDERS': 'MAZDOCK.NS',
    'BHARAT DYNAMICS': 'BDL.NS',
    
    # IT
    'TATA CONSULTANCY SERVICES': 'TCS.NS',
    'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS',
    'HCL TECHNOLOGIES': 'HCLTECH.NS',
    'TECH MAHINDRA': 'TECHM.NS',
    
    # Auto
    'TATA MOTORS': 'TATAMOTORS.NS',
    'TATA MOTORS PASS VEH': 'TATAMOTORS.NS',
    'MARUTI SUZUKI': 'MARUTI.NS',
    'MAHINDRA & MAHINDRA': 'M&M.NS',
    
    # Metals/Mining
    'NMDC': 'NMDC.NS',
    'STEEL AUTHORITY OF INDIA': 'SAIL.NS',
    'TATA STEEL': 'TATASTEEL.NS',
    'HINDALCO': 'HINDALCO.NS',
    
    # Infrastructure
    'IRCON INTERNATIONAL': 'IRCON.NS',
    'RITES': 'RITES.NS',
    'REC': 'RECLTD.NS',
    'POWER FINANCE CORPORATION': 'PFC.NS',
    'CENTRAL DEPO SER (I)': 'CDSL.NS',
    'CENTRAL DEPOSITORY SERVICES': 'CDSL.NS',
    
    # Renewable Energy
    'SUZLON ENERGY': 'SUZLON.NS',
    'SUZLON': 'SUZLON.NS',
    
    # Financial Services
    'TATA CAPITAL': 'TATACAPITAL.NS',
    
    # Others
    'RELIANCE INDUSTRIES': 'RELIANCE.NS',
    'RELIANCE': 'RELIANCE.NS',
    'ITC': 'ITC.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'AXIS BANK': 'AXISBANK.NS',
    
    # Graphite India
    'GRAPHITE INDIA': 'GRAPHITE.NS',
    
    # Madhav Copper
    'MADHAV COPPER': 'MADHAV.NS',
    'MADHAV INFRA PROJECTS': 'MADHAV.NS',
    
    # Choice International
    'CHOICE INTERNATIONAL': 'CHOICEIN.NS',
    'CHOICE': 'CHOICEIN.NS',
    
    # Tata AMC
    'TATAAML-TATSILV': 'TATAAMC.NS',
}

# ISIN to Ticker mapping (for precise matching)
ISIN_TO_TICKER = {
    # PSU Banks
    'INE028A01039': 'BANKBARODA.NS',  # Bank of Baroda
    'INE483A01010': 'CENTRALBK.NS',   # Central Bank
    'INE562A01011': 'INDIANB.NS',     # Indian Bank
    'INE160A01022': 'PNB.NS',         # PNB
    'INE062A01020': 'SBIN.NS',        # SBI
    'INE692A01016': 'UNIONBANK.NS',   # Union Bank
    
    # Energy
    'INE752E01010': 'POWERGRID.NS',   # Power Grid
    'INE733E01010': 'NTPC.NS',        # NTPC
    'INE522F01014': 'COALINDIA.NS',   # Coal India
    'INE213A01029': 'ONGC.NS',        # ONGC
    'INE274J01014': 'OIL.NS',         # Oil India
    'INE242A01010': 'IOC.NS',         # Indian Oil
    'INE171A01029': 'BPCL.NS',        # BPCL
    'INE094A01015': 'HINDPETRO.NS',   # Hindustan Petroleum
    'INE129A01019': 'GAIL.NS',        # GAIL
    'INE589A01014': 'NLCINDIA.NS',    # NLC India
    
    # Telecom
    'INE397D01024': 'BHARTIARTL.NS',  # Bharti Airtel
    'INE002A01018': 'RELIANCE.NS',    # Reliance
    
    # IT
    'INE467B01029': 'TCS.NS',         # TCS
    'INE009A01021': 'INFY.NS',        # Infosys
    'INE075A01022': 'WIPRO.NS',       # Wipro
    'INE860A01027': 'HCLTECH.NS',     # HCL Tech
    
    # Banks
    'INE040A01034': 'HDFCBANK.NS',    # HDFC Bank
    'INE090A01021': 'ICICIBANK.NS',   # ICICI Bank
    'INE238A01034': 'AXISBANK.NS',    # Axis Bank
    
    # Others
    'INE154A01025': 'ITC.NS',         # ITC
    'INE101A01026': 'MARUTI.NS',      # Maruti
    'INE155A01022': 'TATAMOTORS.NS',  # Tata Motors
    'INE101D01020': 'NMDC.NS',        # NMDC
    'INE202A01019': 'RECLTD.NS',      # REC
    'INE134E01011': 'PFC.NS',         # PFC
    'INE139A01034': 'SUZLON.NS',      # Suzlon
    'INE448A01013': 'RITES.NS',       # RITES
    'INE255A01020': 'GRAPHITE.NS',    # Graphite India
}


def get_ticker_from_name(stock_name: str) -> Optional[str]:
    """
    Get Yahoo Finance ticker from stock name
    
    Args:
        stock_name: Stock name (e.g., 'Bank of Baroda', 'COAL INDIA')
    
    Returns:
        Ticker symbol or None
    """
    if not stock_name:
        return None
    
    # Normalize name - remove common suffixes
    name_clean = stock_name.upper().strip()
    
    # Remove common suffixes
    suffixes_to_remove = [
        ' LIMITED', ' LTD.', ' LTD', ' CORP.', ' CORP', ' CORPORATION',
        ' INC.', ' INC', ' PVT.', ' PVT', ' PASS VEH LTD', ' (INDIA)',
        ' (I) LTD', ' SERVICES', ' SER (I) LTD'
    ]
    
    for suffix in suffixes_to_remove:
        if name_clean.endswith(suffix):
            name_clean = name_clean[:-len(suffix)].strip()
    
    # Direct lookup
    if name_clean in STOCK_NAME_TO_TICKER:
        return STOCK_NAME_TO_TICKER[name_clean]
    
    # Partial match (check if any key is in the name)
    for key, ticker in STOCK_NAME_TO_TICKER.items():
        if key in name_clean or name_clean in key:
            return ticker
    
    return None


def get_ticker_from_isin(isin: str) -> Optional[str]:
    """
    Get Yahoo Finance ticker from ISIN
    
    Args:
        isin: ISIN code (e.g., 'INE522F01014')
    
    Returns:
        Ticker symbol or None
    """
    if not isin:
        return None
    
    return ISIN_TO_TICKER.get(isin.upper().strip())


def get_ticker(stock_name: Optional[str] = None, isin: Optional[str] = None, 
               symbol: Optional[str] = None) -> str:
    """
    Get best possible ticker from available information
    
    Priority:
    1. Symbol (if already has .NS or .BO)
    2. ISIN mapping
    3. Stock name mapping
    4. Symbol + .NS
    
    Args:
        stock_name: Stock name
        isin: ISIN code
        symbol: Stock symbol
    
    Returns:
        Yahoo Finance ticker with .NS suffix
    """
    # 1. If symbol already has exchange suffix, use it
    if symbol and ('.NS' in symbol or '.BO' in symbol):
        return symbol
    
    # 2. Try ISIN mapping (most accurate)
    if isin:
        ticker = get_ticker_from_isin(isin)
        if ticker:
            return ticker
    
    # 3. Try stock name mapping
    if stock_name:
        ticker = get_ticker_from_name(stock_name)
        if ticker:
            return ticker
    
    # 4. Default: use symbol + .NS
    if symbol:
        return f"{symbol.upper()}.NS"
    
    # 5. Last resort: try first word of stock name
    if stock_name:
        first_word = stock_name.split()[0].upper()
        return f"{first_word}.NS"
    
    return "UNKNOWN.NS"


# ============================================================================
# PUBLIC API - BACKWARDS COMPATIBILITY
# ============================================================================

# Re-export commonly used functions at module level for easy imports
__all__ = [
    # CSV Utilities
    'detect_date_price_columns',
    'find_column_case_insensitive',
    'extract_holding_from_row',
    'safe_float_conversion',
    'create_diagnostics_report',
    
    # Sector Classification
    'infer_sector_from_name',
    'fetch_sector_from_api',
    'get_sector',
    
    # Ticker Mapping
    'get_ticker_from_name',
    'get_ticker_from_isin',
    'get_ticker',
    'STOCK_NAME_TO_TICKER',
    'ISIN_TO_TICKER',
]
