"""
Indian Stock Symbol Mapper
Maps common stock names and ISIN codes to correct Yahoo Finance tickers
"""

from typing import Optional

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
