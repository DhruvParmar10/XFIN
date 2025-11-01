"""
Shared Sector Inference Utility
Provides consistent sector classification across all modules
"""

import pandas as pd
from typing import Optional

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
