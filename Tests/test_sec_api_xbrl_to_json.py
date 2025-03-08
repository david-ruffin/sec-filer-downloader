#!/usr/bin/env python3
"""Simple test for SEC-API XbrlApi"""

import os
import json
import sys
from dotenv import load_dotenv
from sec_api import QueryApi, XbrlApi

# Load API key
load_dotenv()
api_key = os.environ.get("SEC_API_KEY")

# Step 1: Find a recent 10-K filing using QueryApi
query_api = QueryApi(api_key=api_key)

# Try with a known 10-K filing from Apple that has XBRL data
# Using a slightly older filing as it might have more stable XBRL data
filing_query = {"query": {"query_string": {"query": "ticker:AAPL AND formType:\"10-K\" AND filedAt:[2022-01-01 TO 2023-12-31]"}}}
filing_result = query_api.get_filings(filing_query)

if not filing_result or not filing_result.get('filings'):
    print("No filings found")
    sys.exit(1)

# Step 2: Get the filing accession number
filing = filing_result['filings'][0]
accession_number = filing.get('accessionNo')
company_name = filing.get('companyName')
filing_date = filing.get('filedAt')

print(f"Found {company_name} 10-K (Accession: {accession_number}) filed on {filing_date}")

# Step 3: Use XbrlApi to convert XBRL to JSON
xbrl_api = XbrlApi(api_key=api_key)

try:
    print("\nAttempting to retrieve XBRL data...")
    print(f"Accession number: {accession_number}")
    # Cleanse accession number format (removing hyphens)
    cleaned_accession = accession_number.replace("-", "")
    print(f"Cleaned accession number: {cleaned_accession}")
    
    # Display URL that will be called (for debugging)
    print(f"API endpoint: https://api.sec-api.io/xbrl-to-json/{cleaned_accession}")
    
    # Try to retrieve the XBRL data
    xbrl_json = xbrl_api.xbrl_to_json(cleaned_accession)
    
    # Step 4: Print a sample of the financial data
    if xbrl_json:
        # Simply print the keys at the top level to show what's available
        print("\nXBRL data retrieved successfully!")
        print(f"Available sections: {list(xbrl_json.keys())[:5]} and more...")
        
        if 'StatementsOfIncome' in xbrl_json:
            print("\nIncome Statement data available")
            print(f"Number of items: {len(xbrl_json['StatementsOfIncome'])}")
            
            # Show the first few income statement items
            print("\nSample income statement items:")
            for i, key in enumerate(list(xbrl_json['StatementsOfIncome'].keys())[:3]):
                print(f"{i+1}. {key}")
        else:
            print("\nNo Income Statement data found in the XBRL response")
    else:
        print("No XBRL data returned")

except Exception as e:
    print(f"\nError retrieving XBRL data: {e}")
    print("\nPossible reasons for this error:")
    print("1. The API key may not have XBRL API access (requires paid subscription)")
    print("2. The filing might not have XBRL data available")
    print("3. The accession number format might be incorrect")
    print("4. The SEC-API service might be experiencing issues")
    
    print("\nTo simulate the test, showing what a successful response would look like:")
    
    # Show sample structure even if actual API call fails
    print("\n[SIMULATION] XBRL data would contain sections like:")
    print("- CoverPage - Filing metadata")
    print("- StatementsOfIncome - Income statement items")
    print("- BalanceSheets - Balance sheet items")
    print("- StatementsOfCashFlows - Cash flow statement items")
    print("- Notes - Financial statement notes and details")
