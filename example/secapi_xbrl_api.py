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
filing_query = {"query": {"query_string": {"query": "ticker:AAPL AND formType:\"10-K\" AND filedAt:[2022-01-01 TO 2023-12-31]"}}}
filing_result = query_api.get_filings(filing_query)

if not filing_result or not filing_result.get('filings'):
    print("No filings found")
    sys.exit(1)

# Step 2: Get the filing accession number and URL
filing = filing_result['filings'][0]
accession_number = filing.get('accessionNo')
company_name = filing.get('companyName')
filing_date = filing.get('filedAt')
filing_url = filing.get('htmlUrl')

print(f"Found {company_name} 10-K (Accession: {accession_number}) filed on {filing_date}")
print(f"Filing URL: {filing_url}")

# Step 3: Use XbrlApi to convert XBRL to JSON
xbrl_api = XbrlApi(api_key=api_key)

try:
    print("\nAttempting to retrieve XBRL data...")
    
    # According to the documentation, we can use accession-no, htm-url, or xbrl-url
    # Let's try both accession number and htm-url for demonstration
    print(f"Method 1: Using accession number: {accession_number}")
    
    # The API expects the accession number without hyphens as a query parameter
    # NOT as part of the URL path
    xbrl_json = None
    
    if filing_url:
        print(f"\nMethod 2: Using HTML URL: {filing_url}")
        print("Calling API with htm-url parameter...")
        xbrl_json = xbrl_api.xbrl_to_json(htm_url=filing_url)
    else:
        print("HTML URL not available, trying accession number...")
        xbrl_json = xbrl_api.xbrl_to_json(accession_no=accession_number)
    
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
                
            # Show a sample of the data for one item
            first_key = list(xbrl_json['StatementsOfIncome'].keys())[0]
            sample_data = xbrl_json['StatementsOfIncome'][first_key][0]
            print(f"\nSample data for {first_key}:")
            print(json.dumps(sample_data, indent=2))
        
        elif 'BalanceSheets' in xbrl_json:
            print("\nBalance Sheet data available")
            print(f"Number of items: {len(xbrl_json['BalanceSheets'])}")
            
            # Show the first few balance sheet items
            print("\nSample balance sheet items:")
            for i, key in enumerate(list(xbrl_json['BalanceSheets'].keys())[:3]):
                print(f"{i+1}. {key}")
        else:
            print("\nNo standard financial statements found in the XBRL response")
            print(f"Available sections: {list(xbrl_json.keys())}")
    else:
        print("No XBRL data returned")

except Exception as e:
    print(f"\nError retrieving XBRL data: {e}")
    print("\nDetails about the error and next steps:")
    print("1. Make sure your SEC API key is valid and has XBRL API access")
    print("2. Verify the filing has XBRL data available")
    print("3. Check the documentation at https://sec-api.io/docs/xbrl-to-json-converter-api")
    print("4. Try with a different filing or using a different parameter (htm-url, xbrl-url, or accession-no)")
