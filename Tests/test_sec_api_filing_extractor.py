#!/usr/bin/env python3
"""Simple test for SEC-API ExtractorApi"""

import os
from dotenv import load_dotenv
from sec_api import QueryApi, ExtractorApi

# Load API key
load_dotenv()
api_key = os.environ.get("SEC_API_KEY")

# Step 1: Find a recent 10-K filing using QueryApi
query_api = QueryApi(api_key=api_key)
filing_query = {"query": {"query_string": {"query": "ticker:AAPL AND formType:\"10-K\""}}}
filing_result = query_api.get_filings(filing_query)

if not filing_result or not filing_result.get('filings'):
    print("No filings found")
    exit()

# Step 2: Get the filing URL from the result
filing = filing_result['filings'][0]
filing_url = filing.get('linkToFilingDetails')
company_name = filing.get('companyName')
filing_date = filing.get('filedAt')

print(f"Found {company_name} 10-K filed on {filing_date}")

# Step 3: Use ExtractorApi to extract Risk Factors (Item 1A)
extractor_api = ExtractorApi(api_key=api_key)
risk_factors = extractor_api.get_section(filing_url, "1A", "text")

# Print an excerpt of the risk factors
if risk_factors:
    # Show just the first 200 characters as a preview
    preview = risk_factors[:200].replace('\n', ' ').strip()
    print(f"\nRisk Factors excerpt:")
    print(f"{preview}...")
    print(f"\nExtracted {len(risk_factors)} characters of risk factor content")
else:
    print("No risk factors found in the filing")
