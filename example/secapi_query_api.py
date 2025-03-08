#!/usr/bin/env python3
"""Simple test for SEC-API QueryApi"""

import os
from dotenv import load_dotenv
from sec_api import QueryApi

# Load API key
load_dotenv()
api_key = os.environ.get("SEC_API_KEY")

# Initialize the QueryApi
query_api = QueryApi(api_key=api_key)

# Test 1: Search for Apple's most recent 10-K
print("Test 1: Search for Apple's most recent 10-K")
apple_10k_query = {
    "query": {
        "query_string": {
            "query": "ticker:AAPL AND formType:\"10-K\""
        }
    },
    "from": 0,
    "size": 1,
    "sort": [{"filedAt": {"order": "desc"}}]
}

result = query_api.get_filings(apple_10k_query)

if result and result.get('filings'):
    filing = result['filings'][0]
    print(f"Found: {filing.get('companyName')} {filing.get('formType')} filed on {filing.get('filedAt')}")
    print(f"Accession number: {filing.get('accessionNo')}")
else:
    print("No filings found")

# Test 2: Search for Apple's Form 8-K filings in the last year
print("\nTest 2: Search for Apple's recent 8-K filings")
today = "2025-03-08"  # Hard-coded for reproducibility
one_year_ago = "2024-03-08"  # Hard-coded for reproducibility
apple_8k_query = {
    "query": {
        "query_string": {
            "query": f"ticker:AAPL AND formType:\"8-K\" AND filedAt:[{one_year_ago} TO {today}]"
        }
    },
    "from": 0,
    "size": 5,
    "sort": [{"filedAt": {"order": "desc"}}]
}

result = query_api.get_filings(apple_8k_query)

if result and result.get('filings'):
    print(f"Found {len(result.get('filings'))} recent 8-K filings for Apple")
    for i, filing in enumerate(result.get('filings')[:3], 1):  # Show top 3
        print(f"{i}. {filing.get('formType')} filed on {filing.get('filedAt')}")
        
        # Print 8-K item types if available
        if filing.get('items'):
            items = ", ".join(filing.get('items'))
            print(f"   Items: {items}")
else:
    print("No 8-K filings found")

# Test 3: Search for Apple's quarterly reports (10-Q) in the past year
print("\nTest 3: Search for Apple's recent 10-Q filings")
today = "2025-03-08"  # Hard-coded for reproducibility
one_year_ago = "2024-03-08"  # Hard-coded for reproducibility
quarterly_query = {
    "query": {
        "query_string": {
            "query": f"ticker:AAPL AND formType:\"10-Q\" AND filedAt:[{one_year_ago} TO {today}]"
        }
    },
    "from": 0,
    "size": 5,
    "sort": [{"filedAt": {"order": "desc"}}]
}

result = query_api.get_filings(quarterly_query)

if result and result.get('filings'):
    print(f"Found {len(result.get('filings'))} recent 10-Q filings for Apple")
    for i, filing in enumerate(result.get('filings')[:3], 1):  # Show top 3
        print(f"{i}. 10-Q for period ending {filing.get('periodOfReport', 'Unknown')} filed on {filing.get('filedAt')}")
else:
    print("No recent 10-Q filings found")
