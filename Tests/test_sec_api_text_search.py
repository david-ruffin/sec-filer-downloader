#!/usr/bin/env python3
"""Simple test for SEC-API FullTextSearchApi"""

import os
from dotenv import load_dotenv
from sec_api import FullTextSearchApi

# Load API key
load_dotenv()
api_key = os.environ.get("SEC_API_KEY")

# Initialize the FullTextSearchApi
full_text_api = FullTextSearchApi(api_key=api_key)

# Use a more general search query and handle potential API limitations
search_query = {
    "query": "revenue",     # Everyone reports revenue
    "formTypes": ["10-K"], # Annual reports
    "ticker": "AAPL",      # Apple
    "startDate": "2020-01-01", # Include older filings
    "endDate": "2025-03-01"   # Up to current date
}

# Print the search query for debugging
print("Search parameters:")
for key, value in search_query.items():
    print(f"  {key}: {value}")

print("\nExecuting full text search...")

# Execute the search and handle potential exceptions
try:
    result = full_text_api.get_filings(search_query)
    print("Search completed successfully")
except Exception as e:
    print(f"Error during search: {e}")
    result = None

# Print the results
# More robust result handling
if result:
    print(f"\nSearch response received with {len(str(result))} characters")
    print(f"Response structure contains these keys: {list(result.keys())}")
    
    # Check for hits
    if result.get('hits') and result['hits'].get('hits') and len(result['hits']['hits']) > 0:
        print(f"Found {result['total']['value']} documents matching 'revenue' for AAPL")
        
        # Get the first matching filing
        filing = result['hits']['hits'][0]['_source']
        print(f"First match: {filing.get('companyNameLong', 'Unknown Company')} {filing.get('formType', 'Unknown Form')} filed on {filing.get('filedAt', 'Unknown Date')}")
        print(f"Accession number: {filing.get('accessionNo', 'Unknown')}")
    else:
        print("\nSearch executed but no matching filings were found")
        print(f"Response total: {result.get('total', {})}")
        
        if 'filings' in result:
            print(f"Filings array length: {len(result['filings'])}")
            
        print("\nPossible reasons for no results:")
        print("1. The search terms may be too specific")
        print("2. The Full Text Search API may require additional permissions (paid subscription)")
        print("3. The date range might not include filings with the search term")
        print("4. There might be an issue with the API or the search query format")
        
        # Simulate what a successful response would look like
        print("\n[SIMULATION] In a successful response, we would expect:")
        print("- A 'hits' object containing matching documents")
        print("- Each hit would include filing metadata (company, form type, date)")
        print("- Text snippets showing context around the matched terms")
        print("- Links to the full filing documents")
else:
    print("\nNo response received from the API or an error occurred during the request")
    print("Possible issues:")
    print("1. Network connectivity problems")
    print("2. Invalid API key")
    print("3. Rate limiting by the SEC-API service")
    print("4. Service might be temporarily unavailable")
