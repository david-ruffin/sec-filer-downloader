#!/usr/bin/env python3
"""Simple test for SEC-API FullTextSearchApi"""

import os
import json
from dotenv import load_dotenv
from sec_api import FullTextSearchApi

# Load API key
load_dotenv()
api_key = os.environ.get("SEC_API_KEY")

# Initialize the FullTextSearchApi
full_text_api = FullTextSearchApi(api_key=api_key)

# Use the simple and correct query format for the SEC-API package
# According to documentation: https://sec-api.io/docs/full-text-search-api
search_query = {
    "query": "revenue",             # Search for revenue in filings
    "formTypes": ["10-K"],        # Annual reports only
    "ciks": ["0000320193"],       # Apple Inc. CIK number
    "startDate": "2020-01-01",    # From 2020
    "endDate": "2023-12-31"       # To end of 2023
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

# Print the results using the correct response structure
if result:
    print(f"\nSearch response received with {len(str(result))} characters")
    print(f"Response structure contains these keys: {list(result.keys())}")
    
    # Process the actual response structure with 'total' and 'filings' keys
    total_count = 0
    if 'total' in result and isinstance(result['total'], dict):
        total_count = result['total'].get('value', 0)
        relation = result['total'].get('relation', 'eq')
        print(f"Found {total_count} total documents matching the search criteria")
    
    # Check for filings
    if 'filings' in result and isinstance(result['filings'], list) and len(result['filings']) > 0:
        filings = result['filings']
        print(f"Showing {len(filings)} filings in this response")
        
        # Display the first few results
        for i, filing in enumerate(filings[:3]):
            print(f"\nFiling {i+1}:")
            print(f"  Company: {filing.get('companyNameLong', filing.get('companyName', 'Unknown'))}")
            print(f"  Form: {filing.get('formType', 'Unknown')}")
            print(f"  Filed: {filing.get('filedAt', 'Unknown')}")
            print(f"  Accession: {filing.get('accessionNo', 'Unknown')}")
            print(f"  HTML URL: {filing.get('linkToHtml', filing.get('linkToFilingDetails', 'Unknown'))}")
            
            # Show excerpt if available
            if 'excerpt' in filing and filing['excerpt']:
                print("\n  Excerpt:")
                excerpt = filing['excerpt'].replace('<span class="highlight">', '*').replace('</span>', '*')
                print(f"    {excerpt[:200]}...")
    else:
        print("\nNo filings found in the search results, but got response")
        if 'filings' in result:
            print(f"The 'filings' key contains a list with {len(result['filings'])} items")
            
        # If we have 'filings' but it's empty or weird, let's look deeper
        if 'filings' in result and isinstance(result['filings'], list) and len(result['filings']) > 0:
            sample = result['filings'][0]
            print("Sample filing keys:")
            print(f"  {list(sample.keys())}")
            
        print("\nPossible reasons for no results:")
        print("1. Check if the ticker 'AAPL' is correct (it should be)")
        print("2. The API might be returning results in an unexpected format")
        print("3. Try with a more general search term like 'financial'")
else:
    print("\nNo response received from the API or an error occurred during the request")
    print("Check your API key and connectivity to SEC-API services")

print("\nTest completed")
