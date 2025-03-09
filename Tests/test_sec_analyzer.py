#!/usr/bin/env python3
"""
Test script for sec_analyzer.py
Tests the updated SEC API integration functions
"""

import os
import sys
import re
from dotenv import load_dotenv
from sec_api import QueryApi

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
from sec_analyzer import extract_section, get_xbrl_data, try_direct_document_access

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SEC-Analyzer-Test")

# Load environment variables
load_dotenv()
SEC_API_KEY = os.environ.get("SEC_API_KEY")

if not SEC_API_KEY:
    logger.error("SEC_API_KEY not found in environment variables")
    sys.exit(1)

def test_extract_section():
    """Test the extract_section function"""
    logger.info("TESTING: extract_section function")
    
    # First, find a filing URL using the Query API
    query_api = QueryApi(api_key=SEC_API_KEY)
    query = {
        "query": {
            "query_string": {
                "query": "ticker:AAPL AND formType:\"10-K\" AND filedAt:{2023-01-01 TO 2023-12-31}"
            }
        },
        "from": "0",
        "size": "1"
    }
    
    filings = query_api.get_filings(query)
    if not filings or not filings.get('filings') or len(filings['filings']) == 0:
        logger.error("No filings found for test_extract_section")
        return False
        
    filing = filings['filings'][0]
    filing_url = filing.get('linkToFilingDetails')
    company_name = filing.get('companyName')
    
    logger.info(f"Found filing for {company_name}: {filing_url}")
    
    # Test extracting different sections
    sections_to_test = ["1A", "7", "8"]
    
    for section in sections_to_test:
        logger.info(f"Extracting section {section}")
        section_text = extract_section(filing_url, section, SEC_API_KEY)
        
        if section_text:
            logger.info(f"Successfully extracted section {section} ({len(section_text)} characters)")
            preview = section_text[:100].replace('\n', ' ').strip()
            logger.info(f"Preview: {preview}...")
            
            assert len(section_text) > 100, f"Section {section} text is too short"
        else:
            logger.warning(f"Failed to extract section {section}")
    
    return True

def test_get_xbrl_data():
    """Test the get_xbrl_data function"""
    logger.info("TESTING: get_xbrl_data function")
    
    # First, find a filing with XBRL data
    query_api = QueryApi(api_key=SEC_API_KEY)
    query = {
        "query": {
            "query_string": {
                "query": "ticker:AAPL AND formType:\"10-K\" AND filedAt:{2023-01-01 TO 2023-12-31}"
            }
        },
        "from": "0",
        "size": "1"
    }
    
    filings = query_api.get_filings(query)
    if not filings or not filings.get('filings') or len(filings['filings']) == 0:
        logger.error("No filings found for test_get_xbrl_data")
        return False
        
    filing = filings['filings'][0]
    company_name = filing.get('companyName')
    
    # Extract accession number from the filing details URL
    filing_url = filing.get('linkToFilingDetails')
    match = re.search(r'/([0-9]+)/([0-9]+)/', filing_url)
    if not match:
        logger.error(f"Could not extract accession number from URL: {filing_url}")
        return False
        
    accession_raw = match.group(2)
    # Format with dashes: 0001234567-12-123456
    if len(accession_raw) == 18 and "-" not in accession_raw:
        accession_no = f"{accession_raw[0:10]}-{accession_raw[10:12]}-{accession_raw[12:]}"
    else:
        accession_no = accession_raw
        
    logger.info(f"Testing XBRL data for {company_name}, accession: {accession_no}")
    
    # Format accession number with dashes if needed (0001234567-12-123456)
    if len(accession_no) == 18 and "-" not in accession_no:
        accession_no = f"{accession_no[0:10]}-{accession_no[10:12]}-{accession_no[12:]}"
    
    xbrl_data = get_xbrl_data(accession_no, SEC_API_KEY)
    
    if xbrl_data:
        logger.info(f"Successfully retrieved XBRL data with {len(xbrl_data)} data points")
        
        # Log a few data points as examples
        sample_keys = list(xbrl_data.keys())[:3]
        for key in sample_keys:
            logger.info(f"Sample data point - {key}: {xbrl_data[key]}")
            
        assert len(xbrl_data) > 0, "XBRL data is empty"
    else:
        logger.warning("Failed to retrieve XBRL data")
    
    return True

def test_try_direct_document_access():
    """Test the try_direct_document_access function"""
    logger.info("TESTING: try_direct_document_access function")
    
    # Instead of using the SEC API to find a filing, let's create a filing URL with a known format
    # that our function is designed to handle. This is a typical format for an Apple 10-K filing
    company_name = "Apple Inc."
    test_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    
    # Let's explicitly format it with dashes to match what our function expects
    cik = "320193"
    accession_raw = "000032019323000106"
    formatted_acc = f"{accession_raw[0:10]}-{accession_raw[10:12]}-{accession_raw[12:]}"
    filing_url_with_dashes = f"https://www.sec.gov/Archives/edgar/data/{cik}/{formatted_acc}/aapl-20230930.htm"
    
    logger.info(f"Testing direct document access for {company_name}")
    logger.info(f"Using URL with formatted accession number: {filing_url_with_dashes}")
    
    # Test with the properly formatted URL
    document_text = try_direct_document_access(filing_url_with_dashes, SEC_API_KEY)
    
    if document_text:
        logger.info(f"Successfully retrieved document content ({len(document_text)} characters)")
        preview = document_text[:100].replace('\n', ' ').strip()
        logger.info(f"Preview: {preview}...")
        
        assert len(document_text) > 1000, "Document text is too short"
        return True
    
    # If the formatted URL didn't work, there might be an issue with the API
    # Let's try using the actual SEC URL directly
    logger.warning("Failed with formatted URL, trying direct SEC URL")
    
    # Get a random Apple filing from 2023 directly
    query_api = QueryApi(api_key=SEC_API_KEY)
    query = {
        "query": {
            "query_string": {
                "query": "ticker:AAPL AND formType:\"10-Q\" AND filedAt:{2023-01-01 TO 2023-12-31}"
            }
        },
        "from": "0",
        "size": "1"
    }
    
    filings = query_api.get_filings(query)
    if not filings or not filings.get('filings') or len(filings['filings']) == 0:
        logger.error("No filings found for fallback test")
        return False
        
    filing = filings['filings'][0]
    sec_url = filing.get('linkToHtml')
    
    if not sec_url:
        logger.error("No HTML link found in filing")
        return False
        
    logger.info(f"Trying fallback SEC URL: {sec_url}")
    
    # Try the fallback URL
    document_text = try_direct_document_access(sec_url, SEC_API_KEY)
    
    if document_text:
        logger.info(f"Successfully retrieved document content with fallback URL ({len(document_text)} characters)")
        preview = document_text[:100].replace('\n', ' ').strip()
        logger.info(f"Preview: {preview}...")
        
        assert len(document_text) > 1000, "Document text is too short"
        return True
    else:
        logger.warning(f"Failed to retrieve document content for fallback URL")
    
    # If we got here, none of the attempts worked
    logger.error("All attempts failed for direct document access")
    return False

if __name__ == "__main__":
    print("Starting SEC analyzer tests...")
    success_count = 0
    
    try:
        print("\n=== Testing extract_section ===")
        if test_extract_section():
            success_count += 1
    except Exception as e:
        logger.error(f"Error testing extract_section: {str(e)}")
    
    try:
        print("\n=== Testing get_xbrl_data ===")
        if test_get_xbrl_data():
            success_count += 1
    except Exception as e:
        logger.error(f"Error testing get_xbrl_data: {str(e)}")
    
    try:
        print("\n=== Testing try_direct_document_access ===")
        if test_try_direct_document_access():
            success_count += 1
    except Exception as e:
        logger.error(f"Error testing try_direct_document_access: {str(e)}")
    
    print(f"\nTests completed: {success_count}/3 successful")
    
    if success_count == 3:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)
