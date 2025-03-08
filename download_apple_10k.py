import os
import json
import argparse
from dotenv import load_dotenv
from sec_api import QueryApi, PdfGeneratorApi
from datetime import datetime

# Load environment variables
load_dotenv()

# Get SEC API key from environment variables
sec_api_key = os.getenv("SEC_API_KEY")

if not sec_api_key:
    raise ValueError("SEC_API_KEY is not set in the .env file")

# Function to get a filing for a given CIK, form type, and year range
def get_sec_filing(cik, form_type, year):
    query_api = QueryApi(api_key=sec_api_key)
    
    # Set up the query to find the filing based on the provided parameters
    # Using a date range to ensure we get filings for the specified year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    query = {
        "query": {
            "query_string": {
                "query": f"cik:{cik} AND formType:\"{form_type}\" AND filedAt:[{start_date} TO {end_date}]"
            }
        },
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    
    # Get the filing information
    response = query_api.get_filings(query)
    return response

# Function to download filing as PDF and save to file
def download_filing_as_pdf(sec_url, cik, form_type, year, filing_date):
    # Create the output directory if it doesn't exist
    output_dir = "filings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the PDF Generator API with the API key
    pdf_generator_api = PdfGeneratorApi(api_key=sec_api_key)
    
    print("Converting filing to PDF format...")
    # Generate a PDF from the filing
    pdf_content = pdf_generator_api.get_pdf(sec_url)
    
    # Create a nice filename using the company CIK, form type, and date
    date_str = filing_date.split('T')[0]
    output_filename = f"{cik}_{form_type}_{year}_{date_str}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the PDF content to a file
    with open(output_path, "wb") as f:
        f.write(pdf_content)
    
    return output_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download SEC filings for a company as PDF")
    parser.add_argument("--cik", default="320193", help="Company CIK number (default: 320193 for Apple)")
    parser.add_argument("--form-type", default="10-K", help="SEC Form Type (default: 10-K)")
    parser.add_argument("--year", default="2023", help="Year of the filing (default: 2023)")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        print(f"Searching for {args.form_type} filings for CIK {args.cik} in {args.year}...")
        
        # Get the filing information
        filing_data = get_sec_filing(args.cik, args.form_type, args.year)
        
        if 'filings' not in filing_data or not filing_data['filings']:
            print(f"No {args.form_type} filings found for CIK {args.cik} in {args.year}")
            return
        
        filing = filing_data['filings'][0]
        
        # Get the document URL from the filing data
        sec_url = filing.get('linkToFilingDetails')
        
        print(f"Found {args.form_type} filing dated {filing['filedAt']}")
        print(f"SEC URL: {sec_url}")
        
        # Download the filing as PDF
        output_path = download_filing_as_pdf(sec_url, args.cik, args.form_type, args.year, filing['filedAt'])
        
        print(f"Successfully downloaded and converted {args.form_type} to PDF: {output_path}")
        print("This PDF file should be much more readable than the HTML version.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
