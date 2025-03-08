import os
import json
import argparse
from dotenv import load_dotenv
from sec_api import QueryApi, PdfGeneratorApi
from datetime import datetime
import re
from typing import Dict, List, Optional, Union

# Import Langchain components
from langchain_community.llms import OpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Get API keys from environment variables
SEC_API_KEY = os.getenv("SEC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

if not SEC_API_KEY:
    raise ValueError("SEC_API_KEY is not set in the .env file")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

class CompanyLookup:
    """Class for looking up company information in the SEC database."""
    
    def __init__(self, company_tickers_path: str = "reference_data/company_tickers.json"):
        self.company_data = self._load_company_data(company_tickers_path)
        
    def _load_company_data(self, company_tickers_path: str) -> Dict:
        """Load company data from the JSON file."""
        try:
            with open(company_tickers_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Company tickers file not found at {company_tickers_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in company tickers file at {company_tickers_path}")
    
    def find_company_by_name(self, company_name: str) -> Optional[Dict]:
        """Find company information by name or ticker."""
        # Use LLM to help with flexible matching
        llm = self._get_llm()
        
        # Create a prompt template for company name matching
        prompt_template = PromptTemplate(
            input_variables=["company_name"],
            template="""
            I need to match the company name "{company_name}" to one of the companies in the SEC database.
            The match should be the official company name or ticker symbol.
            Just respond with the exact company name or ticker symbol that best matches. 
            If there are multiple possible matches, pick the most likely one.
            Only respond with the exact match, nothing else.
            """
        )
        
        # Create a runnable sequence instead of LLMChain
        chain = prompt_template | llm
        # Use invoke instead of run
        result = chain.invoke({"company_name": company_name})
        normalized_name = result.strip() if isinstance(result, str) else result.content.strip()
        
        # Search for the normalized name in our data
        for _, company_info in self.company_data.items():
            if (normalized_name.lower() in company_info['title'].lower() or 
                normalized_name.lower() == company_info['ticker'].lower()):
                return company_info
                
        # If no exact match, try a more flexible search
        for _, company_info in self.company_data.items():
            if (company_name.lower() in company_info['title'].lower() or 
                company_name.lower() == company_info['ticker'].lower()):
                return company_info
        
        return None
    
    def _get_llm(self) -> BaseLanguageModel:
        """Get the LLM instance based on environment configuration."""
        callbacks = [StreamingStdOutCallbackHandler()]
        
        if OPENAI_MODEL.startswith("gpt-4") or OPENAI_MODEL.startswith("gpt-3.5"):
            return ChatOpenAI(
                model_name=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY,
                streaming=True,
                temperature=0.2,
                callbacks=callbacks
            )
        else:
            return OpenAI(
                model_name=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY,
                streaming=True,
                temperature=0.2,
                callbacks=callbacks
            )

class SecFilingDownloader:
    """Downloader for SEC filings with LLM enhancement."""
    
    def __init__(self, sec_api_key: str):
        self.sec_api_key = sec_api_key
        self.query_api = QueryApi(api_key=sec_api_key)
        self.pdf_generator_api = PdfGeneratorApi(api_key=sec_api_key)
        self.company_lookup = CompanyLookup()
        
    def get_sec_filing(self, cik: str, form_type: str, year: str) -> Dict:
        """Get a filing for a given CIK, form type, and year range."""
        # Set up the query to find the filing based on the provided parameters
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
        response = self.query_api.get_filings(query)
        return response
    
    def download_filing_as_pdf(self, sec_url: str, cik: str, form_type: str, year: str, filing_date: str) -> str:
        """Download filing as PDF and save to file."""
        # Create the output directory if it doesn't exist
        output_dir = "filings"
        os.makedirs(output_dir, exist_ok=True)
        
        print("Converting filing to PDF format...")
        # Generate a PDF from the filing
        pdf_content = self.pdf_generator_api.get_pdf(sec_url)
        
        # Create a nice filename using the company CIK, form type, and date
        date_str = filing_date.split('T')[0]
        output_filename = f"{cik}_{form_type}_{year}_{date_str}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the PDF content to a file
        with open(output_path, "wb") as f:
            f.write(pdf_content)
        
        return output_path
    
    def download_by_company_name(self, company_name: str, form_type: str, year: str) -> Optional[str]:
        """Download a filing using company name instead of CIK."""
        # Look up the company CIK
        company_info = self.company_lookup.find_company_by_name(company_name)
        
        if not company_info:
            print(f"Could not find company information for: {company_name}")
            return None
        
        cik = str(company_info['cik_str'])  # Convert CIK to string
        company_title = company_info['title']
        
        print(f"Found company: {company_title} (CIK: {cik})")
        
        try:
            print(f"Searching for {form_type} filings for {company_title} (CIK: {cik}) in {year}...")
            
            # Get the filing information
            filing_data = self.get_sec_filing(cik, form_type, year)
            
            if 'filings' not in filing_data or not filing_data['filings']:
                print(f"No {form_type} filings found for {company_title} (CIK: {cik}) in {year}")
                return None
            
            filing = filing_data['filings'][0]
            
            # Get the document URL from the filing data
            sec_url = filing.get('linkToFilingDetails')
            
            print(f"Found {form_type} filing dated {filing['filedAt']}")
            print(f"SEC URL: {sec_url}")
            
            # Download the filing as PDF
            output_path = self.download_filing_as_pdf(sec_url, cik, form_type, year, filing['filedAt'])
            
            print(f"Successfully downloaded and converted {form_type} to PDF: {output_path}")
            print("This PDF file should be much more readable than the HTML version.")
            
            return output_path
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download SEC filings for a company as PDF")
    parser.add_argument("--company", default="Apple", help="Company name or ticker symbol (default: Apple)")
    parser.add_argument("--form-type", default="10-K", help="SEC Form Type (default: 10-K)")
    parser.add_argument("--year", default="2023", help="Year of the filing (default: 2023)")
    parser.add_argument("--cik", help="Company CIK number (optional, overrides company name lookup)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the downloader
    downloader = SecFilingDownloader(SEC_API_KEY)
    
    # If CIK is provided directly, use the original method
    if args.cik:
        try:
            print(f"Searching for {args.form_type} filings for CIK {args.cik} in {args.year}...")
            
            # Get the filing information
            filing_data = downloader.get_sec_filing(args.cik, args.form_type, args.year)
            
            if 'filings' not in filing_data or not filing_data['filings']:
                print(f"No {args.form_type} filings found for CIK {args.cik} in {args.year}")
                return
            
            filing = filing_data['filings'][0]
            
            # Get the document URL from the filing data
            sec_url = filing.get('linkToFilingDetails')
            
            print(f"Found {args.form_type} filing dated {filing['filedAt']}")
            print(f"SEC URL: {sec_url}")
            
            # Download the filing as PDF
            output_path = downloader.download_filing_as_pdf(sec_url, args.cik, args.form_type, args.year, filing['filedAt'])
            
            print(f"Successfully downloaded and converted {args.form_type} to PDF: {output_path}")
            print("This PDF file should be much more readable than the HTML version.")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        # Use the company name to lookup and download
        downloader.download_by_company_name(args.company, args.form_type, args.year)

if __name__ == "__main__":
    main()
