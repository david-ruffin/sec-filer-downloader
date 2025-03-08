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
        """Get the LLM instance for company name matching."""
        # Use ChatOpenAI for consistent behavior and simpler implementation
        return ChatOpenAI(
            model_name=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0  # Use 0.0 for deterministic, factual responses
        )

class SecFilingDownloader:
    """Downloader for SEC filings with LLM enhancement and conversational interface."""
    
    def __init__(self, sec_api_key: str):
        self.sec_api_key = sec_api_key
        self.query_api = QueryApi(api_key=sec_api_key)
        self.pdf_generator_api = PdfGeneratorApi(api_key=sec_api_key)
        self.company_lookup = CompanyLookup()
        # Initialize LLM for conversation
        self.llm = ChatOpenAI(
            model_name=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0
        )
        
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
    
    def extract_parameters(self, query: str) -> Dict:
        """Extract filing parameters from a natural language query.
        
        Args:
            query: User's natural language query about SEC filings
            
        Returns:
            Dictionary with extracted parameters (company, form_type, year)
        """
        prompt = f"""
        Extract the following parameters from this query about SEC filings:
        - Company: The company name or ticker symbol mentioned
        - Form Type: The type of SEC form (e.g., 10-K, 10-Q, 8-K, etc.)
        - Year: The year of the filing
        
        If any parameter is missing, set its value to None.
        Return the parameters as a JSON object with keys 'company', 'form_type', and 'year'.
        
        Query: {query}
        """
        
        # Get structured response from LLM
        response = self.llm.invoke(prompt)
        result = response.content.strip()
        
        # Extract JSON from response if needed
        if '{' in result:
            json_content = result[result.find('{'):result.rfind('}')+1]
            try:
                params = json.loads(json_content)
                return params
            except json.JSONDecodeError:
                print("Failed to parse parameters as JSON")
        
        # Default return if parsing fails
        return {"company": None, "form_type": None, "year": None}
    
    def confirm_understanding(self, params: Dict) -> str:
        """Generate a confirmation message based on extracted parameters.
        
        Args:
            params: Dictionary with extracted parameters
            
        Returns:
            Confirmation message string
        """
        company = params.get('company', 'unknown company')
        form_type = params.get('form_type', 'unknown form type')
        year = params.get('year', 'unknown year')
        
        # Handle cases where parameters are None
        if not company or company == 'None':
            company = 'unknown company'
        if not form_type or form_type == 'None':
            form_type = 'unknown form type'
        if not year or year == 'None':
            year = 'unknown year'
            
        confirmation_templates = [
            f"I'll look for {company}'s {form_type} filing from {year}. Is that correct?",
            f"Just to confirm, you want the {form_type} for {company} from {year}?",
            f"I understand you're looking for {company}'s {form_type} from {year}. Is that right?"
        ]
        
        import random
        return random.choice(confirmation_templates)
    
    def process_conversation(self) -> Optional[str]:
        """Have a conversation with the user to gather SEC filing parameters.
        
        Returns:
            Path to downloaded filing PDF or None if download failed
        """
        print("Welcome to the SEC Filing Downloader. I can help you find and download SEC filings.")
        print("What filing would you like to download? (e.g., 'I need Apple's 10-K from 2023')")
        
        # Initialize parameters
        params = {"company": None, "form_type": None, "year": None}
        confirmed = False
        
        while not confirmed:
            # Get user query
            user_input = input("> ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                return None
            
            # Extract parameters from user query
            if not all(params.values()):
                extracted = self.extract_parameters(user_input)
                
                # Update only missing parameters
                for key, value in extracted.items():
                    if (not params[key] or params[key] == 'None') and value and value != 'None':
                        params[key] = value
            
            # If we still have missing parameters, ask specifically for them
            if not params['company'] or params['company'] == 'None':
                print("What company are you interested in?")
                company = input("> ")
                params['company'] = company
                continue
                
            if not params['form_type'] or params['form_type'] == 'None':
                print("What type of filing do you need? (e.g., 10-K, 10-Q, 8-K)")
                form_type = input("> ")
                params['form_type'] = form_type
                continue
                
            if not params['year'] or params['year'] == 'None':
                print("For which year do you need this filing?")
                year = input("> ")
                params['year'] = year
                continue
            
            # Confirm understanding
            confirmation = self.confirm_understanding(params)
            print(confirmation)
            confirm_input = input("(yes/no) > ").lower()
            
            if confirm_input in ['y', 'yes', 'yeah', 'correct', 'right']:
                confirmed = True
            else:
                print("Let's try again. What filing are you looking for?")
                # Reset parameters that might be wrong
                params = {"company": None, "form_type": None, "year": None}
        
        # Parameters confirmed, download the filing
        print(f"Great! I'll download the {params['form_type']} filing for {params['company']} from {params['year']}")
        return self.download_by_company_name(params['company'], params['form_type'], params['year'])
    
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
    parser.add_argument("--interactive", action="store_true", help="Use conversational mode to gather filing parameters")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the downloader
    downloader = SecFilingDownloader(SEC_API_KEY)
    
    # Check if we should use interactive mode
    if args.interactive:
        print("Starting conversational SEC filing downloader...")
        downloader.process_conversation()
        return
    
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
