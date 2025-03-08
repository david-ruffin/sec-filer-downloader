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

# Import logging utilities
from utils.logger import get_logger, log_section_boundary

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
    
    def __init__(self, company_tickers_path: str = "reference_data/company_tickers.json", llm: Optional[BaseLanguageModel] = None):
        self.logger = get_logger()
        self.logger.info("[CompanyLookup] Initializing")
        self.company_data = self._load_company_data(company_tickers_path)
        self.llm = llm
        self.logger.info(f"[CompanyLookup] Initialized with {len(self.company_data)} companies")
        
    def _load_company_data(self, company_tickers_path: str) -> Dict:
        """Load company data from the JSON file."""
        self.logger.info(f"Loading company data from {company_tickers_path}")
        try:
            with open(company_tickers_path, 'r') as f:
                data = json.load(f)
                self.logger.info(f"Successfully loaded company data with {len(data)} entries")
                return data
        except FileNotFoundError:
            self.logger.error(f"Company tickers file not found at {company_tickers_path}")
            raise FileNotFoundError(f"Company tickers file not found at {company_tickers_path}")
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in company tickers file at {company_tickers_path}")
            raise ValueError(f"Invalid JSON format in company tickers file at {company_tickers_path}")
    
    def find_company_by_name(self, company_name: str) -> Optional[Dict]:
        """Find company information by name or ticker."""
        self.logger.info(f"Finding company by name: {company_name}")
        # Use LLM to help with flexible matching
        if not self.llm:
            self.logger.debug("No LLM provided, initializing default LLM")
            self.llm = self._get_llm()
        
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
        self.logger.debug("Creating prompt chain for company name normalization")
        chain = prompt_template | self.llm
        # Use invoke instead of run
        self.logger.debug(f"Invoking LLM to normalize company name: {company_name}")
        result = chain.invoke({"company_name": company_name})
        normalized_name = result.strip() if isinstance(result, str) else result.content.strip()
        self.logger.info(f"Normalized company name: '{company_name}' to '{normalized_name}'")
        
        # Search for the normalized name in our data
        self.logger.debug(f"Searching for normalized name '{normalized_name}' in company database")
        for _, company_info in self.company_data.items():
            if (normalized_name.lower() in company_info['title'].lower() or 
                normalized_name.lower() == company_info['ticker'].lower()):
                self.logger.info(f"Found exact match for '{normalized_name}': {company_info['title']} (CIK: {company_info['cik_str']})")
                return company_info
                
        # If no exact match, try a more flexible search
        self.logger.debug(f"No exact match found, trying flexible search with original name '{company_name}'")
        for _, company_info in self.company_data.items():
            if (company_name.lower() in company_info['title'].lower() or 
                company_name.lower() == company_info['ticker'].lower()):
                self.logger.info(f"Found flexible match for '{company_name}': {company_info['title']} (CIK: {company_info['cik_str']})")
                return company_info
        
        self.logger.warning(f"No company match found for '{company_name}'")
        return None
    
    def _get_llm(self) -> BaseLanguageModel:
        """Create a new LLM instance if one wasn't provided.
        
        Note: This is a fallback method in case an LLM wasn't passed to the constructor.
        """
        return ChatOpenAI(
            model_name=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0
        )

class SecFilingDownloader:
    """Downloader for SEC filings with LLM enhancement and conversational interface."""
    
    def __init__(self, sec_api_key: str):
        self.logger = get_logger()
        self.logger.info("[Downloader] Initializing SecFilingDownloader")
        
        self.sec_api_key = sec_api_key
        self.query_api = QueryApi(api_key=sec_api_key)
        self.pdf_generator_api = PdfGeneratorApi(api_key=sec_api_key)
        self.logger.info("[Downloader] SEC API clients initialized")
        
        # Initialize LLM for conversation and company lookup
        self.logger.info(f"[Downloader] Initializing LLM model: {OPENAI_MODEL}")
        self.llm = ChatOpenAI(
            model_name=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0
        )
        
        # Pass the LLM to CompanyLookup
        self.logger.info("[Downloader] Initializing CompanyLookup")
        self.company_lookup = CompanyLookup(llm=self.llm)
        self.logger.info("[Downloader] SecFilingDownloader initialization complete")
        
    def get_sec_filing(self, cik: str, form_type: str, year: str) -> Dict:
        """Get a filing for a given CIK, form type, and year range."""
        log_section_boundary(f"Get SEC Filing - CIK:{cik}, Form:{form_type}, Year:{year}", True)
        
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
        
        self.logger.info(f"Querying SEC API for CIK:{cik}, Form:{form_type}, Year:{year}")
        self.logger.debug(f"Query: {json.dumps(query)}")
        
        # Get the filing information
        try:
            response = self.query_api.get_filings(query)
            if 'filings' in response and response['filings']:
                self.logger.info(f"Found {len(response['filings'])} filing(s)")
                for i, filing in enumerate(response['filings']):
                    self.logger.info(f"Filing {i+1}: {filing.get('formType')} filed on {filing.get('filedAt')}")
            else:
                self.logger.warning(f"No filings found for CIK:{cik}, Form:{form_type}, Year:{year}")
            
            log_section_boundary(f"Get SEC Filing - CIK:{cik}, Form:{form_type}, Year:{year}", False)
            return response
        except Exception as e:
            self.logger.error(f"Error querying SEC API: {str(e)}")
            log_section_boundary(f"Get SEC Filing - CIK:{cik}, Form:{form_type}, Year:{year}", False)
            raise
    
    def download_filing_as_pdf(self, sec_url: str, cik: str, form_type: str, year: str, filing_date: str) -> str:
        """Download filing as PDF and save to file."""
        log_section_boundary(f"Download Filing As PDF - CIK:{cik}, Form:{form_type}", True)
        
        # Create the output directory if it doesn't exist
        output_dir = "filings"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"[PDF] Output directory ensured: {output_dir}")
        
        print("Converting filing to PDF format...")
        self.logger.info(f"[PDF] Generating PDF from SEC URL: {sec_url}")
        
        try:
            # Generate a PDF from the filing
            pdf_content = self.pdf_generator_api.get_pdf(sec_url)
            self.logger.info(f"[PDF] PDF generated successfully, size: {len(pdf_content)} bytes")
            
            # Create a nice filename using the company CIK, form type, and date
            date_str = filing_date.split('T')[0]
            output_filename = f"{cik}_{form_type}_{year}_{date_str}.pdf"
            output_path = os.path.join(output_dir, output_filename)
            self.logger.info(f"[PDF] Saving PDF to: {output_path}")
            
            # Save the PDF content to a file
            with open(output_path, "wb") as f:
                f.write(pdf_content)
            
            self.logger.info(f"[PDF] PDF saved successfully: {output_path}")
            log_section_boundary(f"Download Filing As PDF - CIK:{cik}, Form:{form_type}", False)
            return output_path
        except Exception as e:
            self.logger.error(f"[PDF] Error generating/saving PDF: {str(e)}")
            log_section_boundary(f"Download Filing As PDF - CIK:{cik}, Form:{form_type}", False)
            raise
    
    def extract_parameters(self, query: str) -> Dict:
        """Extract filing parameters from a natural language query.
        
        Args:
            query: User's natural language query about SEC filings
            
        Returns:
            Dictionary with extracted parameters (company, form_type, year)
        """
        self.logger.info(f"Extracting parameters from query: '{query}'")
        
        prompt = f"""
        Extract the following parameters from this query about SEC filings:
        - Company: The company name or ticker symbol mentioned
        - Form Type: The type of SEC form (e.g., 10-K, 10-Q, 8-K, etc.)
        - Year: The year of the filing
        
        If any parameter is missing, set its value to None.
        Return the parameters as a JSON object with keys 'company', 'form_type', and 'year'.
        
        Query: {query}
        """
        
        self.logger.debug(f"Sending parameter extraction prompt to LLM")
        # Get structured response from LLM
        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            self.logger.debug(f"LLM response for parameter extraction: {result}")
            
            # Extract JSON from response if needed
            if '{' in result:
                json_content = result[result.find('{'):result.rfind('}')+1]
                try:
                    params = json.loads(json_content)
                    self.logger.info(f"Extracted parameters: company='{params.get('company', 'None')}', form_type='{params.get('form_type', 'None')}', year='{params.get('year', 'None')}'")
                    return params
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse parameters as JSON from: {json_content}")
                    print("Failed to parse parameters as JSON")
            else:
                self.logger.warning(f"No JSON found in LLM response: {result}")
            
            # Default return if parsing fails
            self.logger.warning("Returning default empty parameters")
            return {"company": None, "form_type": None, "year": None}
        except Exception as e:
            self.logger.error(f"Error during parameter extraction: {str(e)}")
            return {"company": None, "form_type": None, "year": None}
    
    def confirm_understanding(self, params: Dict) -> str:
        """Generate a confirmation message based on extracted parameters.
        
        Args:
            params: Dictionary with extracted parameters
            
        Returns:
            Confirmation message string
        """
        self.logger.info("Generating confirmation message from extracted parameters")
        company = params.get('company', 'unknown company')
        form_type = params.get('form_type', 'unknown form type')
        year = params.get('year', 'unknown year')
        
        # Handle cases where parameters are None
        if not company or company == 'None':
            self.logger.debug("No valid company name, using default")
            company = 'unknown company'
        if not form_type or form_type == 'None':
            self.logger.debug("No valid form type, using default")
            form_type = 'unknown form type'
        if not year or year == 'None':
            self.logger.debug("No valid year, using default")
            year = 'unknown year'
            
        self.logger.info(f"Confirmation parameters: company='{company}', form_type='{form_type}', year='{year}'")
            
        confirmation_templates = [
            f"I'll look for {company}'s {form_type} filing from {year}. Is that correct?",
            f"Just to confirm, you want the {form_type} for {company} from {year}?",
            f"I understand you're looking for {company}'s {form_type} from {year}. Is that right?"
        ]
        
        import random
        confirmation = random.choice(confirmation_templates)
        self.logger.debug(f"Selected confirmation message: '{confirmation}'")
        return confirmation
    
    def process_conversation(self) -> Optional[str]:
        """Have a conversation with the user to gather SEC filing parameters.
        
        Returns:
            Path to downloaded filing PDF or None if download failed
        """
        log_section_boundary("Starting Conversation Session", True)
        
        self.logger.info("[Conversation] Initializing conversation for SEC filing download")
        print("Welcome to the SEC Filing Downloader. I can help you find and download SEC filings.")
        print("What filing would you like to download? (e.g., 'I need Apple's 10-K from 2023')")
        
        # Initialize parameters
        params = {"company": None, "form_type": None, "year": None}
        confirmed = False
        self.logger.info("[Conversation] Parameters initialized, waiting for user input")
        
        while not confirmed:
            # Get user query
            user_input = input("> ")
            self.logger.info(f"[Conversation] User input: '{user_input}'")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                self.logger.info("[Conversation] User requested to exit the conversation")
                print("Goodbye!")
                log_section_boundary("Conversation Session Ended", False)
                return None
            
            # Extract parameters from user query
            if not all(params.values()):
                self.logger.info("[Conversation] Extracting parameters from user query")
                extracted = self.extract_parameters(user_input)
                
                # Update only missing parameters
                self.logger.debug(f"[Conversation] Current parameters: {params}")
                for key, value in extracted.items():
                    if (not params[key] or params[key] == 'None') and value and value != 'None':
                        self.logger.info(f"[Conversation] Setting {key} to '{value}'")
                        params[key] = value
                self.logger.debug(f"[Conversation] Updated parameters: {params}")
            
            # If we still have missing parameters, ask specifically for them
            if not params['company'] or params['company'] == 'None':
                self.logger.info("[Conversation] Company parameter missing, asking user specifically")
                print("What company are you interested in?")
                company = input("> ")
                self.logger.info(f"[Conversation] Company input: '{company}'")
                params['company'] = company
                continue
                
            if not params['form_type'] or params['form_type'] == 'None':
                self.logger.info("[Conversation] Form type parameter missing, asking user specifically")
                print("What type of filing do you need? (e.g., 10-K, 10-Q, 8-K)")
                form_type = input("> ")
                self.logger.info(f"[Conversation] Form type input: '{form_type}'")
                params['form_type'] = form_type
                continue
                
            if not params['year'] or params['year'] == 'None':
                self.logger.info("[Conversation] Year parameter missing, asking user specifically")
                print("For which year do you need this filing?")
                year = input("> ")
                self.logger.info(f"[Conversation] Year input: '{year}'")
                params['year'] = year
                continue
            
            # Confirm understanding
            self.logger.info("[Conversation] Confirming parameters with user")
            confirmation = self.confirm_understanding(params)
            print(confirmation)
            confirm_input = input("(yes/no) > ").lower()
            self.logger.info(f"[Conversation] User confirmation response: '{confirm_input}'")
            
            if confirm_input in ['y', 'yes', 'yeah', 'correct', 'right']:
                self.logger.info("[Conversation] User confirmed parameters are correct")
                confirmed = True
            else:
                self.logger.info("[Conversation] User rejected parameters, resetting conversation")
                print("Let's try again. What filing are you looking for?")
                # Reset parameters that might be wrong
                params = {"company": None, "form_type": None, "year": None}
                self.logger.debug("[Conversation] Parameters reset to None")
        
        # Parameters confirmed, download the filing
        self.logger.info(f"[Conversation] All parameters confirmed, proceeding to download: company='{params['company']}', form_type='{params['form_type']}', year='{params['year']}'")
        print(f"Great! I'll download the {params['form_type']} filing for {params['company']} from {params['year']}")
        
        result = self.download_by_company_name(params['company'], params['form_type'], params['year'])
        
        if result:
            self.logger.info(f"[Conversation] Filing download successful: {result}")
        else:
            self.logger.warning("[Conversation] Filing download failed")
            
        log_section_boundary("Conversation Session Ended", False)
        return result
    
    def download_by_company_name(self, company_name: str, form_type: str, year: str) -> Optional[str]:
        """Download a filing using company name instead of CIK."""
        log_section_boundary(f"Download By Company Name - Company:{company_name}, Form:{form_type}, Year:{year}", True)
        
        self.logger.info(f"Looking up company: '{company_name}'")
        # Look up the company CIK
        company_info = self.company_lookup.find_company_by_name(company_name)
        
        if not company_info:
            self.logger.warning(f"Could not find company information for: '{company_name}'")
            print(f"Could not find company information for: {company_name}")
            log_section_boundary(f"Download By Company Name - Company:{company_name}", False)
            return None
        
        cik = str(company_info['cik_str'])  # Convert CIK to string
        company_title = company_info['title']
        
        self.logger.info(f"Company found: '{company_title}' (CIK: {cik})")
        print(f"Found company: {company_title} (CIK: {cik})")
        
        try:
            self.logger.info(f"Searching for {form_type} filings for {company_title} (CIK: {cik}) in {year}")
            print(f"Searching for {form_type} filings for {company_title} (CIK: {cik}) in {year}...")
            
            # Get the filing information
            filing_data = self.get_sec_filing(cik, form_type, year)
            
            if 'filings' not in filing_data or not filing_data['filings']:
                self.logger.warning(f"No {form_type} filings found for {company_title} (CIK: {cik}) in {year}")
                print(f"No {form_type} filings found for {company_title} (CIK: {cik}) in {year}")
                log_section_boundary(f"Download By Company Name - Company:{company_name}", False)
                return None
            
            filing = filing_data['filings'][0]
            self.logger.info(f"Selected filing: {filing['formType']} filed on {filing['filedAt']}")
            
            # Get the document URL from the filing data
            sec_url = filing.get('linkToFilingDetails')
            self.logger.info(f"Filing URL: {sec_url}")
            
            print(f"Found {form_type} filing dated {filing['filedAt']}")
            print(f"SEC URL: {sec_url}")
            
            # Download the filing as PDF
            self.logger.info("Downloading filing as PDF")
            output_path = self.download_filing_as_pdf(sec_url, cik, form_type, year, filing['filedAt'])
            
            self.logger.info(f"PDF downloaded successfully: {output_path}")
            print(f"Successfully downloaded and converted {form_type} to PDF: {output_path}")
            print("This PDF file should be much more readable than the HTML version.")
            
            log_section_boundary(f"Download By Company Name - Company:{company_name}", False)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error downloading filing: {str(e)}")
            print(f"Error: {str(e)}")
            log_section_boundary(f"Download By Company Name - Company:{company_name}", False)
            return None

def main():
    # Set up logging for main function
    logger = get_logger()
    log_section_boundary("SEC Filing Downloader Started", True)
    
    # Set up argument parser
    logger.info("[Main] Setting up argument parser")
    parser = argparse.ArgumentParser(description="Download SEC filings for a company as PDF")
    parser.add_argument("--company", default="Apple", help="Company name or ticker symbol (default: Apple)")
    parser.add_argument("--form-type", default="10-K", help="SEC Form Type (default: 10-K)")
    parser.add_argument("--year", default="2023", help="Year of the filing (default: 2023)")
    parser.add_argument("--cik", help="Company CIK number (optional, overrides company name lookup)")
    parser.add_argument("--interactive", action="store_true", help="Use conversational mode to gather filing parameters")
    
    # Parse arguments
    args = parser.parse_args()
    logger.info(f"[Main] Arguments parsed: company='{args.company}', form_type='{args.form_type}', year='{args.year}', cik='{args.cik}', interactive={args.interactive}")
    
    # Initialize the downloader
    logger.info("[Main] Initializing SEC Filing Downloader")
    downloader = SecFilingDownloader(SEC_API_KEY)
    
    # Check if we should use interactive mode
    if args.interactive:
        logger.info("[Main] Starting in interactive mode")
        print("Starting conversational SEC filing downloader...")
        result = downloader.process_conversation()
        logger.info(f"[Main] Interactive session completed, result: {result}")
        log_section_boundary("SEC Filing Downloader Completed", False)
        return
    
    # If CIK is provided directly, use the original method
    if args.cik:
        logger.info(f"[Main] Using direct CIK method with CIK={args.cik}")
        try:
            logger.info(f"[Main] Searching for {args.form_type} filings for CIK {args.cik} in {args.year}")
            print(f"Searching for {args.form_type} filings for CIK {args.cik} in {args.year}...")
            
            # Get the filing information
            filing_data = downloader.get_sec_filing(args.cik, args.form_type, args.year)
            
            if 'filings' not in filing_data or not filing_data['filings']:
                logger.warning(f"[Main] No {args.form_type} filings found for CIK {args.cik} in {args.year}")
                print(f"No {args.form_type} filings found for CIK {args.cik} in {args.year}")
                log_section_boundary("SEC Filing Downloader Completed", False)
                return
            
            filing = filing_data['filings'][0]
            logger.info(f"[Main] Found filing: {filing['formType']} filed on {filing['filedAt']}")
            
            # Get the document URL from the filing data
            sec_url = filing.get('linkToFilingDetails')
            logger.info(f"[Main] Filing URL: {sec_url}")
            
            print(f"Found {args.form_type} filing dated {filing['filedAt']}")
            print(f"SEC URL: {sec_url}")
            
            # Download the filing as PDF
            logger.info("[Main] Downloading filing as PDF")
            output_path = downloader.download_filing_as_pdf(sec_url, args.cik, args.form_type, args.year, filing['filedAt'])
            
            logger.info(f"[Main] PDF downloaded successfully: {output_path}")
            print(f"Successfully downloaded and converted {args.form_type} to PDF: {output_path}")
            print("This PDF file should be much more readable than the HTML version.")
            
        except Exception as e:
            logger.error(f"[Main] Error processing filing: {str(e)}")
            print(f"Error: {str(e)}")
        
        log_section_boundary("SEC Filing Downloader Completed", False)
    else:
        # Use the company name to lookup and download
        logger.info(f"[Main] Using company name method with company='{args.company}'")
        result = downloader.download_by_company_name(args.company, args.form_type, args.year)
        logger.info(f"[Main] Company name download completed, result: {result}")
        log_section_boundary("SEC Filing Downloader Completed", False)

if __name__ == "__main__":
    main()
