#!/usr/bin/env python3
"""
SEC Chatbot - Proof of Concept
A conversational interface for querying SEC filings data
"""

import logging
import os
from datetime import datetime

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'sec_chatbot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('sec_chatbot')

import os
import json
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

import requests
import numpy as np
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Use HuggingFace embeddings to avoid rate limits
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# Import FAISS for vector similarity search
try:
    import faiss
except ImportError:
    print("FAISS not available. Please install it for vector search capabilities.")

# Try to import sentence_transformers which is needed for HuggingFace embeddings
try:
    import sentence_transformers
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    print("sentence_transformers not available. Will fall back to OpenAI embeddings.")
    HUGGINGFACE_AVAILABLE = False

# Load environment variables
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# SEC API endpoints
SEC_QUERY_API = "https://api.sec-api.io"
SEC_XBRL_API = "https://api.sec-api.io/xbrl-to-json"
SEC_EXTRACTOR_API = "https://api.sec-api.io/extractor"

# Section mapping for 10-K filings with detailed descriptions for agent reference
SEC_10K_SECTIONS = {
    "1": {
        "name": "Business", 
        "keywords": ["business", "overview", "operations"],
        "description": "Contains information about the company's business operations, subsidiaries, markets, regulations, labor issues, operating costs, seasonal factors, and insurance matters."
    },
    "1A": {
        "name": "Risk Factors", 
        "keywords": ["risk", "factors", "uncertainties"],
        "description": "Details potential risks, external threats, possible future failures, and other risks disclosed to warn investors about factors that could affect business performance."
    },
    "1B": {
        "name": "Unresolved Staff Comments", 
        "keywords": ["unresolved", "staff", "comments"],
        "description": "Explains any comments received from SEC staff on previously filed reports that have not been resolved after an extended period of time."
    },
    "1C": {
        "name": "Cybersecurity", 
        "keywords": ["cybersecurity", "cyber", "security"],
        "description": "Explains risk management strategies for cybersecurity threats, processes for assessing and managing risks, previous incidents, and board oversight of cybersecurity matters."
    },
    "2": {
        "name": "Properties", 
        "keywords": ["properties", "facilities", "real estate"],
        "description": "Lists significant physical properties and assets of the company, including manufacturing facilities, office buildings, and land holdings."
    },
    "3": {
        "name": "Legal Proceedings", 
        "keywords": ["legal", "proceedings", "litigation"],
        "description": "Discloses significant pending lawsuits or other legal proceedings that could impact the company's financial position or operations."
    },
    "4": {
        "name": "Mine Safety Disclosures", 
        "keywords": ["mine", "safety", "disclosures"],
        "description": "Provides information about mine safety violations or other regulatory matters for companies with mining operations."
    },
    "5": {
        "name": "Market for Registrant's Common Equity", 
        "keywords": ["market", "equity", "stockholder"],
        "description": "Details stock price highs and lows, market information, dividend policies, and issuer purchases of equity securities."
    },
    "6": {
        "name": "Selected Financial Data", 
        "keywords": ["selected", "financial", "data"],
        "description": "Presents consolidated financial data and key metrics over multiple years to show trends in the company's financial performance."
    },
    "7": {
        "name": "MD&A", 
        "keywords": ["management", "discussion", "analysis", "MD&A"],
        "description": "Management's detailed discussion of operations, comparing current period versus prior periods, explaining financial condition, liquidity, capital resources, and often includes sustainability initiatives, environmental programs, and ESG information."
    },
    "7A": {
        "name": "Quantitative and Qualitative Disclosures", 
        "keywords": ["quantitative", "qualitative", "market risk"],
        "description": "Forward-looking assessments of market risks, including sensitivity analyses for interest rates, foreign exchange, commodities, and other risks."
    },
    "8": {
        "name": "Financial Statements", 
        "keywords": ["financial", "statements", "notes", "accounting"],
        "description": "Contains the core financial statements (balance sheet, income statement, cash flows), independent auditor's report, accounting policies, and explanatory notes."
    },
    "9": {
        "name": "Changes in and Disagreements with Accountants", 
        "keywords": ["changes", "disagreements", "accountants"],
        "description": "Discloses any changes in accounting firms and any disagreements with current or former auditors on accounting principles."
    },
    "9A": {
        "name": "Controls and Procedures", 
        "keywords": ["controls", "procedures", "disclosure"],
        "description": "Includes information about the effectiveness of the company's disclosure controls and its internal control over financial reporting."
    },
    "9B": {
        "name": "Other Information", 
        "keywords": ["other", "information"],
        "description": "Contains miscellaneous information not covered elsewhere that must be disclosed to investors."
    },
    "10": {
        "name": "Directors and Executive Officers", 
        "keywords": ["directors", "officers", "governance"],
        "description": "Information about company leadership, board composition, corporate governance practices, and background of key executives."
    },
    "11": {
        "name": "Executive Compensation", 
        "keywords": ["executive", "compensation", "salary"],
        "description": "Details about pay packages, benefits, incentives, and compensation policies for senior executives and board members."
    },
    "12": {
        "name": "Security Ownership", 
        "keywords": ["security", "ownership", "beneficial"],
        "description": "Information about major shareholders, beneficial ownership of securities, and insider holdings of company stock."
    },
    "13": {
        "name": "Related Transactions", 
        "keywords": ["related", "transactions", "independence"],
        "description": "Discloses business dealings between the company and its executives, directors, or major shareholders, and assessments of director independence."
    },
    "14": {
        "name": "Principal Accountant Fees", 
        "keywords": ["accountant", "fees", "services"],
        "description": "Details of fees paid for audit, tax, and consulting services to the company's independent accounting firm."
    },
    "15": {
        "name": "Exhibits and Financial Statement Schedules", 
        "keywords": ["exhibits", "schedules"],
        "description": "Lists additional financial information, required exhibits, and other documents filed as part of the 10-K report."
    }
}

# Confirmation templates
CONFIRMATION_PHRASES = [
    "Just to confirm, you're asking about {company}'s {topic}, correct?",
    "Let me make sure I understand - you want information about {company}'s {topic}?",
    "To clarify, you're looking for details on {topic} from {company}'s {form_type}, right?"
]

class SECChatbot:
    """Main chatbot class for interacting with SEC filings"""
    
    def __init__(self):
        """Initialize the chatbot with necessary components"""
        logger.info("Initializing SECChatbot")
        # Set temperature to 0.0 for factual responses with no creativity
        self.llm = ChatOpenAI(temperature=0.0, model=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Create the conversation chain
        template = """
        You are a helpful assistant that answers questions about SEC filings.
        You always use a warm, conversational tone.
        You must NEVER lie no matter what and only provide information you are certain is accurate.
        
        Current conversation:
        {history}
        Human: {input}
        AI:"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"], 
            template=template
        )
        
        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=True
        )
        logger.info("SECChatbot initialization complete")
    
    def query_sec_api(self, query_string, limit=10):
        """Query the SEC API for filing information"""
        headers = {
            "Authorization": SEC_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query_string,
            "from": "0",
            "size": str(limit),
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        response = requests.post(SEC_QUERY_API, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error querying SEC API: {response.status_code}")
            return None
    
    def extract_section(self, filing_url, item):
        """Extract a specific section from a filing using the Extractor API"""
        params = {
            "token": SEC_API_KEY,
            "url": filing_url,
            "item": item,
            "type": "text"
        }
        
        response = requests.get(SEC_EXTRACTOR_API, params=params)
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error extracting section: {response.status_code}")
            return None
            
    def extract_multiple_sections(self, filing_url, items):
        """Extract multiple sections in parallel using ThreadPoolExecutor
        
        Args:
            filing_url: URL of the filing
            items: List of item numbers to extract
            
        Returns:
            Dictionary mapping item numbers to section content
        """
        if not filing_url or not items:
            return {}
        
        # Define worker function for thread pool
        def extract_worker(item):
            content = self.extract_section(filing_url, item)
            return item, content
        
        # Use ThreadPoolExecutor to extract sections in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(extract_worker, item) for item in items]
            for future in futures:
                item, content = future.result()
                if content:  # Only add if content was found
                    results[item] = content
        
        return results
    
    def get_xbrl_data(self, accession_no):
        """Get structured XBRL data from a filing"""
        params = {
            "token": SEC_API_KEY,
            "accession-no": accession_no
        }
        
        response = requests.get(SEC_XBRL_API, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting XBRL data: {response.status_code}")
            return None
    
    def extract_parameters(self, query):
        """Extract key parameters from user query"""
        prompt = f"""
        Extract the following parameters from this query about SEC filings:
        Query: {query}
        
        Please extract:
        1. Company name (and ticker if present)
        2. Time period or year 
        3. Filing type (10-K, 10-Q, etc.)
        4. Information requested (specific data point, section, or topic)
        
        Format your response as a JSON object with the keys: company, timeframe, form_type, and info_type.
        If a parameter is not specified, use null as the value.
        """
        
        # For ChatOpenAI, we need to use a HumanMessage
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages).content
        
        try:
            # Extract just the JSON part from the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
            
            params = json.loads(json_str)
            return params
        except (json.JSONDecodeError, ValueError):
            print("Error parsing parameters")
            # Return default structure if parsing fails
            return {
                "company": None,
                "timeframe": None,
                "form_type": None,
                "info_type": None
            }
    
    def confirm_understanding(self, params):
        """Generate a confirmation message based on extracted parameters"""
        import random
        
        company = params.get("company", "the company")
        topic = params.get("info_type", "the requested information")
        form_type = params.get("form_type", "filing")
        
        template = random.choice(CONFIRMATION_PHRASES)
        return template.format(company=company, topic=topic, form_type=form_type)
    
    def generate_response(self, query, context_data):
        """Generate a response based on the query and retrieved context
        
        Uses dynamic context handling to create appropriate prompts based on query complexity
        and the type of information being requested. The method adapts to different query types
        with specific processing instructions.
        """
        logger.info(f"Generating response for query: {query}")
        logger.info(f"Context data size: {len(context_data)} characters")
        # Limit context data to avoid token limits in testing
        if len(context_data) > 12000:
            context_data = context_data[:5000] + "\n[...content truncated...]\n" + context_data[-5000:]
        
        # Detect query topic/category for specialized handling
        query_lower = query.lower()
        
        # Identify query categories for specialized processing
        is_financial_query = any(word in query_lower for word in ["financial", "revenue", "profit", "earnings", "income", "balance", "statement"])
        is_risk_query = any(word in query_lower for word in ["risk", "threat", "uncertainty", "litigation", "legal"])
        is_sustainability_query = any(word in query_lower for word in ["sustainability", "environment", "esg", "climate", "emission"])
        is_governance_query = any(word in query_lower for word in ["governance", "board", "executive", "management", "compensation"])
        
        # Base prompt template that applies to all query types
        base_prompt = f"""
        Based on the following information from SEC filings, answer this question:
        
        Question: {query}
        
        Filing Information:
        {context_data}
        
        You are a financial expert answering questions about SEC filings.
        All answers must be truthful, accurate, and based ONLY on the data received from the SEC API.
        NEVER misrepresent or distort the SEC filing data in any way.
        If the API data does not fully answer the question, explicitly state that.
        If you don't have the information to answer the question completely, say so clearly.
        NEVER lie about or alter the SEC filing information and NEVER make up facts not present in the retrieved API data.
        
        IMPORTANT REQUIREMENTS FOR YOUR RESPONSE:
        1. ALWAYS begin your response by stating the filing date (e.g., "Based on the 10-K filed on [date]...")
        2. Include specific filing information, including form type and period covered when available
        3. Include specific numbers, dates, and facts from the filing when relevant
        4. Be clear when information is from a specific year or time period
        5. Use section references when information comes from specific sections of the filing
        """
        
        # Add specialized instructions based on query type
        specialized_instructions = ""
        
        if is_financial_query:
            specialized_instructions = """
            FINANCIAL INFORMATION GUIDELINES:
            - Extract and present specific financial metrics, figures, and trends
            - Compare current values with previous periods when available
            - Note any significant changes or trends in financial performance
            - Present data in a structured format when appropriate
            - Include any footnotes or qualifications mentioned in the filing
            """
        
        elif is_risk_query:
            specialized_instructions = """
            RISK FACTOR GUIDELINES:
            - Identify key risks mentioned in the filing
            - Note any risk prioritization or categorization
            - Include management's assessment of risk likelihood and impact when available
            - Connect risks to business operations and performance
            - Note any mitigation strategies mentioned
            """
        
        elif is_sustainability_query:
            specialized_instructions = """
            SUSTAINABILITY INFORMATION GUIDELINES:
            - Identify any ESG initiatives, goals, or metrics mentioned
            - Extract sustainability commitments, targets, and timeframes
            - Note environmental impact data, carbon emissions, and reduction efforts
            - Include references to sustainability frameworks or standards
            - The information may be embedded within other sections like Risk Factors or MD&A
            """
        
        elif is_governance_query:
            specialized_instructions = """
            GOVERNANCE INFORMATION GUIDELINES:
            - Extract information about board composition and committees
            - Note executive compensation structures and amounts
            - Include information about corporate governance policies
            - Reference any shareholder rights or voting structures
            - Extract information about leadership changes or succession planning
            """
        
        # Add general extraction guidelines for other query types
        else:
            specialized_instructions = """
            GENERAL INFORMATION GUIDELINES:
            - Extract the most relevant information from all provided sections
            - Connect information across sections when it helps answer the question
            - Prioritize recent developments and forward-looking statements
            - Note any limitations in the available information
            """
        
        # Combine base prompt with specialized instructions
        prompt = base_prompt + "\n" + specialized_instructions + """
        
        Provide a conversational, helpful response that directly answers the question.
        Focus on the most relevant information for this specific query.
        """
        
        # Use ChatOpenAI format with HumanMessage
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages).content
        
        # Log the final generated response
        logger.info(f"Generated response: {response}")
        
        return response
    
    def process_query(self, query, skip_confirmation=False):
        """Main method to process a user query"""
        # Log the incoming query
        logger.info(f"Processing query: {query}")
        
        # Extract parameters
        params = self.extract_parameters(query)
        print(f"Extracted parameters: {params}")
        logger.info(f"Extracted parameters: {params}")
        
        # Confirm understanding
        confirmation = self.confirm_understanding(params)
        print(f"Confirmation: {confirmation}")
        
        # Wait for user confirmation (simplified for POC)
        # Skip confirmation if requested (for automated testing)
        if skip_confirmation:
            confirm = "yes"
        else:
            confirm = input("Is this correct? (yes/no): ").lower().strip()
            
        # Accept various forms of positive confirmation
        if confirm in ["yes", "y", "yeah", "yep", "correct", "right", "sure", ""]:
            # Continue with the query
            pass
        else:
            return "I'm sorry, let me try again. Could you rephrase your question?"
        
        # Query SEC API based on parameters
        company = params.get("company")
        # Always default to 10-K if form_type is None or not provided
        form_type = params.get("form_type") 
        if not form_type or form_type.lower() == "none":
            form_type = "10-K"
            
        timeframe = params.get("timeframe")
        
        query_string = ""
        if company:
            # Try to use ticker if provided, otherwise use company name
            if "(" in company and ")" in company:
                ticker = company.split("(")[1].split(")")[0]
                query_string += f"ticker:{ticker} AND "
            else:
                        # Special handling for known company name changes/rebrands
                if company.lower() == "facebook":
                    # Use ticker directly - most reliable approach for Meta
                    query_string += f"ticker:META AND "
                elif company.lower() == "google" or "google" in company.lower():
                    # Google is now under Alphabet - use a more flexible approach to find filings
                    # Use an OR condition to find filings with either Google or Alphabet
                    query_string += f"(companyName:\"Google\" OR companyName:\"Alphabet\" OR ticker:GOOGL) AND "
                else:
                    query_string += f"companyName:\"{company}\" AND "
        
        query_string += f"formType:\"{form_type}\""
        
        # Track if we're looking for the latest filing to limit results
        is_latest_query = False
        
        if timeframe:
            # Handle different timeframe formats
            if timeframe.lower() in ["last", "latest", "recent", "most recent"]:
                # For 'last' or 'latest', we'll limit results to 1 later
                is_latest_query = True
            elif "-" in timeframe:
                # Handle year range with hyphen (YYYY-YYYY)
                start, end = timeframe.split("-")
                query_string += f" AND filedAt:[{start}-01-01 TO {end}-12-31]"
            elif " to " in timeframe.lower():
                # Handle "YYYY to YYYY" format
                # Extract years using digit pattern
                import re
                years = re.findall(r'\b(20\d{2})\b', timeframe)
                if len(years) >= 2:
                    start_year, end_year = years[0], years[1]
                    query_string += f" AND filedAt:[{start_year}-01-01 TO {end_year}-12-31]"
            elif timeframe.isdigit() and len(timeframe) == 4:
                # Handle specific year
                query_string += f" AND filedAt:[{timeframe}-01-01 TO {timeframe}-12-31]"
            else:
                # Try to extract year if it's in a format like "2023" or "FY2023"
                import re
                year_match = re.search(r'\b(20\d{2})\b', timeframe)
                if year_match:
                    year = year_match.group(1)
                    query_string += f" AND filedAt:[{year}-01-01 TO {year}-12-31]"
        
        print(f"SEC API Query: {query_string}")
        
        # If looking for latest, limit to just 1 result
        result_limit = 1 if is_latest_query else 10
        results = self.query_sec_api(query_string, limit=result_limit)
        
        if not results or results.get("total", 0) == 0:
            error_message = f"Filing date: Not available\n\nI couldn't find any {form_type} filings for {company} in the specified timeframe."  
            return error_message
        
        # Get the first filing
        filings = results.get("filings", [])
        if not filings:
            error_message = f"Filing date: Not available\n\nI couldn't find any {form_type} filings for {company} in the specified timeframe."  
            return error_message
            
        filing = filings[0]
        
        # Extract relevant section based on info_type
        info_type = params.get("info_type")
        context_data = ""
        
        # Get document URL from filing
        filing_url = None
        if "linkToFilingDetails" in filing:
            filing_url = filing["linkToFilingDetails"]
        elif "linkToHtml" in filing:
            filing_url = filing["linkToHtml"]
        
        filing_details = self._get_filing_details(filing, company)
        
        # NEW APPROACH: Extract the full document and chunk it
        if filing_url:
            logger.info(f"Using whole-document chunking approach for query: {info_type}")
            
            # Extract the full document using any method available
            document_text = self._extract_document_content(filing_url, filing, info_type)
            
            if document_text:
                # Split into chunks for processing
                chunks = self._chunk_document(document_text)
                
                if chunks:
                    # Find the most relevant chunks using vector similarity
                    # Increase number of chunks for more comprehensive context
                    relevant_chunks = self._find_relevant_chunks(info_type, chunks, top_k=5)
                    
                    if relevant_chunks:
                        # Combine the relevant chunks with filing metadata
                        context_data = filing_details + "\n\n"
                        
                        # Add the top relevant chunks to the context
                        logger.info(f"Building context with {len(relevant_chunks)} relevant chunks (using top {min(5, len(relevant_chunks))})")
                        for i, (chunk, score) in enumerate(relevant_chunks):
                            # Add chunk with relevance score
                            context_data += f"--- Relevant Document Section (Relevance: {score:.2f}) ---\n\n"
                            context_data += chunk
                            context_data += "\n\n"
                            
                            # Log chunk details for debugging
                            preview = chunk[:150].replace('\n', ' ')
                            preview += '...' if len(chunk) > 150 else ''
                            logger.info(f"Added chunk {i+1} to context (score: {score:.2f}): {preview}")
                            
                            # Use more chunks for better context, but limit to avoid overflow
                            if i >= 4:  # Increased from 2 to 4
                                break
                        logger.info(f"Final context data size: {len(context_data)} characters")
        
        # If we didn't get context data from section extraction, try XBRL for financial data
        if not context_data and info_type and "financ" in info_type.lower() and "accessionNo" in filing:
            xbrl_data = self.get_xbrl_data(filing.get("accessionNo"))
            if xbrl_data:
                # For POC, just use the basic company facts
                context_data = json.dumps(xbrl_data.get("facts", {}), indent=2)
        
        # Generate response
        if context_data:
            return self.generate_response(query, context_data)
        else:
            # Return error message if no context data found
            return self._format_error_message(filing, company, form_type)
            
    def _get_filing_details(self, filing, company):
        """Extract and format filing metadata for inclusion in context"""
        # Include filing details in context
        filing_details = f"Filing: {filing.get('formType', 'Unknown')} for {filing.get('companyName', company)} "
        if filing.get('ticker'):
            filing_details += f"({filing.get('ticker')}) "
        filing_details += "\n"
        
        if filing.get('filedAt'):
            # Format date as YYYY-MM-DD for readability
            filed_date = filing.get('filedAt')
            try:
                # Extract just the date portion, removing time and timezone
                filed_date = filed_date.split('T')[0]
            except:
                # Keep original if parsing fails
                pass
            filing_details += f"Filed on: {filed_date}\n"
            
        if filing.get('periodOfReport'):
            filing_details += f"Period: {filing.get('periodOfReport')}\n"
        
        return filing_details
    
    def _format_error_message(self, filing, company, form_type):
        """Format an error message when section extraction fails"""
        # Even in error cases, include filing date information
        filing_date = "unknown date"
        if filing and filing.get('filedAt'):
            try:
                filing_date = filing.get('filedAt').split('T')[0]  # Format as YYYY-MM-DD
            except:
                filing_date = filing.get('filedAt')
                
        return f"I found a {form_type} filing for {company} (filed on {filing_date}), but couldn't extract the specific information you requested. This may be due to the information not being available in the expected format or section."
    
    def _determine_section_item(self, info_type):
        """Determine which 10-K section(s) to look for information in.
        
        Uses a fully dynamic, vector-based approach to match the query
        against section descriptions using semantic similarity.
        
        Args:
            info_type: The query or information type to match against sections
            
        Returns:
            List of section items (strings) to search for relevant information
        """
        logger.info(f"Determining relevant sections for: {info_type}")
        if not info_type:
            return ["8"]  # Default to financial statements
        
        # For very short queries or single terms, use a broad search across key sections
        # This prevents overly narrow matches for simple queries
        if len(info_type.split()) <= 2:
            return ["1", "1A", "7", "8", "9"]  # Key sections covering most information
        
        # Generate section descriptions for embedding comparison
        section_descriptions = []
        section_items = []
        
        for item, section_info in SEC_10K_SECTIONS.items():
            # Create rich context for each section by combining name, description and keywords
            description = f"Section {item}: {section_info['name']}. "
            if section_info.get('description'):
                description += section_info['description']
            
            # Add keywords to improve matching
            if section_info.get('keywords'):
                keywords = ", ".join(section_info['keywords'])
                description += f" Keywords: {keywords}."
                
            section_descriptions.append(description)
            section_items.append(item)
        
        # Use embeddings to find the most similar sections
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Get embeddings
        query_embedding = embeddings.embed_query(info_type)
        section_embeddings = embeddings.embed_documents(section_descriptions)
        
        # Convert to numpy arrays for similarity calculation
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        section_embeddings_np = np.array(section_embeddings, dtype=np.float32)
        
        # Create FAISS index for similarity search
        dimension = len(query_embedding)
        index = faiss.IndexFlatL2(dimension)
        index.add(section_embeddings_np)
        
        # Find top k similar sections
        # For complex or unclear queries, get more sections to ensure comprehensive coverage
        k = min(5, len(section_items))  # Get top 5 or all if less than 5
        logger.info(f"Searching for top {k} relevant sections using vector similarity")
        distances, indices = index.search(query_embedding_np, k)
        
        # Extract the matched section items
        matched_sections = [section_items[idx] for idx in indices[0]]
        
        section_names = [SEC_10K_SECTIONS.get(item, {}).get("name", f"Section {item}") for item in matched_sections]
        logger.info(f"Vector search found sections: {list(zip(matched_sections, section_names))}")
        
        return matched_sections
        
    def _find_relevant_sections(self, query, section_contents):
        """Find the most relevant sections based on vector similarity
        
        Args:
            query: The user's query or information type
            section_contents: Dictionary mapping section items to their text content
            
        Returns:
            List of tuples (item, relevance_score, content) sorted by relevance
        """
        if not query or not section_contents:
            return []
            
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Create query embedding
        query_embedding = embeddings.embed_query(query)
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        # Create section embeddings
        section_items = []
        section_texts = []
        
        for item, content in section_contents.items():
            # Skip empty content
            if not content or len(content.strip()) < 10:
                logger.debug(f"Skipping section {item} due to insufficient content")
                continue
                
            # Add a description prefix to help with context
            section_name = SEC_10K_SECTIONS.get(item, {}).get("name", f"Section {item}")
            # Enrich the content with section metadata for better contextual matching
            enriched_content = f"Section {item}: {section_name}\n\n{content}"
            section_texts.append(enriched_content)
            section_items.append(item)
            logger.debug(f"Added section {item} with {len(content)} characters")
        
        if not section_texts:
            logger.warning("No valid section content found for vector search")
            return []
            
        # Get embeddings for all sections
        logger.info(f"Generating embeddings for {len(section_texts)} sections")
        section_embeddings = embeddings.embed_documents(section_texts)
        
        # Convert to numpy array and create FAISS index
        section_embeddings_np = np.array(section_embeddings, dtype=np.float32)
        dimension = len(section_embeddings_np[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(section_embeddings_np)
        
        # Search in the index
        k = min(len(section_texts), 4)  # Get top k results
        logger.info(f"Searching for top {k} most relevant section contents")
        distances, indices = index.search(query_embedding_np, k)
        
        # Prepare results as (item, relevance_score, content)
        results = []
        for i, idx in enumerate(indices[0]):
            # Convert distance to similarity score (1.0 is most similar)
            relevance_score = 1.0 - (distances[0][i] / 100.0)  # Normalize to [0,1] range
            relevance_score = max(0.0, min(1.0, relevance_score))  # Clamp to [0,1]
            
            item = section_items[idx]
            content = section_texts[idx]
            results.append((item, relevance_score, content))
        
        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def start_conversation(self):
        """Start the conversation loop"""
        print("Welcome to the SEC Chatbot!")
        print("I can answer questions about SEC filings for publicly traded companies.")
        print("Type 'exit' to end the conversation.\n")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Thank you for using SEC Chatbot. Goodbye!")
                break
            
            response = self.process_query(user_input)
            print(f"\nSEC Chatbot: {response}\n")


    def _extract_document_content(self, filing_url, filing, query_topic):
        """Extract document content using the most appropriate method based on query type
        
        This is a smart extraction method that tries multiple approaches:
        1. Direct document access (when available)
        2. Section-based extraction via SEC API
        3. XBRL data for financial information
        
        Args:
            filing_url: URL to the SEC filing
            filing: Filing metadata dictionary
            query_topic: Topic of the query to guide extraction strategy
            
        Returns:
            String containing the document content most relevant to the query
        """
        logger.info(f"Smart document extraction for query topic: {query_topic}")
        
        # Initialize content container
        full_content = ""
        
        # APPROACH 1: Try direct document access first
        direct_content = self._try_direct_document_access(filing_url)
        if direct_content:
            return direct_content
        
        # APPROACH 2: Try section-based extraction
        # Don't hardcode specific sections - extract ALL major sections
        all_sections = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]
        logger.info(f"Extracting all major sections")
        
        section_contents = self.extract_multiple_sections(filing_url, all_sections)
        
        if section_contents and len(section_contents) > 0:
            # Combine all extracted sections
            for item, content in section_contents.items():
                if content and len(content.strip()) > 0:
                    section_name = SEC_10K_SECTIONS.get(item, {}).get("name", f"Section {item}")
                    full_content += f"\n\nITEM {item}. {section_name.upper()}\n\n"
                    full_content += content
        
        # APPROACH 3: Add XBRL data for financial/accounting queries
        # This is especially important for accounting guidance
        if "accounting" in query_topic.lower() or "financ" in query_topic.lower() or "guidance" in query_topic.lower():
            logger.info(f"Adding XBRL data for financial/accounting query")
            if "accessionNo" in filing:
                try:
                    xbrl_data = self.get_xbrl_data(filing["accessionNo"])
                    if xbrl_data:
                        full_content += "\n\nFINANCIAL STATEMENT DATA (XBRL):\n\n"
                        full_content += json.dumps(xbrl_data, indent=2)
                except Exception as e:
                    logger.warning(f"Could not add XBRL data: {str(e)}")
        
        if full_content:
            logger.info(f"Successfully extracted document content ({len(full_content)} characters)")
            return full_content
        else:
            logger.error("All document extraction methods failed")
            return None
    
    def _try_direct_document_access(self, filing_url):
        """Attempt to directly access a document from SEC.gov
        
        Args:
            filing_url: URL to the SEC filing
            
        Returns:
            Document content if successful, None otherwise
        """
        try:
            # Try text version first
            text_url = filing_url.replace("/ix?doc=/", "/")
            if not text_url.endswith(".htm") and not text_url.endswith(".txt"):
                text_url = text_url + ".txt"
                
            # Use browser-like headers to avoid being blocked
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.sec.gov/"
            }
            
            response = requests.get(text_url, headers=headers)
            
            # If text version failed, try HTML version
            if response.status_code != 200:
                response = requests.get(filing_url, headers=headers)
                
            if response.status_code == 200:
                # Clean the content
                content = response.text
                
                # If HTML, clean it
                if "<html" in content.lower() or "<body" in content.lower():
                    import re
                    content = re.sub('<[^<]+?>', ' ', content)  # Remove HTML tags
                    content = re.sub('&nbsp;', ' ', content)    # Replace non-breaking spaces
                    content = re.sub('\\s+', ' ', content)      # Normalize whitespace
                
                logger.info(f"Successfully extracted document via direct access ({len(content)} characters)")
                return content
            else:
                logger.warning(f"Direct document access failed with code {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"Error in direct document access: {str(e)}")
            return None
    
    def _chunk_document(self, document_text, chunk_size=800, overlap=200):
        """Split document into overlapping chunks for processing
        
        Args:
            document_text: Full text of the document
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of document chunks
        """
        if not document_text:
            return []
            
        # First, try to split by common section delimiters
        chunks = []
        
        # Split by Item X.X patterns first (SEC document structure)
        import re
        
        # Pattern to match common section headers in SEC filings
        # Matches: "Item X", "Item X:", "ITEM X.", "ITEM X -", etc.
        section_pattern = re.compile(r'(^|\n\s*)(ITEM|Item)\s+\d+(\.\d+)?[\s\-\.:]+', re.MULTILINE)
        
        # Find all section matches
        section_matches = list(section_pattern.finditer(document_text))
        
        # If we found sections, use them as chunk boundaries
        if section_matches:
            logger.info(f"Found {len(section_matches)} section headers for chunking")
            
            # Process each section
            for i, match in enumerate(section_matches):
                start_pos = match.start()
                
                # Determine end position (next section or end of document)
                if i < len(section_matches) - 1:
                    end_pos = section_matches[i+1].start()
                else:
                    end_pos = len(document_text)
                
                # Get the section content
                section_content = document_text[start_pos:end_pos].strip()
                
                # Skip very short sections
                if len(section_content) < 50:
                    continue
                    
                # For very large sections, further break them down into overlapping chunks
                if len(section_content) > chunk_size * 1.5:
                    # Get the section header for context
                    header_end = section_content.find('\n')
                    if header_end == -1:
                        header_end = min(100, len(section_content))
                    
                    section_header = section_content[:header_end].strip()
                    
                    # Break the section into chunks
                    pos = header_end
                    while pos < len(section_content):
                        # Include section header with each chunk for context
                        chunk_text = section_header + "\n\n"
                        
                        # Add the chunk content
                        end = min(pos + chunk_size, len(section_content))
                        chunk_text += section_content[pos:end]
                        
                        chunks.append(chunk_text)
                        
                        # Move position with overlap
                        pos += chunk_size - overlap
                        if pos >= len(section_content):
                            break
                else:
                    # Add the entire section as one chunk
                    chunks.append(section_content)
        else:
            # No section headers found, fall back to simple chunking
            logger.info("No section headers found, using simple chunking")
            pos = 0
            while pos < len(document_text):
                end = min(pos + chunk_size, len(document_text))
                
                # Try to find a paragraph break to make cleaner chunks
                if end < len(document_text):
                    # Look for paragraph break within a window
                    window_size = min(100, end - pos)
                    paragraph_break = document_text[end-window_size:end].rfind('\n\n')
                    if paragraph_break != -1:
                        end = end - window_size + paragraph_break
                
                chunks.append(document_text[pos:end])
                
                # Move position with overlap
                pos += (end - pos) - overlap
                
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _find_relevant_chunks(self, query, chunks, top_k=3):
        """Find the most relevant chunks using vector similarity
        
        Args:
            query: The user's query
            chunks: List of document chunks
            top_k: Number of top chunks to return
            
        Returns:
            List of tuples (chunk_text, relevance_score) sorted by relevance
        """
        if not query or not chunks:
            return []
            
        logger.info(f"Finding most relevant chunks for query: {query}")
        
        # Limit the number of chunks to process to avoid rate limits
        # If there are too many chunks, sample a reasonable number
        max_chunks_to_process = 500  # Set a reasonable limit
        if len(chunks) > max_chunks_to_process:
            logger.warning(f"Too many chunks ({len(chunks)}), sampling down to {max_chunks_to_process}")
            
            # Prioritize chunks that might have keywords related to the query
            # This is a simple keyword matching to pre-filter chunks
            query_keywords = query.lower().split()
            scored_chunks = []
            
            for i, chunk in enumerate(chunks):
                # Simple keyword matching score
                keyword_score = sum(1 for keyword in query_keywords if keyword in chunk.lower())
                scored_chunks.append((i, keyword_score, chunk))
            
            # Sort by keyword score and take top chunks, plus some random ones for diversity
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            top_by_keywords = [chunks[idx] for idx, _, _ in scored_chunks[:max_chunks_to_process//2]]
            
            # Add some random chunks for diversity (in case keyword matching isn't sufficient)
            import random
            random_indices = random.sample(range(len(chunks)), min(max_chunks_to_process//2, len(chunks) - max_chunks_to_process//2))
            random_chunks = [chunks[i] for i in random_indices if chunks[i] not in top_by_keywords]
            
            # Combine and ensure we're within limit
            chunks_to_process = top_by_keywords + random_chunks
            if len(chunks_to_process) > max_chunks_to_process:
                chunks_to_process = chunks_to_process[:max_chunks_to_process]
        else:
            chunks_to_process = chunks
        
        logger.info(f"Processing {len(chunks_to_process)} chunks for embedding")
        
        # Use HuggingFace embeddings to avoid OpenAI rate limits
        if HUGGINGFACE_AVAILABLE:
            try:
                # Try using HuggingFace embeddings first (like the example)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                logger.info("Using HuggingFace embeddings model")
            except Exception as e:
                # Fall back to OpenAI if HuggingFace has an error
                logger.warning(f"Error with HuggingFace embeddings: {str(e)}")
                logger.warning("Falling back to OpenAI embeddings")
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        else:
            # If sentence_transformers isn't installed, use OpenAI
            logger.warning("HuggingFace embeddings not available, using OpenAI embeddings")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Create a FAISS vector store from the chunks
        try:
            from langchain.docstore.document import Document
            from langchain.vectorstores import FAISS
            
            # Convert chunks to documents
            documents = [Document(page_content=chunk) for chunk in chunks_to_process]
            
            # Create vectorstore - use try/except to handle potential errors
            try:
                vectorstore = FAISS.from_documents(documents, embeddings)
                logger.info("Successfully created FAISS vectorstore")
                
                # Create a retriever with search_kwargs
                retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
                
                # Perform the search
                retrieved_docs = retriever.get_relevant_documents(query)
                logger.info(f"Retrieved {len(retrieved_docs)} documents from FAISS search")
                
                # Log preview of each retrieved document
                for i, doc in enumerate(retrieved_docs):
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    preview = content[:100].replace('\n', ' ') + '...' if len(content) > 100 else content
                    logger.info(f"Retrieved doc {i+1}/{len(retrieved_docs)}: {preview}")
            except Exception as e:
                # If FAISS vectorstore fails, fall back to a simpler keyword-based approach
                logger.warning(f"FAISS vectorstore error: {str(e)}")
                logger.warning("Falling back to keyword-based search")
                retrieved_docs = self._fallback_keyword_search(query, chunks_to_process, top_k)
        except Exception as e:
            logger.error(f"Error in vector search setup: {str(e)}")
            # Fallback to a simpler approach if the imports or Document creation fails
            retrieved_docs = self._fallback_keyword_search(query, chunks_to_process, top_k)
        
        # Create the result tuples with relevance scores
        matched_chunks = []
        for i, doc in enumerate(retrieved_docs):
            if hasattr(doc, 'page_content'):  # Handle Document objects
                content = doc.page_content
            else:  # Handle plain text if using fallback
                content = str(doc)
                
            # Calculate a dummy score since FAISS retriever doesn't return scores
            # The earlier in the results, the higher the score
            similarity = 1.0 - (i * 0.1)  # Simple scoring based on position
            matched_chunks.append((content, similarity))
            
            # Log the content of each retrieved chunk with its score
            # Clean up content for logging (remove newlines for readability)
            clean_content = content.replace('\n', ' ')[:200]
            truncated_content = clean_content + '...' if len(content) > 200 else clean_content
            logger.info(f"Retrieved chunk {i+1}/{len(retrieved_docs)} (score: {similarity:.2f}): {truncated_content}")
        
        logger.info(f"Found {len(matched_chunks)} relevant chunks with similarity scores")
        return matched_chunks


    def _fallback_keyword_search(self, query, chunks, top_k=3):
        """Fallback method that uses simple keyword matching when vector search fails
        
        Args:
            query: The user's query
            chunks: List of document chunks
            top_k: Number of top chunks to return
        
        Returns:
            List of most relevant chunks based on keyword matching
        """
        logger.info(f"Using fallback keyword search for query: {query}")
        
        # Break query into keywords
        query_keywords = query.lower().split()
        
        # Score chunks based on keyword occurrence
        chunk_scores = []
        for chunk in chunks:
            score = sum(1 for keyword in query_keywords if keyword in chunk.lower())
            chunk_scores.append((chunk, score))
        
        # Sort by score (highest first) and take top_k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the top_k chunks
        top_chunks = [chunk for chunk, _ in chunk_scores[:top_k]]
        logger.info(f"Fallback search found {len(top_chunks)} chunks")
        
        return top_chunks


if __name__ == "__main__":
    logger.info("Starting SEC Chatbot application")
    logger.info("="*50)
    logger.info("NEW SESSION STARTED")
    logger.info("="*50)
    try:
        chatbot = SECChatbot()
        chatbot.start_conversation()
    except Exception as e:
        logger.error(f"Error in SEC Chatbot: {e}", exc_info=True)
    logger.info("="*50)
    logger.info("SESSION TERMINATED")
    logger.info("="*50)
    logger.info("SEC Chatbot application terminated")
