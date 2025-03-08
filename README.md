# SEC Filing Downloader

A powerful tool for downloading and converting SEC EDGAR filings to readable PDF format, enhanced with LLM capabilities for intuitive company lookup.

## Features

- **Smart Company Lookup**: Find companies by name, partial name, or ticker symbol using LLM-powered matching
- **Interactive Mode**: Conversational interface to guide you through finding and downloading filings
- **PDF Conversion**: Automatically convert hard-to-read EDGAR HTML filings to readable PDF format
- **Flexible Querying**: Search by company name, CIK, form type, and year
- **Unified Logging System**: Comprehensive logging for tracking and debugging
- **Command Line Interface**: Easy to use with customizable arguments

## Prerequisites

- **Python 3.7+**
- **API Keys**:
  - **SEC API key**: Required for accessing SEC filings (get one at [sec-api.io](https://sec-api.io/))
  - **OpenAI API key**: Required for company name lookup and conversation features

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sec-filing-downloader
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```
   SEC_API_KEY=your_sec_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o  # or another OpenAI model
   ```

## Usage

### Interactive Mode (Recommended)

The interactive mode provides a conversational interface that guides you through finding and downloading SEC filings:

```bash
python sec_filing_downloader.py --interactive
```

Follow the prompts to specify the company, form type, and year for the filing you want.

### Command Line Mode

Download filings directly using command line arguments:

```bash
python sec_filing_downloader.py --company "Apple" --form-type "10-K" --year "2023"
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--company` | Company name or ticker symbol | "Apple" |
| `--form-type` | SEC form type (10-K, 10-Q, 8-K, etc.) | "10-K" |
| `--year` | Year of the filing | "2023" |
| `--cik` | Company CIK number (optional, overrides company lookup) | None |
| `--interactive` | Use conversational mode | False |

## Project Structure

```
sec-filing-downloader/
│
├── sec_filing_downloader.py    # Main application file
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
│
├── utils/                      # Utility modules
│   └── logger.py               # Unified logging system
│
├── reference_data/             # Reference data for company lookup
│   └── company_tickers.json    # SEC company database
│
├── Tests/                      # Directory for unit tests
│
├── example/                    # Example API usage scripts
│   ├── secapi_xbrl_api.py          # Example XBRL API usage
│   ├── secapi_full_text_search_api.py # Example full-text search
│   ├── secapi_query_api.py         # Example query API usage
│   └── secapi_extractor_api.py     # Example extractor API usage
│
├── Docs/                       # API documentation
│   ├── xbrl-to-json-converter-api.txt  # XBRL API documentation
│   ├── full-text-search-api.txt       # Full-text search API documentation
│   └── ... other documentation files
│
├── filings/                    # Downloaded filing PDFs
└── Logs/                       # Application logs
```

## Logging System

The application uses a unified logging system that:

- Creates timestamped log files in the `Logs/` directory
- Logs all application activity including API requests, file operations, and user interactions
- Provides context-specific logging with component prefixes (e.g., `[Downloader]`, `[PDF]`, `[Conversation]`)
- Helps with debugging and tracking application behavior

## Examples

1. **Interactive mode**:
   ```bash
   python sec_filing_downloader.py --interactive
   ```

2. **Download Apple's 2023 10-K filing**:
   ```bash
   python sec_filing_downloader.py --company "Apple" --form-type "10-K" --year "2023"
   ```

3. **Download Microsoft's Q1 2023 quarterly report**:
   ```bash
   python sec_filing_downloader.py --company "Microsoft" --form-type "10-Q" --year "2023"
   ```

4. **Download Tesla's 2022 annual report using CIK**:
   ```bash
   python sec_filing_downloader.py --cik 1318605 --form-type "10-K" --year "2022"
   ```

## Output

Downloaded files are saved in the `filings/` directory with the naming convention:
```
{cik}_{form_type}_{year}_{filing_date}.pdf
```

## Troubleshooting

- **Company not found**: Try using the ticker symbol instead of the company name, or provide the CIK directly
- **No filings found**: Verify the form type and year are correct for the company
- **API errors**: Ensure your API keys are correctly set in the `.env` file
- **Check logs**: Review the logs in the `Logs/` directory for detailed error information

## SEC-API Example Scripts

The project includes example scripts for demonstrating SEC-API functionality. These scripts are located in the `example/` directory and can be run individually to understand different aspects of the SEC-API.

### Available Example Scripts

1. **XBRL API Examples** (`secapi_xbrl_api.py`)
   - Demonstrates the conversion of XBRL data to JSON format
   - Retrieves Apple's 10-K filing and extracts financial data
   - Shows how to access income statement items

2. **Full-Text Search API Examples** (`secapi_full_text_search_api.py`)
   - Demonstrates searching across SEC filings for specific text content
   - Uses CIK parameter to filter for Apple Inc. filings
   - Shows how to parse and display search results

3. **Query API Examples** (`secapi_query_api.py`)
   - Demonstrates finding specific filings by form type, company, and date
   - Shows how to search for 10-K, 10-Q, and 8-K filings
   - Illustrates how to extract metadata from filing results

4. **Extractor API Examples** (`secapi_extractor_api.py`)
   - Demonstrates extracting specific sections from filings
   - Shows how to retrieve risk factors from 10-K filings
   - Illustrates content extraction capabilities

### Running the Example Scripts

To run an example script, use Python to execute the desired file:

```bash
python example/secapi_xbrl_api.py
```

> **Note**: These examples require a valid SEC-API key in your `.env` file.

## License

[MIT License](LICENSE)

## Acknowledgments

- [SEC-API.io](https://sec-api.io/) for providing API access to SEC EDGAR filings
- [OpenAI](https://openai.com/) for the LLM capabilities
