# SEC Filing Downloader

This project provides tools for downloading SEC filings from the EDGAR database. It can retrieve filings in PDF format for better readability.

## Features

- Download SEC filings by company name, CIK, form type, and year
- Convert filings to readable PDF format
- LLM-powered company name lookup (no need to remember CIK numbers)
- Customizable with command-line arguments

## Prerequisites

- Python 3.7+
- SEC API key (sign up at [sec-api.io](https://sec-api.io/))
- OpenAI API key (for company name lookup feature)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
SEC_API_KEY=your_sec_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o  # or another OpenAI model
```

## Usage

### Enhanced SEC Filing Downloader (with Company Name Lookup)

Use the `sec_filing_downloader.py` script to download filings by company name:

```bash
python sec_filing_downloader.py --company "Apple" --form-type "10-K" --year "2023"
```

#### Available Command-Line Arguments

- `--company`: Company name or ticker symbol (default: "Apple")
- `--form-type`: SEC form type (default: "10-K")
- `--year`: Year of the filing (default: "2023")
- `--cik`: Company CIK number (optional, overrides company name lookup)

### Basic SEC Filing Downloader

Use the `download_apple_10k.py` script for a simpler approach:

```bash
python download_apple_10k.py --cik 320193 --form-type "10-K" --year "2023"
```

#### Available Command-Line Arguments

- `--cik`: Company CIK number (default: 320193 for Apple)
- `--form-type`: SEC form type (default: "10-K")
- `--year`: Year of the filing (default: "2023")

## File Management

Downloaded files are saved in the `filings` directory with filenames formatted as:
```
{cik}_{form_type}_{year}_{filing_date}.pdf
```

The script will overwrite existing files if you download the same filing twice (same CIK, form type, year, and filing date).

## Example Usage

1. Download Apple's 2023 10-K filing:
```bash
python sec_filing_downloader.py --company "Apple" --form-type "10-K" --year "2023"
```

2. Download Microsoft's Q1 2023 quarterly report:
```bash
python sec_filing_downloader.py --company "Microsoft" --form-type "10-Q" --year "2023"
```

3. Download Tesla's 2022 annual report using CIK:
```bash
python sec_filing_downloader.py --cik 1318605 --form-type "10-K" --year "2022"
```

## Notes

- The initial downloaded HTML files may not be readable due to XBRL formatting
- The PDF conversion makes the filings much more readable
- Company name lookup uses LLM to find the best match in the SEC company database
- If the script fails to find a match for a company name, try using the ticker symbol or CIK number
