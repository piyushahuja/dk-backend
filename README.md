# TailorFlow

A FastAPI-based data validation service that helps validate and analyze data files against schema definitions.

## Overview

TailorFlow provides a robust API for:

- Validating data files against schema definitions
- Detecting data quality issues and inconsistencies
- Generating comprehensive data quality reports
- Suggesting data cleaning solutions

## Features

- **Schema Validation**: Ensures data files conform to specified schema requirements
- **Error Detection**: Identifies issues like:
  - Missing mandatory fields
  - Invalid data types
  - Length violations
  - Invalid characters
  - Duplicate records
- **Quality Analysis**: Uses AI to provide detailed quality reports and cleaning recommendations
- **Secure File Handling**: Implements secure file upload and storage mechanisms

## Setup

1. Clone the repository:

```bash
git clone https://github.com/piyushahuja/tailorflow-backend.git
cd tailorflow-backend
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

4. Set up environment variables:
   Create a `.env` file with:

```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:

```bash
uvicorn app.main:app --reload
```

## API Endpoints

### POST /upload

Upload a file and receive a file identifier for future operations.

### POST /validate_schema

Validate a data file against a schema file.

### POST /detect_errors

Perform comprehensive error detection and receive a detailed quality report.

## Security

- Implements secure file handling with sanitized filenames
- Restricts upload directory permissions
- Validates file content before processing
- Cleans up temporary files after processing

## Requirements

- Python 3.8+
- FastAPI
- Pandas
- OpenAI API access
- Additional dependencies in requirements.txt
