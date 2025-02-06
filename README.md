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

**Request:**

- Content-Type: multipart/form-data
- Body:
  - `file`: File to upload (required)

**Response:**

```json
{
	"file_id": "string" // UUID for future reference
}
```

### POST /validate_schema

Validate a data file against a schema file.

**Request:**

```json
{
	"schema_file_id": "string", // UUID of uploaded schema file
	"data_file_id": "string" // UUID of data file to validate
}
```

**Response:**

```json
{
    "status": "success",
    "is_valid": boolean,
    "errors": [
        // Array of validation errors if any
    ]
}
```

### POST /detect_errors

Perform comprehensive error detection and receive a detailed quality report.

**Request:**

```json
{
	"schema_file_id": "string", // UUID of uploaded schema file
	"data_file_id": "string" // UUID of data file to analyze
}
```

**Response:**

```json
{
	"status": "success",
	"errors": [
		{
			"type": "string", // Short identifier for the issue type
			"description": "string" // Detailed description of the issue
		}
	],
	"cleanupOptions": [
		{
			"id": "cleanup_1",
			"description": "Description of the cleanup operation"
		}
	]
}
```

### POST /cleanup

Apply multiple cleanup operations to a data file in sequence.

**Request:**

```json
{
	"file_id": "string", // UUID of file to clean
	"cleanup_operations": [
		{
			"id": "string", // Operation identifier
			"description": "string" // Operation description
		}
	]
}
```

**Response:**

```json
{
	"status": "success",
	"new_file_id": "string", // UUID of cleaned file
	"changes_made": [
		// Array of changes applied
	]
}
```

### GET /download/{file_id}

Download a file using its file ID.

**Parameters:**

- `file_id`: UUID of the file to download (path parameter)

**Response:**

- File download response (CSV file)
- Content-Type: text/csv

### POST /detect_error_code

Generate and run code to detect specific errors in a data file using AI assistance.

**Request:**

```json
{
	"file_id": "string", // UUID of the data file to analyze
	"error_description": "string" // Description of the error to detect
}
```

**Response:**

```json
{
	"status": "success",
	"affected_rows": [
		// Array of row indices where the error was found
	],
	"code_description": "string" // Description of the code used to detect the error
}
```

### Error Responses

All endpoints may return the following error responses:

```json
{
	"detail": "string" // Error message
}
```

Common HTTP status codes:

- 400: Bad Request (invalid input)
- 404: File Not Found
- 500: Internal Server Error

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
