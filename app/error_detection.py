from typing import List, Dict, Tuple
import os
import pandas as pd
import pandera as pa
from openai import OpenAI
import json
from dotenv import load_dotenv
from app.helper import parse_llm_json_response, parse_natural_language_response
from app.helper import generate_and_run_data_checks
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from .assistant_service import AssistantService
import tempfile

load_dotenv()

# Set up logging
logger = logging.getLogger('data_quality_app')
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = RotatingFileHandler('app.log', maxBytes=10485760, backupCount=5)  # 10MB per file, keep 5 backups
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def detect_data_errors(schema_file: str, data_file: str, llm: bool = True) -> List[Dict]:
    """
    Wrapper function that detects data errors using either LLM or traditional detection.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
        llm: If True, uses LLM detection. If False, uses traditional detection.
    
    Returns:
        List of dictionaries containing error information
    """
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if llm:
        return detect_data_errors_llm(schema_file, data_file)
    else:
        return detect_data_errors_traditional(schema_file, data_file)


def detect_data_errors_llm(schema_file: str, data_file: str) -> List[Dict]:
    """
    Uses LLM to detect errors in data values based on schema rules.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
    
    Returns:
        List of dictionaries containing error information
    """
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Read files
    schema_df = pd.read_excel(schema_file)
    data_df = pd.read_csv(data_file)
    
    # Prepare content for LLM
    schema_json = schema_df.to_json(orient='records')
    data_sample = data_df.to_json(orient='records')
    
    client = OpenAI()
    prompt = f"""Please check the data for any errors or invalid values.
    
    Schema (showing field definitions):
    {schema_json}
    
    Data to validate:
    {data_sample}
    
    Please check:
    1. Mandatory fields have values
    2. Values match their specified lengths
    3. Values contain valid characters for their type
    4. Any other data quality issues
    
    Return your response as a JSON array of error objects with this structure:
    [
        {{
            "field": field_name,
            "row": row_number,
            "value": invalid_value,
            "error": error_description
        }}
    ]
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.choices[0].message.content)

def detect_data_errors_traditional(schema_file: str, data_file: str) -> List[Dict]:
    """
    Detects errors in data values based on schema rules using traditional methods.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
    
    Returns:
        List of dictionaries containing error information
    """
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    schema_df = pd.read_excel(schema_file)
    data_df = pd.read_csv(data_file)
    errors = []
    
    # Create schema dictionary for easy lookup
    schema_dict = {row['Field Name']: row.to_dict() for _, row in schema_df.iterrows()}
    
    # Check each field's values
    for field, info in schema_dict.items():
        data_type = info['Data Type']
        is_mandatory = info['Attribute'] == 'Mandatory'
        
        if data_type.startswith('CHAR'):
            max_length = int(data_type.split('(')[1].rstrip(')'))
            
            # Check for empty values in mandatory fields
            if is_mandatory:
                empty_mask = data_df[field].isna() | (data_df[field].astype(str).str.strip() == '')
                for idx in data_df[empty_mask].index:
                    errors.append({
                        "field": field,
                        "row": idx + 1,  # +1 for human-readable row numbers
                        "value": None,
                        "error": "Mandatory field is empty"
                    })
            
            # Check field lengths
            too_long_mask = data_df[field].astype(str).str.strip().str.len() > max_length
            for idx in data_df[too_long_mask].index:
                errors.append({
                    "field": field,
                    "row": idx + 1,
                    "value": data_df.at[idx, field],
                    "error": f"Value exceeds maximum length of {max_length}"
                })
            
            # Check for invalid characters
            invalid_char_mask = ~data_df[field].astype(str).str.match(r'^[A-Za-z0-9\s\-_.,\'\"@#&+()\/]*$')
            for idx in data_df[invalid_char_mask].index:
                errors.append({
                    "field": field,
                    "row": idx + 1,
                    "value": data_df.at[idx, field],
                    "error": "Contains invalid characters"
                })
    
    return errors

def describe_data_quality_issues(schema_file: str, data_file: str) -> List[str]:
    """
    Uses LLM to describe data quality issues in natural language.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
    
    Returns:
        List of strings describing data quality issues found
    """
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Read files
    schema_df = pd.read_excel(schema_file)
    data_df = pd.read_csv(data_file)
    
    # Prepare content for LLM
    schema_json = schema_df.to_json(orient='records')
    data_sample = data_df.head().to_json(orient='records')
    
    client = OpenAI()
    prompt = f"""Please analyze this dataset and describe any data quality issues you find.

    Schema (showing field definitions):
    {schema_json}
    
    Data to analyze:
    {data_sample}
    
    Please check for and describe issues such as:
    1. Missing values in mandatory fields
    2. Values that exceed specified lengths
    3. Invalid characters or formats
    4. Inconsistent data patterns
    5. Outliers or suspicious values
    6. Any other data quality concerns
    
    Return your response as a JSON array of objects, where each object contains the following fields:
    {{
        "type": "type of issue",
        "description": "one line description of the issue",
        "solution": "one line suggested solution to the issue (this should be a data cleaning solution)"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return parse_llm_json_response(response.choices[0].message.content)

def get_data_quality_report(schema_file: str, data_file: str, use_code_interpreter: bool = False) -> dict:
    """
    Generate a comprehensive data quality report including issues and cleanup options.
    
    Args:
        schema_file (str): Path to the schema file (Excel format)
        data_file (str): Path to the data file (CSV format)
        use_code_interpreter (bool): If True, uses OpenAI Code Interpreter instead of generating code
        
    Returns:
        dict: A report containing detected errors and cleanup options
    """
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"use_code_interpreter: {use_code_interpreter}")
    
    if use_code_interpreter:
        # Use Code Interpreter version
        errors = detect_data_errors_with_code_interpreter(
            schema_file=schema_file,
            data_file=data_file
        )
        
        # Transform errors into API response format
        api_response = {
            "errors": [],
            "cleanupOptions": []
        }
        
        for error in errors["errors"]:
            api_response["errors"].append({
                "type": error["type"],
                "count": error["count"],
                "rows": error["rows"],
                "suggested_fix": error["suggested_fix"],
                "description": error["description"]
            })
            
            api_response["cleanupOptions"].append({
                "id": f"cleanup_{len(api_response['cleanupOptions']) + 1}",
                "description": error["suggested_fix"]
            })
            
        return api_response
    else:
        # Use existing code generation version
        # Get the natural language description of issues using existing LLM function
        issues = describe_data_quality_issues(
            schema_file=schema_file,
            data_file=data_file
        )
        
        '''# Generate cleanup options based on the detected issues
        cleanup_options = generate_cleanup_options(
            schema_file=schema_file,
            data_file=data_file,
            issues=issues
        )
        
        # Run the checks to find problematic rows
        results = generate_and_run_data_checks(
            issues=issues,
            data_file=data_file
        )'''
        
        # Transform results into the API response format
        api_response = {
            "errors": [],
            "cleanupOptions": []
        }
        
        '''for issue, problem_rows in results.items():
            if isinstance(problem_rows, str) and problem_rows.startswith("Error"):
                continue
            
            error_count = len(problem_rows) if isinstance(problem_rows, pd.DataFrame) else 0
            
            # Get the full issue description from the original issues list
            matching_issue = next(
                (i for i in issues if i["issue"].startswith(issue)),
                None
            )
            
            if matching_issue:
                api_response["errors"].append({
                    "type": issue,
                    "count": error_count,
                    "description": matching_issue["issue"]
                })'''
        
        for i in range(len(issues["errors"])):
            api_response["errors"].append({
                "id": i,
                "type": issues["errors"][i]["type"],
                "description": issues["errors"][i]["description"]
            })
            api_response["cleanupOptions"].append({
                "id": i,
                "description": issues["errors"][i]["solution"]
            })
        
        return api_response

def generate_cleanup_options(schema_file: str, data_file: str, issues: list) -> list:
    """
    Generate cleanup options based on detected issues.
    
    Args:
        schema_file: Path to the schema file
        data_file: Path to the data file
        issues: List of detected issues
        
    Returns:
        List of cleanup options
    """
    # Read files
    schema_df = pd.read_excel(schema_file)
    data_df = pd.read_csv(data_file)
    
    # Prepare content for LLM
    schema_json = schema_df.to_json(orient='records')
    data_sample = data_df.head(20).to_json(orient='records')
    issues_str = "\n".join([f"- {issue['issue']}" for issue in issues])
    
    client = OpenAI()
    prompt = f"""Based on the following data quality issues found in the dataset:

{issues_str}

Schema (showing field definitions):
{schema_json}

Data sample:
{data_sample}

Generate a list of possible cleanup operations that could fix these issues.
Each cleanup option should be independent and actionable.

Return your response as a JSON array of cleanup options with this structure:
[
    {{
        "id": "cleanup_1",
        "description": "Clear description of what the cleanup will do",
    }}
]

Guidelines:
1. Each option should be specific and focused on one type of cleanup
2. Include both simple fixes and more complex transformations
3. Consider data type constraints from the schema
4. Prioritize non-destructive operations where possible
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return parse_llm_json_response(response.choices[0].message.content)

def detect_data_errors_with_code_interpreter(schema_file: str, data_file: str) -> List[Dict]:
    """Uses OpenAI Code Interpreter to detect errors in data values."""
    logger.info(f"Starting error detection with Code Interpreter for files: {schema_file}, {data_file}")
    
    assistant_service = AssistantService()
    
    try:
        # Create assistant with files
        assistant, file_ids = assistant_service.create_assistant_with_files(
            name="Data Validator",
            instructions="""You are a data validation assistant. Analyze the provided data file to:
            1. Check for data quality issues
            2. Identify problematic rows
            3. Suggest fixes for each issue
            
            After analysis, create a JSON file named 'validation_results.json' with this structure:
            {
                "errors": [
                    {
                        "type": "description of the issue",
                        "count": number of rows with this issue,
                        "rows": list of row indices with this issue,
                        "description": "detailed description of the issue",
                        "suggested_fix": "description of how to fix this issue"
                    }
                ]
            }
            
            Make sure to save and attach the JSON file when you're done.""",
            files=[data_file]
        )
        
        # Run conversation
        response = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message="""Please analyze the data file and identify any data quality issues.
            For each issue:
            1. Describe the problem
            2. Count how many rows are affected
            3. List the row indices (0-based) where the issue occurs
            4. Suggest how to fix it
            
            Use Python to analyze the file and save your findings as 'validation_results.json'.
            Make sure to create and attach the JSON file."""
        )
        
        if response["file_id"]:
            # Download and parse the JSON file
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                assistant_service.download_file(response["file_id"], temp_file.name)
                with open(temp_file.name, 'r') as f:
                    results = json.load(f)
                os.unlink(temp_file.name)
        else:
            logger.error("Assistant did not provide results file")
            raise Exception("Assistant did not generate results file")
        
        # Clean up resources
        assistant_service.cleanup_resources(assistant.id, file_ids)
        
        return results
            
    except Exception as e:
        logger.error(f"Error in detect_data_errors_with_code_interpreter: {str(e)}", exc_info=True)
        raise

def cleanup_data_with_code_interpreter(schema_file: str, data_file: str, cleanup_operations: List[Dict], thread_id: str = None) -> Tuple[str, List[str]]:
    """
    Uses OpenAI Code Interpreter to clean data based on selected operations.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
        cleanup_operations: List of cleanup operations to perform
        thread_id: Optional ID of existing thread from validation
        
    Returns:
        Tuple containing:
            - Path to cleaned file
            - List of cleanup operation descriptions
    """
    logger.info(f"Starting cleanup with Code Interpreter for file: {data_file}")
    
    assistant_service = AssistantService()
    
    try:
        # Create assistant with files
        assistant, file_ids = assistant_service.create_assistant_with_files(
            name="Data Cleaner",
            instructions="""You are a data cleaning assistant. Apply the requested cleanup operations to the data file.
            For each operation:
            1. Apply the changes
            2. Save the cleaned file
            
            Make sure to save and attach the final cleaned file.""",
            files=[data_file]
        )
        
        # Format cleanup operations for the message
        operations_text = "\n".join([f"{i+1}. {op['description']}" for i, op in enumerate(cleanup_operations)])
        
        # Run conversation
        response = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message=f"""Please apply the following cleanup operations to the data file:

{operations_text}

Make sure to save and attach the cleaned file when you're done.""",
            thread_id=thread_id
        )
        
        if response["file_id"]:
            # Download and save the cleaned file
            cleaned_file_path = Path(data_file).parent / f"cleaned_{Path(data_file).name}"
            assistant_service.download_file(response["file_id"], str(cleaned_file_path))
            logger.info(f"Successfully saved cleaned file to: {cleaned_file_path}")
        else:
            logger.error("Assistant did not provide cleaned file")
            raise Exception("Assistant did not generate a cleaned file")
        
        # Clean up resources
        assistant_service.cleanup_resources(assistant.id, file_ids)
        
        # Return the file path and original cleanup descriptions
        return str(cleaned_file_path), [{"cleanup_id": op["id"], "changes": op["description"]} for op in cleanup_operations]
            
    except Exception as e:
        logger.error(f"Error in cleanup_data_with_code_interpreter: {str(e)}", exc_info=True)
        raise