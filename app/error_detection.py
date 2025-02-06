from typing import List, Dict, Tuple
import os
import pandas as pd
import pandera as pa
from openai import OpenAI
import json
from dotenv import load_dotenv
from app.helper import parse_llm_json_response, parse_error_analysis_csv
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from .assistant_service import AssistantService
import tempfile
import uuid

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

    This is a sample from a larger dataset. You must extrapolate the issues to the entire dataset.
    
    Return your response as a JSON array of objects, where each object contains the following fields:
    {{
        "type": "what the issue is",
        "description": "description of the issue",
        "solution": "one line suggested solution to the issue (this should be a data cleaning solution)"
    }}
    """

    response = client.chat.completions.create(
        model="o1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    json_response = parse_llm_json_response(response.choices[0].message.content)
    print("Returned json_response: ", json_response)
    return json_response


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
    
    use_code_interpreter = True

    print(f"use_code_interpreter: {use_code_interpreter}")
    
    if use_code_interpreter:
        # First get the issues using natural language description
        issues = describe_data_quality_issues(
            schema_file=schema_file,
            data_file=data_file
        )

        logger.info(f"Issues: {issues}")
        
        # Then get detailed row-level analysis
        results_csv = detect_data_errors_with_code_interpreter_detailed(
            data_file=data_file,
            issues=issues
        )

        logger.info(f"Results CSV returned")
        
        # Parse the detailed results
        detailed_results = parse_error_analysis_csv(results_csv)

        logger.info(f"Detailed results: {detailed_results}")

        # Read the original data file to get full row data
        data_df = pd.read_csv(data_file)
        
        # Transform errors into API response format
        api_response = {
            "errors": [],
            "cleanupOptions": []
        }
        
        # Match detailed results with original issues
        for i, issue in enumerate(issues):
            issue_id = f"issue_{i+1}"
            # Find matching detailed result
            detailed_result = next((r for r in detailed_results if r["id"] == issue_id), None)
            affected_rows = detailed_result["rows"] if detailed_result else []

            # Get the full row data for affected rows
            full_rows = []
            for row_idx in affected_rows:
                if 0 <= row_idx < len(data_df):
                    row_data = data_df.iloc[row_idx].to_dict()
                    full_rows.append({
                        "index": row_idx,
                        "data": row_data
                    })
            
            api_response["errors"].append({
                "id": i,
                "type": issue["type"],
                "count": len(affected_rows),
                "rows": full_rows,
                "description": issue["description"]
            })
            
            api_response["cleanupOptions"].append({
                "id": i,
                "description": issue["solution"]
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
        
        for i in range(len(issues)):
            api_response["errors"].append({
                "id": i,
                "type": issues[i]["type"],
                "description": issues[i]["description"]
            })
            api_response["cleanupOptions"].append({
                "id": i,
                "description": issues[i]["solution"]
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
    
    json_response = parse_llm_json_response(response.choices[0].message.content)
    print("Returned json_response: ", json_response)
    return json_response


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
                # Close the file before reading it again
                temp_file.close()
                
                try:
                    with open(temp_file.name, 'r') as f:
                        results = json.load(f)
                finally:
                    # Make sure to close the file before trying to delete it
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        logger.warning(f"Could not delete temporary file {temp_file.name}: {str(e)}")
        else:
            logger.error("Assistant did not provide results file")
            raise Exception("Assistant did not generate results file")
        
        # Clean up resources
        assistant_service.cleanup_resources(assistant.id, file_ids)
        
        return results
            
    except Exception as e:
        logger.error(f"Error in detect_data_errors_with_code_interpreter: {str(e)}", exc_info=True)
        raise

def detect_data_errors_with_code_interpreter_detailed(data_file: str, issues: List[Dict]) -> str:
    """
    Uses OpenAI Code Interpreter to analyze data quality issues and create a detailed CSV report.
    
    Args:
        data_file: Path to the CSV data file
        issues: List of dictionaries containing detected issues with their descriptions
        
    Returns:
        Path to the generated CSV file containing detailed error analysis
    """
    logger.info(f"Starting detailed error detection with Code Interpreter for file: {data_file}")
    
    assistant_service = AssistantService()
    
    try:
        # Create assistant with files
        assistant, file_ids = assistant_service.create_assistant_with_files(
            name="Data Error Analyzer",
            instructions="""You are a data quality analysis assistant. Your task is to:
            1. Read the provided data file
            2. For each issue in the list provided, write code to detect rows that have that issue
            3. Create a CSV file where:
               - Each row corresponds to a row in the original data
               - Each column corresponds to an issue
               - Values are True/False indicating if the issue exists in that row
            4. Save and attach the results CSV file
            
            Make sure to handle edge cases and potential errors gracefully.""",
            files=[data_file]
        )
        
        # Format issues for the message
        issues_text = "\n".join([f"{i+1}. {issue['description']}" for i, issue in enumerate(issues)])
        
        # Run conversation
        response = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message=f"""Please analyze the data file and create a detailed error report CSV.

For each row in the data, check for these issues:

{issues_text}

Create a CSV file with these columns:
1. row_index: The index of the row from the original file
2. One column for each issue, named issue_1, issue_2, etc., containing True/False

Please write Python code to:
1. Read the data file
2. Create functions to detect each issue
3. Apply those functions to each row
4. Create and save the results CSV
5. Make sure to handle potential errors (null values, type mismatches, etc.)

Name the output file 'error_analysis_results.csv'."""
        )
        
        if response["file_id"]:
            # Download and save the results file
            results_file_path = Path(data_file).parent / "error_analysis_results.csv"
            assistant_service.download_file(response["file_id"], str(results_file_path))
            logger.info(f"Successfully saved error analysis results to: {results_file_path}")
        else:
            logger.error("Assistant did not provide results file")
            raise Exception("Assistant did not generate a results file")
        
        # Clean up resources
        assistant_service.cleanup_resources(assistant.id, file_ids)
        
        return str(results_file_path)
            
    except Exception as e:
        logger.error(f"Error in detect_data_errors_with_code_interpreter_detailed: {str(e)}", exc_info=True)
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
            file_uuid = str(uuid.uuid4())
            cleaned_file_path = Path(data_file).parent / f"cleaned_{file_uuid}_{Path(data_file).name}"
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