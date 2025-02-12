from typing import Dict, List
import os
import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
from .assistant_service import AssistantService
from .helper import parse_llm_json_response
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

def validate_data_against_schema(schema_file: str, data_file: str, llm: bool = True, use_chat: bool = True) -> Dict:
    """
    Wrapper function that validates data file against schema using either LLM or traditional validation.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
        llm: If True, uses LLM validation. If False, uses traditional validation.
        use_chat: If True and llm is True, uses Chat Completions API instead of Assistants API
    
    Returns:
        Dict containing validation results
    """
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if llm:
        if use_chat:
            return validate_data_against_schema_chat(schema_file, data_file)
        return validate_data_against_schema_llm(schema_file, data_file)
    else:
        return validate_data_against_schema_traditional(schema_file, data_file)

def validate_data_against_schema_traditional(schema_file: str, data_file: str) -> Dict:
    """
    Validates that data file structure conforms to schema (field names and types).
    Does not validate individual rows.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
    
    Returns:
        Dict containing validation results
    """
    print("Traditional validation")

    schema_df = pd.read_excel(schema_file)
    data_df = pd.read_csv(data_file)
    
    errors = []
    
    # Create schema dictionary for easy lookup
    schema_dict = {row['Field Name']: row.to_dict() for _, row in schema_df.iterrows()}
    
    # Check all schema fields exist in data
    missing_fields = [field for field in schema_dict.keys() 
                     if field not in data_df.columns]
    if missing_fields:
        errors.append(f"Missing fields from schema: {', '.join(missing_fields)}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }

def validate_data_against_schema_llm(schema_file: str, data_file: str) -> Dict:
    """
    Validates data file against schema using GPT-4 LLM.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
    
    Returns:
        Dict containing validation results
    """
    print("LLM validation")
    assistant_service = AssistantService()
    
    try:
        # Create assistant with files
        assistant, file_ids = assistant_service.create_assistant_with_files(
            name="Schema Validator",
            instructions="""You are a schema validation assistant provided with schema and data files with the purported schema structure.
            
            Output should check for the following conditions and nothing else:
            1. Field names match the schema column names

            Guardrail: do not ouput errors of the form "Column 'Customer Type' exceeds maximum length of 4 characters"
            
            
            Return your response as a JSON object with this structure:
            {
                "is_valid": boolean,
                "errors": [list of error messages]
            }
            """,
            files=[schema_file, data_file]
        )
        
        
        # Run conversation
        response = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message="""Please validate the structure of columns in the data file against the schema file. 
            
            No No No. Do not look at row values at this time. 
            Return your response as a JSON object with this structure:
            {
                "is_valid": true,
                "errors": [
    "Required column 'user_id' is missing from the data file",
    "Unexpected column 'nickname' found in the data file",
    "Field order mismatch: Expected 'user_id', 'name', 'email', but found 'name', 'user_id', 'email'",
]
            }"""
        )
        
        # Clean up resources
        assistant_service.cleanup_resources(assistant.id, file_ids)
        
        return json.loads(response["message"])
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise

def validate_data_against_schema_chat(schema_file: str, data_file: str) -> Dict:
    """
    Validates data file against schema using GPT-4 via Chat Completions API.
    Includes file contents directly in the prompt.
    
    Args:
        schema_file: Path to the XLSX schema file
        data_file: Path to the CSV data file
    
    Returns:
        Dict containing validation results
    """
    # Read the files
    schema_df = pd.read_excel(schema_file)
    data_df = pd.read_csv(data_file)
    
    # Convert to string representations
    schema_str = schema_df.to_string()
    data_sample = data_df.head(20).to_string()

    print("Using Chat Completions API")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""I have a data file and a schema file that I need to validate.

Schema file contents:
{schema_str}

First few rows of the data file:
{data_sample}

Please validate the structure of the data file against the schema file. 
Focus only on schema-level validation:
1. Check if all required fields from the schema exist in the data file

Make sure you only return unique errors.  
Do not look at row values at this time. Do not ouput errors of the form "Column 'Customer Type' exceeds maximum length of 4 characters"

Return your response as a JSON object with this structure:
{{
    "is_valid": false,
    "errors": [
        "Column 'customer_id' missing in data file",
    ]
}}"""

    response = client.chat.completions.create(
        model="o1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    print(response.choices[0].message.content)

    try:
        # Use the parse_llm_json_response from helper
        json_response = parse_llm_json_response(response.choices[0].message.content)
        print("Returned json_response: ", json_response)
        return json_response
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        raise