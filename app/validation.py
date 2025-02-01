from typing import Dict, List
import os
import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

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
    client = OpenAI()
    
    # Upload files
    schema_file_obj = client.files.create(
        file=open(schema_file, "rb"),
        purpose="assistants"
    )
    data_file_obj = client.files.create(
        file=open(data_file, "rb"),
        purpose="assistants"
    )

    # Create assistant
    assistant = client.beta.assistants.create(
        name="Data Validator",
        instructions="""You are a data validation assistant. Analyze the provided schema and data files and validate that:
        1. Data types match the schema
        2. Mandatory fields are present
        3. Field lengths match specifications
        4. Any other validation issues
        
        Return your response as a JSON object with this structure:
        {
            "is_valid": boolean,
            "errors": [list of error messages]
        }""",
        model="gpt-4o",
        tools=[{"type": "code_interpreter"}],
        tool_resources={
            "code_interpreter": {
                "file_ids": [schema_file_obj.id, data_file_obj.id]
            }
        }
    )

    # Create thread and message
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="""Please validate the structure of the data file against the schema file. 
        Focus only on schema-level validation:
        1. Check if all required fields from the schema exist in the data file
        2. Verify that the columns have the correct data types as specified in the schema
        3. Validate any field length constraints defined in the schema
        
        Do not validate individual row values at this time.
        
        Return your response as a JSON object with this structure:
        {
            "is_valid": false,
            "errors": [
                "Required column 'user_id' is missing from the data file",
                "Column 'age' is defined as INTEGER in schema but contains string data",
                "Column 'email' is defined as VARCHAR(255) but contains values longer than 255 characters"
            ]
        }"""
    )

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Wait for completion
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.status == 'completed':
            break
        elif run.status in ['failed', 'cancelled', 'expired']:
            raise Exception(f"Assistant run failed with status: {run.status}")

    # Get the response
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    response = messages.data[0].content[0].text.value

    # Clean up
    client.beta.assistants.delete(assistant.id)
    client.files.delete(schema_file_obj.id)
    client.files.delete(data_file_obj.id)

    print(response)
    
    return json.loads(response)

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
    data_sample = data_df.head().to_string()
    
    client = OpenAI()
    
    prompt = f"""I have a data file and a schema file that I need to validate.

Schema file contents:
{schema_str}

First few rows of the data file:
{data_sample}

Please validate the structure of the data file against the schema file. 
Focus only on schema-level validation:
1. Check if all required fields from the schema exist in the data file
2. Verify that the columns have the correct data types as specified in the schema
3. Validate any field length constraints defined in the schema

Do not validate individual row values at this time.

Make sure you only return unique errors.

Return your response as a JSON object with this structure:
{{
    "is_valid": false,
    "errors": [
        "Required column 'user_id' is missing from the data file",
        "Column 'age' is defined as INTEGER in schema but contains string data",
        "Column 'email' is defined as VARCHAR(255) but contains values longer than 255 characters"
    ]
}}"""

    response = client.chat.completions.create(
        model="o1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    print(response)
    
    return json.loads(response.choices[0].message.content) 