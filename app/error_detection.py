from typing import List, Dict
import os
import pandas as pd
import pandera as pa
from openai import OpenAI
import json
from dotenv import load_dotenv
from app.helper import parse_llm_json_response

load_dotenv()

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
    data_sample = data_df.head(20).to_json(orient='records')
    
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

    You must identify the THREE most important issues in the dataset.
    
    Return your response as a JSON array of objects, where each object contains the following fields:
    {{
        "issue": "description of the issue",
        "solution": "suggested solution to the issue (this should be a data cleaning solution)"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return parse_llm_json_response(response.choices[0].message.content)