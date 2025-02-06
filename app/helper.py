import re
import json
import pandas as pd
from openai import OpenAI
import tempfile
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def parse_natural_language_response(response: str) -> dict:
    """
    Parse the assistant's response into our expected JSON schema.
    Handles both markdown format and JSON format.
    
    Example inputs:
    '''
    1. **Missing Values**:
       - **Description**: Several columns have missing values...
       - **Count**: 481 instances.
       - **Affected Rows**: All rows from index 0 to 98.
       - **Suggested Fix**: Ensure all required fields...
    '''
    
    or
    
    ```json
    [
        {
            "type": "Missing values",
            "count": 13,
            "rows": [5, 9, 12, 13, 22],
            "description": "The column 'Industry' contains...",
            "suggested_fix": "Consider imputing missing values..."
        }
    ]
    ```
    
    Returns:
        {
            "errors": [
                {
                    "type": "Missing Values",
                    "count": 481,
                    "description": "Several columns have missing values...",
                    "suggested_fix": "Ensure all required fields..."
                },
                ...
            ]
        }
    """
    import re
    
    # First try to find and parse JSON in the response
    json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        try:
            json_content = json_match.group(1)
            # Parse the JSON array and convert to our format
            json_data = json.loads(json_content)
            print(json_data)
            return json_data
        except json.JSONDecodeError as e:
            logger.warning(f"Found JSON block but failed to parse: {str(e)}")
    
    # If no valid JSON found, try markdown format
    # Initialize result structure
    result = {"errors": []}
    
    # Split into sections by numbered items
    sections = re.split(r'\d+\.\s+\*\*', response)[1:]  # Skip empty first split
    
    if not sections:  # If no markdown sections found
        logger.warning("No markdown sections or JSON found in response")
        return result
    
    for section in sections:
        try:
            # Extract the issue type (everything before the first ":")
            type_match = re.match(r'([^:]+):', section)
            if not type_match:
                continue
            issue_type = type_match.group(1).strip('*')
            
            # Extract other fields using regex
            count_match = re.search(r'\*\*Count\*\*:\s*(\d+)', section)
            description_match = re.search(r'\*\*Description\*\*:\s*([^*\n]+)', section)
            fix_match = re.search(r'\*\*Suggested Fix\*\*:\s*([^*\n]+)', section)
            
            error = {
                "type": issue_type,
                "count": int(count_match.group(1)) if count_match else 0,
                "description": description_match.group(1).strip() if description_match else "",
                "suggested_fix": fix_match.group(1).strip() if fix_match else ""
            }
            
            result["errors"].append(error)
            
        except Exception as e:
            logger.warning(f"Failed to parse section: {section}. Error: {str(e)}")
            continue
    
    return result

def parse_llm_json_response(response: str) -> dict:
    """Parse the LLM response, handling both JSON and natural language formats."""
    try:
        # First try to parse as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # If that fails, try to parse as natural language
        try:
            print("Parsing natural language response")
            return parse_natural_language_response(response)
        except Exception as e:
            logger.error(f"Failed to parse response in both JSON and natural language formats: {str(e)}")
            raise ValueError("Could not parse response in any supported format")

def generate_and_run_data_checks(issues: list, data_file: str, sample_size: int = 5) -> dict:
    """
    For each identified data quality issue, generates and executes Python code to detect similar issues
    in the full dataset.
    
    Args:
        issues: List of dicts containing 'issue' and 'solution' keys from LLM analysis
        data_file: Path to the full data CSV file
        sample_size: Number of rows to include in prompt for code generation
        
    Returns:
        Dict mapping issue descriptions to DataFrames containing problematic rows
    """
    client = OpenAI()
    data_df = pd.read_csv(data_file)
    sample_df = data_df.head(sample_size)
    results = {}
    
    # Common imports and utility functions that will be added to every generated check
    code_prefix = """
import pandas as pd
import numpy as np
import re
from typing import List

"""
    
    for issue_dict in issues:
        issue = issue_dict['issue']
        solution = issue_dict['solution']
        
        prompt = f"""Given this data quality issue and proposed solution:

Issue: {issue}
Solution: {solution}

Here's a sample of the data (first {sample_size} rows):
{sample_df.to_string()}

Write a Python function that identifies rows with this issue in a pandas DataFrame.
Requirements:
1. Function should take a pandas DataFrame as input
2. Return a list of row indices where the issue is found
3. Use clear variable names and add comments
4. Handle potential errors (null values, type mismatches, etc.)
5. Only include the function definition, no example usage or imports
6. Handle edge cases gracefully (empty DataFrame, missing columns, etc.)

Example format:
def check_data_quality(df):
    # Initialize list to store problematic row indices
    problematic_indices = []
    
    try:
        # Your check logic here
        pass
    except Exception as e:
        print(f"Error in check: {{str(e)}}")
        return []
        
    return problematic_indices
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract code from response and add necessary imports
        generated_code = parse_llm_json_response(response.choices[0].message.content, expect_python=True)
        full_code = code_prefix + generated_code
        
        # Create a temporary Python file with the generated code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            # Import and execute the generated function
            import importlib.util
            spec = importlib.util.spec_from_file_location("check_module", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Run the check on the full dataset and get problematic rows
            problematic_indices = module.check_data_quality(data_df)
            
            # Handle empty results
            if not problematic_indices:
                results[issue] = pd.DataFrame()
                continue
                
            problematic_rows = data_df.iloc[problematic_indices].copy()
            problematic_rows['row_index'] = problematic_indices  # Add original row indices
            results[issue] = problematic_rows
            
        except Exception as e:
            results[issue] = f"Error executing check: {str(e)}"
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {str(e)}")
    
    return results 

def parse_error_analysis_csv(csv_file_path: str) -> List[Dict]:
    """
    Parse the error analysis CSV file into a structured JSON format.
    
    Args:
        csv_file_path: Path to the CSV file containing error analysis results
        
    Returns:
        List of dictionaries, each containing:
            - id: The issue identifier (issue_1, issue_2, etc.)
            - rows: List of row indices where this issue was found
            
    Example:
    [
        {"id": "issue_1", "rows": [0, 5, 7]},
        {"id": "issue_2", "rows": [3, 9]},
        ...
    ]
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Get issue columns (all columns except row_index)
        issue_columns = [col for col in df.columns if col != 'row_index']
        
        # Initialize results list
        results = []
        
        # For each issue column
        for issue_id in issue_columns:
            # Get row indices where issue is True
            affected_rows = df[df[issue_id] == True]['row_index'].tolist()
            
            # Add to results if there are any affected rows
            if affected_rows:
                results.append({
                    "id": issue_id,
                    "rows": affected_rows
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error parsing error analysis CSV: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to parse error analysis CSV: {str(e)}") 