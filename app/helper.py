import re
import json
import pandas as pd
from openai import OpenAI
import tempfile
import os

def parse_llm_json_response(response: str, expect_python: bool = False) -> any:
    """
    Parses JSON or Python code from LLM responses that may be wrapped in markdown code fences.
    
    Args:
        response: String response from LLM, potentially containing markdown code fences
        expect_python: If True, looks for Python code block instead of JSON
        
    Returns:
        If expect_python=False: Parsed JSON data
        If expect_python=True: Python code as string
        
    Example:
        Input: '```json\n["item1", "item2"]\n```'
        Output: ['item1', 'item2']
        
        Input: '```python\ndef my_function():\n    pass\n```'
        Output: 'def my_function():\n    pass'
    """
    # First try direct JSON parsing if we're not expecting Python
    if not expect_python:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
    
    # Look for content between code fences
    pattern = r'```(?:python|json)?\n(.*?)\n```' if expect_python else r'```(?:json)?\n(.*?)\n```'
    code_block_match = re.search(pattern, response, re.DOTALL)
    
    if code_block_match:
        content = code_block_match.group(1)
        if expect_python:
            return content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from code block: {e}")
    
    raise ValueError("No valid code block found in response")

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