from typing import Tuple, List, Dict
import pandas as pd
from openai import OpenAI
import tempfile
from pathlib import Path
import os
from .helper import parse_llm_json_response

def perform_cleanup_sequence(file_path: str, cleanup_operations: List[Dict]) -> Tuple[Path, List[str]]:
    """
    Performs multiple cleanup operations in sequence on the data file.
    
    Args:
        file_path: Path to the data file
        cleanup_operations: List of dicts containing cleanup operation details:
            [
                {
                    "id": str,
                    "description": str
                },
                ...
            ]
        
    Returns:
        Tuple containing:
            - Path to the final cleaned file
            - List of changes made by each operation
    """
    # Read the original data
    current_df = pd.read_csv(file_path)
    all_changes = []
    
    # Apply each cleanup operation in sequence
    for operation in cleanup_operations:
        try:
            # Generate and execute cleanup code for this operation
            cleanup_code = generate_cleanup_code(current_df, operation)
            
            # Create a temporary Python file with the generated code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(cleanup_code)
                temp_file = f.name
            
            try:
                # Import and execute the generated function
                import importlib.util
                spec = importlib.util.spec_from_file_location("cleanup_module", temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Run the cleanup function and update the current DataFrame
                cleaned_df, changes_made = module.cleanup_data(current_df)
                current_df = cleaned_df
                all_changes.append({
                    "cleanup_id": operation["id"],
                    "changes": changes_made
                })
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file}: {str(e)}")
                    
        except Exception as e:
            all_changes.append({
                "cleanup_id": operation["id"],
                "changes": f"Error during cleanup: {str(e)}"
            })
            # Continue with next cleanup operation despite error
            continue
    
    # Save final cleaned data to a new file
    cleaned_file = Path(file_path).parent / f"cleaned_{Path(file_path).name}"
    current_df.to_csv(cleaned_file, index=False)
    
    return cleaned_file, all_changes

def generate_cleanup_code(df: pd.DataFrame, operation: Dict) -> str:
    """
    Generates Python code to perform a specific cleanup operation.
    
    Args:
        df: Current state of the data to clean
        operation: Dict containing:
            - id: Unique identifier for the cleanup operation
            - description: Description of what the cleanup should do
        
    Returns:
        String containing Python code that defines a cleanup_data function
    """
    client = OpenAI()
    
    prompt = f"""Write Python code to clean this dataset based on this cleanup operation:

Cleanup Operation: {operation['description']}

Data sample (first few rows):
{df.head().to_string()}

Write a function that takes a pandas DataFrame as input and returns:
1. The cleaned DataFrame
2. A description of changes made

Requirements:
1. Function must be named cleanup_data
2. Must handle potential errors gracefully
3. Must not modify the input DataFrame (create a copy)
4. Must return a tuple (cleaned_df, changes_description)
5. Include clear comments explaining the cleanup steps

Example format:
def cleanup_data(df):
    # Create a copy of the input DataFrame
    cleaned_df = df.copy()
    
    # Track changes
    changes = []
    
    try:
        # Your cleanup logic here
        # Add descriptions to changes list
        pass
    except Exception as e:
        return df, f"Error during cleanup: {{str(e)}}"
    
    return cleaned_df, "; ".join(changes)
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Add necessary imports to the generated code
    code_prefix = """
import pandas as pd
import numpy as np
import re
from typing import Tuple

"""
    
    generated_code = parse_llm_json_response(response.choices[0].message.content, expect_python=True)
    return code_prefix + generated_code 