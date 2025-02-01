from fastapi import FastAPI, HTTPException
from typing import Dict
from .validation import validate_data_against_schema
from .error_detection import detect_data_errors

app = FastAPI()

@app.post("/validate_schema")
async def validate_schema(request: Dict) -> Dict:
    """
    Validate a data file against a schema file.
    
    Args:
        request: Dict containing schema_file and data_file paths

    Returns:
        Dict containing validation results
    """
    schema_file = request["schema_file"]
    data_file = request["data_file"]
    try:

        validation_result = validate_data_against_schema(schema_file, data_file)
        return {
            "status": "success",
            "is_valid": validation_result["is_valid"],
            "errors": validation_result.get("errors", [])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/detect_errors")
async def detect_errors(request: Dict) -> Dict:
    """
    Detect errors in a data file.
    
    Args:
        request: Dict containing data_file path

    Returns:
        Dict containing error detection results
    """
    schema_file = request["schema_file"]
    data_file = request["data_file"]
    try:

        error_results = detect_data_errors(schema_file, data_file)
        return {
            "status": "success",
            "errors": error_results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 