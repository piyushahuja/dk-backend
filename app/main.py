from fastapi import FastAPI, HTTPException
from typing import Dict
from .validation import validate_data_against_schema
from .error_detection import detect_data_errors, get_data_quality_report

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

        validation_result = validate_data_against_schema(schema_file, data_file, llm=True)
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
    Detect errors in a data file and return a comprehensive quality report.
    
    Args:
        request: Dict containing schema_file and data_file paths

    Returns:
        Dict containing error detection results in a structured format
    """
    schema_file = request["schema_file"]
    data_file = request["data_file"]

    try:
        quality_report = get_data_quality_report(schema_file, data_file)
        return {
            "status": "success",
            **quality_report
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 