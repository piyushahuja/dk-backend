from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Dict
from pathlib import Path
import uuid
import os
import re
from fastapi.security import APIKeyHeader
from .validation import validate_data_against_schema
from .error_detection import detect_data_errors, get_data_quality_report

app = FastAPI()

# Create a secure upload directory outside the application root
UPLOAD_DIR = Path(".\\uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Ensure upload directory permissions are restricted
os.chmod(UPLOAD_DIR, 0o700)  # Only owner can read/write/execute

# Store file_id to filepath mappings
file_storage: Dict[str, Path] = {}

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    # Remove any directory components
    filename = Path(filename).name
    # Remove any non-alphanumeric chars except .-_
    return re.sub(r'[^a-zA-Z0-9.-]', '_', filename)

def get_secure_file_path(file_id: str, original_filename: str) -> Path:
    """Generate a secure file path within the upload directory."""
    safe_filename = sanitize_filename(original_filename)
    return UPLOAD_DIR / f"{file_id}_{safe_filename}"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict:
    """
    Upload a file and get a file identifier for future operations.
    
    Args:
        file: The file to upload

    Returns:
        Dict containing the file_id for future reference
    """
    file_id = str(uuid.uuid4())
    file_path = get_secure_file_path(file_id, file.filename)
    
    try:
        # Open file with restricted permissions (owner read/write only)
        with open(file_path, "wb", opener=lambda path, flags: os.open(path, flags, 0o600)) as f:
            content = await file.read()
            f.write(content)
        
        file_storage[file_id] = file_path
        return {"file_id": file_id}
    except Exception as e:
        if file_path.exists():
            file_path.unlink()  # Clean up on error
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/validate_schema")
async def validate_schema(request: Dict) -> Dict:
    """
    Validate a data file against a schema file.
    
    Args:
        request: Dict containing schema_file_id and data_file_id

    Returns:
        Dict containing validation results
    """
    schema_file_id = request["schema_file_id"]
    data_file_id = request["data_file_id"]
    
    if schema_file_id not in file_storage or data_file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Verify files still exist and haven't been tampered with
    if not file_storage[schema_file_id].exists() or not file_storage[data_file_id].exists():
        raise HTTPException(status_code=404, detail="File not found or has been removed")
    
    try:
        validation_result = validate_data_against_schema(
            str(file_storage[schema_file_id]),
            str(file_storage[data_file_id]),
            llm=True
        )
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
        request: Dict containing schema_file_id and data_file_id

    Returns:
        Dict containing error detection results in a structured format
    """
    schema_file_id = request["schema_file_id"]
    data_file_id = request["data_file_id"]
    
    if schema_file_id not in file_storage or data_file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify files still exist and haven't been tampered with
    if not file_storage[schema_file_id].exists() or not file_storage[data_file_id].exists():
        raise HTTPException(status_code=404, detail="File not found or has been removed")

    try:
        quality_report = get_data_quality_report(
            str(file_storage[schema_file_id]),
            str(file_storage[data_file_id])
        )
        return {
            "status": "success",
            **quality_report
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 