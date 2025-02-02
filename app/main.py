from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Dict
from pathlib import Path
import uuid
import os
import re
from fastapi.security import APIKeyHeader
from .validation import validate_data_against_schema
from .error_detection import get_data_quality_report, cleanup_data_with_code_interpreter
from .cleanup import perform_cleanup_sequence
from fastapi.responses import FileResponse
from app.error_detection import logger

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
        request: Dict containing:
            - schema_file_id: ID of the schema file
            - data_file_id: ID of the data file
            - use_code_interpreter: Optional boolean to use OpenAI Code Interpreter instead of generating code

    Returns:
        Dict containing error detection results in a structured format
    """
    logger.info(f"Received error detection request: {request}")
    
    schema_file_id = request["schema_file_id"]
    data_file_id = request["data_file_id"]
    use_code_interpreter = request.get("use_code_interpreter", False)
    
    logger.debug(f"Processing files: schema={schema_file_id}, data={data_file_id}")
    
    if schema_file_id not in file_storage or data_file_id not in file_storage:
        logger.warning(f"File not found: schema={schema_file_id}, data={data_file_id}")
        raise HTTPException(status_code=404, detail="File not found")

    # Verify files still exist and haven't been tampered with
    if not file_storage[schema_file_id].exists() or not file_storage[data_file_id].exists():
        raise HTTPException(status_code=404, detail="File not found or has been removed")

    try:
        quality_report = get_data_quality_report(
            str(file_storage[schema_file_id]),
            str(file_storage[data_file_id]),
            use_code_interpreter=use_code_interpreter
        )
        logger.info("Error detection completed successfully")
        return {
            "status": "success",
            **quality_report
        }
    except Exception as e:
        logger.error(f"Error during error detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/cleanup")
async def cleanup_data(request: Dict) -> Dict:
    """
    Apply multiple cleanup operations to a data file in sequence.
    
    Args:
        request: Dict containing:
            - data_file_id: ID of the file to clean
            - schema_file_id: ID of schema file (required when use_code_interpreter=True)
            - cleanup_operations: List of cleanup operations to perform in sequence
                [
                    {
                        "id": str,
                        "description": str
                    },
                    ...
                ]
            - use_code_interpreter: Optional boolean to use OpenAI Code Interpreter instead of generating code
            - thread_id: Optional ID of existing thread from validation (only used with Code Interpreter)
            
    Returns:
        Dict containing:
            - new_file_id: ID of the cleaned file
            - changes_made: List of changes made by each operation
    """
    logger.info(f"Received cleanup request: {request}")
    
    data_file_id = request.get("data_file_id")
    cleanup_operations = request.get("cleanup_operations")
    use_code_interpreter = request.get("use_code_interpreter", True)
    thread_id = request.get("thread_id") if use_code_interpreter else None
    
    logger.debug(f"Processing cleanup for file: {data_file_id}")
    
    if not data_file_id or not cleanup_operations:
        logger.warning("Missing required parameters")
        raise HTTPException(status_code=400, detail="Missing data_file_id or cleanup_operations")
    
    if not isinstance(cleanup_operations, list):
        raise HTTPException(status_code=400, detail="cleanup_operations must be a list")
    
    if data_file_id not in file_storage:
        raise HTTPException(status_code=404, detail="Data file not found")
    
    if not file_storage[data_file_id].exists():
        raise HTTPException(status_code=404, detail="Data file not found or has been removed")
    
    try:
        if use_code_interpreter:
            schema_file_id = request.get("schema_file_id")
            if not schema_file_id:
                logger.warning("Missing schema_file_id for Code Interpreter cleanup")
                raise HTTPException(status_code=400, detail="schema_file_id is required when using Code Interpreter")
            if schema_file_id not in file_storage:
                raise HTTPException(status_code=404, detail="Schema file not found")
            if not file_storage[schema_file_id].exists():
                raise HTTPException(status_code=404, detail="Schema file not found or has been removed")
                
            # Use Code Interpreter version
            cleaned_file_path, changes_made = cleanup_data_with_code_interpreter(
                schema_file=str(file_storage[schema_file_id]),
                data_file=str(file_storage[data_file_id]),
                cleanup_operations=cleanup_operations,
                thread_id=thread_id
            )
        else:
            # Use existing code generation version
            cleaned_file_path, changes_made = perform_cleanup_sequence(
                str(file_storage[data_file_id]),
                cleanup_operations
            )
        
        logger.info("Cleanup completed successfully")
        
        # Generate new file_id for final cleaned file
        new_file_id = str(uuid.uuid4())
        file_storage[new_file_id] = Path(cleaned_file_path)
        
        return {
            "status": "success",
            "new_file_id": new_file_id,
            "changes_made": changes_made
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """
    Download a file using its file ID.
    
    Args:
        file_id: ID of the file to download
        
    Returns:
        FileResponse containing the requested file
    """
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = file_storage[file_id]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found or has been removed")
    
    try:
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="text/csv"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 