from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Dict
from pathlib import Path
import uuid
import os
import re
from fastapi.security import APIKeyHeader
from .helper import parse_llm_json_response
from .validation import validate_data_against_schema
from .error_detection import get_data_quality_report, cleanup_data_with_code_interpreter
from .cleanup import perform_cleanup_sequence
from fastapi.responses import FileResponse
from app.error_detection import logger
from fastapi.middleware.cors import CORSMiddleware
from .assistant_manager import AssistantManager
from .assistant_service import AssistantService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific frontend URLs for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# # Create a secure upload directory outside the application root
# UPLOAD_DIR = Path(".\\uploads")
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# # Ensure upload directory permissions are restricted
# os.chmod(UPLOAD_DIR, 0o700)  # Only owner can read/write/execute

try:
    UPLOAD_DIR = Path("./uploads")  # Use forward slash for better cross-platform compatibility
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure upload directory permissions are restricted
    # Set permissions to read, write, and execute only by the owner
    UPLOAD_DIR.chmod(0o700)
except PermissionError:
    raise PermissionError("Failed to set directory permissions. Check user permissions.")
except Exception as e:
    raise Exception(f"An error occurred while setting up the upload directory: {str(e)}")

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
            - prompt: Optional string with additional error detection instructions
            
    Returns:
        Dict containing error detection results in a structured format
    """
    logger.info(f"Received error detection request: {request}")
    
    schema_file_id = request["schema_file_id"]
    data_file_id = request["data_file_id"]
    use_code_interpreter = request.get("use_code_interpreter", False)
    custom_prompt = request.get("prompt")
    
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
            use_code_interpreter=use_code_interpreter,
            custom_prompt=custom_prompt
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
            - custom_cleanup_prompt: Optional string with additional cleanup instructions
            
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
    custom_cleanup_prompt = request.get("prompt")
    
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
                thread_id=thread_id,
                custom_cleanup_prompt=custom_cleanup_prompt
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

@app.post("/detect_error_code")
async def detect_error_code(request: Dict) -> Dict:
    """
    Generate and run code to detect specific errors in a data file.
    
    Args:
        request: Dict containing:
            - file_id: ID of the data file to analyze
            - error_description: Description of the error to detect

    Returns:
        Dict containing:
            - affected_rows: List of row indices where the error was found
            - code_description: Description of the code used to detect the error
    """
    file_id = request.get("file_id")
    error_description = request.get("error_description")

    print(f"File ID: {file_id}")
    print(f"Error Description: {error_description}")

    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_storage[file_id].exists():
        raise HTTPException(status_code=404, detail="File not found or has been removed")
    
    assistant_service = AssistantService()
    
    try:
        # Create new assistant with the file
        assistant, file_ids = assistant_service.create_assistant_with_files(
            name="Error Detector",
            instructions="""You are a data error detection assistant. Your task is to:
            1. Write Python code to detect specific data quality issues
            1a. Do this intelligently. Clean and transform the data if necessary.
            2. Return both the code used and the row indices where issues were found
            3. Handle edge cases and errors gracefully""",
            files=[str(file_storage[file_id])]
        )
        
        # Run the conversation
        response = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message=f"""Please write Python code to detect the following error in the data file:
            
            {error_description}
            
            Return your response as a JSON object with:
            1. affected_rows: List of row indices where the error was found
            2. code_description: Brief description of how the code you wrote works"""
        )

        content = parse_llm_json_response(response["message"])
        print(f"Content: {content}")
        
        # Clean up the assistant and files after use
        assistant_service.cleanup_resources(assistant.id, file_ids)
        
        # Parse the response
        if isinstance(content, dict):
            return {
                "status": "success",
                **content
            }
        else:
            raise Exception("Invalid response format from assistant")
            
    except Exception as e:
        logger.error(f"Error in detect_error_code: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) 