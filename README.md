# tailorflow

Route: `/api/validate-files`

Request formation:

```
formData.append('masterFile', masterFile);
formData.append('dataFile', dataFile);

{
    method: 'POST',
    body: formData,

  }
```

```
response = {
    "errors": [
        {
            "type": "Duplicate Records",
            "count": 15,
            "description": "Records sharing the same key identifiers"
        },
        {
            "type": "Missing Values",
            "count": 8,
            "description": "Required fields with no data"
        },
        {
            "type": "Inconsistent Dates",
            "count": 5,
            "description": "Dates not following the standard format"
        },
        {
            "type": "Invalid Product Codes",
            "count": 3,
            "description": "Codes not found in master data"
        }
    ],
    "cleaningOptions": [
        {
            "id": "duplicates",
            "label": "Remove Duplicates",
            "description": "Automatically remove duplicate records based on key identifiers",
            "count": 15,
            "errorType": "Duplicate Records"
        },
        {
            "id": "missing",
            "label": "Fix Missing Data",
            "description": "Fill missing values with appropriate defaults or remove records",
            "count": 8,
            "errorType": "Missing Values"
        },
        {
            "id": "dates",
            "label": "Standardize Dates",
            "description": "Convert all dates to YYYY-MM-DD format",
            "count": 5,
            "errorType": "Inconsistent Dates"
        },
        {
            "id": "products",
            "label": "Validate Product Codes",
            "description": "Check and correct product codes against master data",
            "count": 3,
            "errorType": "Invalid Product Codes"
        }
    ]
}
```

Route: `/api/validate-schema`

```
Request formation:

formData.append('masterFile', masterFile);
formData.append('dataFile', dataFile);

{
    method: 'POST',
    body: formData,

  }
```

```
response = [
  "Column 'customer_id' missing in data file",
  "Expected date format YYYY-MM-DD for 'purchase_date'",
  "Invalid data type in 'quantity' column (expected number)",
]
```

Claude wrote this code to handle formData:

```
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict
import shutil
from pathlib import Path
import os
from datetime import datetime

app = FastAPI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload_file(upload_file: UploadFile, field_name: str) -> str:
    """Save uploaded file and return the path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create safe filename with timestamp
    filename = f"{field_name}_{timestamp}{Path(upload_file.filename).suffix}"
    file_path = UPLOAD_DIR / filename

    try:
        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return str(file_path)
    finally:
        upload_file.file.close()

@app.post("/api/validate-schema")
async def validate_schema(
    masterFile: UploadFile = File(...),
    dataFile: UploadFile = File(...)
) -> Dict:
    try:
        # Save uploaded files
        master_path = save_upload_file(masterFile, "master")
        data_path = save_upload_file(dataFile, "data")

        # Log file information
        print("Master File:", {
            "filename": masterFile.filename,
            "content_type": masterFile.content_type,
            "saved_path": master_path
        })

        print("Data File:", {
            "filename": dataFile.filename,
            "content_type": dataFile.content_type,
            "saved_path": data_path
        })

        # Add your validation logic here
        # For example:
        # master_schema = validate_master_file(master_path)
        # data_validation = validate_data_file(data_path, master_schema)

        return {
            "message": "Files uploaded successfully",
            "masterFile": Path(master_path).name,
            "dataFile": Path(data_path).name
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing uploaded files: {str(e)}"
        )

# Optional: Cleanup endpoint to remove processed files
@app.delete("/api/cleanup/{filename}")
async def cleanup_file(filename: str):
    try:
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            os.remove(file_path)
            return {"message": f"File {filename} deleted successfully"}
        return {"message": f"File {filename} not found"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting file: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

```
