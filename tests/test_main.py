from fastapi.testclient import TestClient
from app.main import app
import pytest
from pathlib import Path

client = TestClient(app)

'''def test_validate_schema_success():
    # Assuming you have test files in a tests/fixtures directory
    response = client.post("/validate_schema", json={
        "schema_file": "tests/fixtures/SAP_Customer_Master_Data.xlsx",
        "data_file": "tests/fixtures/test_no_err_col.csv"
    })

    print(response.json())
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "is_valid" in response.json()

def test_validate_schema_invalid_file():
    response = client.post("/validate_schema", json={
        "schema_file": "nonexistent.json",
        "data_file": "nonexistent.csv"
    })
    
    assert response.status_code == 400'''

'''def test_detect_errors_success():
    response = client.post("/detect_errors", json={
        "schema_file": "tests/fixtures/SAP_Customer_Master_Data.xlsx",
        "data_file": "tests/fixtures/test_no_err_col.csv"
    })

    print(response.json())
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "errors" in response.json()

def test_detect_errors_invalid_file():
    response = client.post("/detect_errors", json={
        "schema_file": "nonexistent.xlsx",
        "data_file": "nonexistent.csv"
    })
    
    assert response.status_code == 400'''

def test_upload_file_success():
    # Create a test file
    test_file_content = b"test content"
    test_filename = "test.txt"
    
    # Simulate file upload using TestClient
    files = {"file": (test_filename, test_file_content, "text/plain")}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    assert "file_id" in response.json()
    assert isinstance(response.json()["file_id"], str)

def test_upload_file_empty():
    # Try uploading an empty file
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    assert "file_id" in response.json()

def test_end_to_end_workflow():
    # 1. Upload schema file
    schema_path = Path("tests/fixtures/SAP_Customer_Master_Data.xlsx")
    with open(schema_path, "rb") as f:
        schema_response = client.post("/upload", 
            files={"file": ("SAP_Customer_Master_Data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
    assert schema_response.status_code == 200
    schema_file_id = schema_response.json()["file_id"]

    # 2. Upload data file
    data_path = Path("tests/fixtures/test_no_err_col.csv")
    with open(data_path, "rb") as f:
        data_response = client.post("/upload", 
            files={"file": ("test_no_err_col.csv", f, "text/csv")}
        )
    assert data_response.status_code == 200
    data_file_id = data_response.json()["file_id"]

    # 3. Test validation with uploaded files
    '''validate_response = client.post("/validate_schema", json={
        "schema_file_id": schema_file_id,
        "data_file_id": data_file_id
    })
    assert validate_response.status_code == 200
    assert validate_response.json()["status"] == "success"'''

    # 4. Test error detection with uploaded files
    detect_response = client.post("/detect_errors", json={
        "schema_file_id": schema_file_id,
        "data_file_id": data_file_id
    })

    print(detect_response.json())
    
    assert detect_response.status_code == 200
    assert detect_response.json()["status"] == "success"