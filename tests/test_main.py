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

'''def test_upload_file_success():
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
    assert "file_id" in response.json()'''

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
    validate_response = client.post("/validate_schema", json={
        "schema_file_id": schema_file_id,
        "data_file_id": data_file_id
    })
    assert validate_response.status_code == 200
    assert validate_response.json()["status"] == "success"

    # 4. Test error detection with uploaded files
    detect_response = client.post("/detect_errors", json={
        "schema_file_id": schema_file_id,
        "data_file_id": data_file_id
    })

    print(detect_response.json())
    
    assert detect_response.status_code == 200
    assert detect_response.json()["status"] == "success"

'''def test_cleanup_success():
    # 1. Upload test data file
    data_path = Path("tests/fixtures/test_no_err_col.csv")
    with open(data_path, "rb") as f:
        data_response = client.post("/upload", 
            files={"file": ("test_no_err_col.csv", f, "text/csv")}
        )
    assert data_response.status_code == 200
    data_file_id = data_response.json()["file_id"]

    # Upload schema file
    schema_path = Path("tests/fixtures/SAP_Customer_Master_Data.xlsx")
    with open(schema_path, "rb") as f:
        schema_response = client.post("/upload", 
            files={"file": ("SAP_Customer_Master_Data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
    assert schema_response.status_code == 200
    schema_file_id = schema_response.json()["file_id"]

    # 2. Test cleanup with multiple operations
    cleanup_response = client.post("/cleanup", json={
        "data_file_id": data_file_id,
        "schema_file_id": schema_file_id,
        "cleanup_operations": [
            {
                "id": "cleanup_1",
                "description": "Fill missing values in numeric columns with median"
            },
            {
                "id": "cleanup_2",
                "description": "Remove special characters from text columns"
            }
        ]
    })
    
    assert cleanup_response.status_code == 200
    assert cleanup_response.json()["status"] == "success"
    assert "new_file_id" in cleanup_response.json()
    assert "changes_made" in cleanup_response.json()
    assert isinstance(cleanup_response.json()["changes_made"], list)
    
    # Verify changes_made structure
    changes = cleanup_response.json()["changes_made"]
    for change in changes:
        assert "cleanup_id" in change
        assert "changes" in change'''

'''def test_cleanup_invalid_file_id():
    response = client.post("/cleanup", json={
        "data_file_id": "nonexistent",
        "schema_file_id": "nonexistent",
        "cleanup_operations": [
            {
                "id": "cleanup_1",
                "description": "Fill missing values"
            }
        ]
    })
    assert response.status_code == 404

def test_cleanup_invalid_request():
    # Test missing file_id
    response = client.post("/cleanup", json={
        "cleanup_operations": [
            {
                "id": "cleanup_1",
                "description": "Fill missing values"
            }
        ]
    })
    assert response.status_code == 400

    # Test missing cleanup_operations
    response = client.post("/cleanup", json={
        "data_file_id": "some_id",
        "schema_file_id": "some_schema_id"
    })
    assert response.status_code == 400

    # Test invalid cleanup_operations type
    response = client.post("/cleanup", json={
        "data_file_id": "some_id",
        "schema_file_id": "some_schema_id",
        "cleanup_operations": "not_a_list"
    })
    assert response.status_code == 400'''

'''def test_cleanup_end_to_end():
    # 1. Upload test data file
    data_path = Path("tests/fixtures/test_no_err_col.csv")
    with open(data_path, "rb") as f:
        data_response = client.post("/upload", 
            files={"file": ("test_no_err_col.csv", f, "text/csv")}
        )
    assert data_response.status_code == 200
    file_id = data_response.json()["file_id"]

    # 2. Get error detection and cleanup suggestions
    detect_response = client.post("/detect_errors", json={
        "schema_file_id": file_id,
        "data_file_id": file_id  # Using same file for simplicity in test
    })
    assert detect_response.status_code == 200
    
    # Extract cleanup operations from suggestions
    cleanup_operations = [
        {
            "id": option["id"],
            "description": option["description"]
        }
        for option in detect_response.json().get("cleanupOptions", [])
    ]
    
    if cleanup_operations:  # Only test cleanup if suggestions were found
        # 3. Apply cleanup operations
        cleanup_response = client.post("/cleanup", json={
            "file_id": file_id,
            "cleanup_operations": cleanup_operations[0:3]
        })
        
        assert cleanup_response.status_code == 200
        assert cleanup_response.json()["status"] == "success"
        assert "new_file_id" in cleanup_response.json()
        
        # Verify changes were recorded for each cleanup operation
        changes = cleanup_response.json()["changes_made"]
        assert len(changes) == len(cleanup_operations[0:3])
        for change, operation in zip(changes, cleanup_operations[0:3]):
            assert change["cleanup_id"] == operation["id"]
            assert isinstance(change["changes"], str)'''

'''def test_download_success():
    # 1. Upload a file first
    data_path = Path("tests/fixtures/test_no_err_col.csv")
    with open(data_path, "rb") as f:
        upload_response = client.post("/upload", 
            files={"file": ("test_no_err_col.csv", f, "text/csv")}
        )
    assert upload_response.status_code == 200
    file_id = upload_response.json()["file_id"]
    
    # 2. Download the file
    response = client.get(f"/download/{file_id}")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "content-disposition" in response.headers
    
    # Verify the content can be read as CSV
    content = response.content.decode()
    assert len(content) > 0
    assert content.count('\n') > 0  # Should have at least header row

def test_download_invalid_file_id():
    response = client.get("/download/nonexistent")
    assert response.status_code == 404

def test_cleanup_and_download_workflow():
    # 1. Upload test data file
    data_path = Path("tests/fixtures/test_no_err_col.csv")
    with open(data_path, "rb") as f:
        data_response = client.post("/upload", 
            files={"file": ("test_no_err_col.csv", f, "text/csv")}
        )
    assert data_response.status_code == 200
    data_file_id = data_response.json()["file_id"]

    # Upload schema file
    schema_path = Path("tests/fixtures/SAP_Customer_Master_Data.xlsx")
    with open(schema_path, "rb") as f:
        schema_response = client.post("/upload", 
            files={"file": ("SAP_Customer_Master_Data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
    assert schema_response.status_code == 200
    schema_file_id = schema_response.json()["file_id"]

    # 2. Apply some cleanup operations
    cleanup_response = client.post("/cleanup", json={
        "data_file_id": data_file_id,
        "schema_file_id": schema_file_id,
        "cleanup_operations": [
            {
                "id": "cleanup_1",
                "description": "Fill missing values in numeric columns with median"
            }
        ]
    })
    assert cleanup_response.status_code == 200
    new_file_id = cleanup_response.json()["new_file_id"]
    
    # 3. Download the cleaned file
    download_response = client.get(f"/download/{new_file_id}")
    assert download_response.status_code == 200
    assert "text/csv" in download_response.headers["content-type"]
    
    # Verify the content can be read as CSV
    content = download_response.content.decode()
    assert len(content) > 0
    assert content.count('\n') > 0  # Should have at least header row'''

'''def test_detect_error_code():
    # 1. Upload test data file
    data_path = Path("tests/fixtures/test_no_err_col.csv")
    with open(data_path, "rb") as f:
        data_response = client.post("/upload", 
            files={"file": ("test_no_err_col.csv", f, "text/csv")}
        )
    assert data_response.status_code == 200
    file_id = data_response.json()["file_id"]
    
    # 2. Test error code detection
    error_description = "In some rows, the phone number column contains non-numeric values"
    response = client.post(
        f"/detect_error_code",
        json={"error_description": error_description, "file_id": file_id}
    )
    
    print("\nError Detection Response:")
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "affected_rows" in response.json()
    assert "code_description" in response.json()
    assert isinstance(response.json()["affected_rows"], list)
    assert isinstance(response.json()["code_description"], str)
    
    # Print detailed results for debugging
    print("\nDetailed Results:")
    print(f"Affected Rows: {response.json()['affected_rows']}")
    print(f"Code Description: {response.json()['code_description']}")'''