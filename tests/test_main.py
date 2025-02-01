from fastapi.testclient import TestClient
from app.main import app
import pytest

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

def test_detect_errors_success():
    response = client.post("/detect_errors", json={
        "schema_file": "tests/fixtures/SAP_Customer_Master_Data.xlsx",
        "data_file": "tests/fixtures/test_no_err_col.csv"
    })

    print(response.json())
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "errors" in response.json()

'''def test_detect_errors_invalid_file():
    response = client.post("/detect_errors", json={
        "schema_file": "nonexistent.xlsx",
        "data_file": "nonexistent.csv"
    })
    
    assert response.status_code == 400''' 