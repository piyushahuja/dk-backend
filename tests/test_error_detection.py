import pytest
from app.error_detection import (
    detect_data_errors,
    detect_data_errors_llm,
    detect_data_errors_traditional,
    describe_data_quality_issues
)
from app.helper import generate_and_run_data_checks
import pandas as pd

'''def test_detect_data_errors():
    # Test with traditional method
    errors_trad = detect_data_errors(
        schema_file="tests/fixtures/SAP_Customer_Master_Data.xlsx",
        data_file="tests/fixtures/test_no_err_col.csv",
        llm=False
    )
    assert isinstance(errors_trad, list)
    
    # Test with LLM method
    errors_llm = detect_data_errors(
        schema_file="tests/fixtures/SAP_Customer_Master_Data.xlsx",
        data_file="tests/fixtures/test_no_err_col.csv",
        llm=True
    )
    assert isinstance(errors_llm, list)

def test_detect_data_errors_invalid_files():
    with pytest.raises(FileNotFoundError):
        detect_data_errors(
            schema_file="nonexistent.xlsx",
            data_file="tests/fixtures/test_no_err_col.csv"
        )
    
    with pytest.raises(FileNotFoundError):
        detect_data_errors(
            schema_file="tests/fixtures/SAP_Customer_Master_Data.xlsx",
            data_file="nonexistent.csv"
        )'''

def test_describe_data_quality_issues():
    issues = describe_data_quality_issues(
        schema_file="tests/fixtures/SAP_Customer_Master_Data.xlsx",
        data_file="tests/fixtures/test_no_err_col.csv"
    )["errors"]

    for issue in issues:
        print(issue)
        print()

    assert isinstance(issues, list)
    assert all(isinstance(issue, object) for issue in issues)

'''def test_describe_data_quality_issues_invalid_files():
    with pytest.raises(FileNotFoundError):
        describe_data_quality_issues(
            schema_file="nonexistent.xlsx",
            data_file="tests/fixtures/test_no_err_col.csv"
        )
    
    with pytest.raises(FileNotFoundError):
        describe_data_quality_issues(
            schema_file="tests/fixtures/SAP_Customer_Master_Data.xlsx",
            data_file="nonexistent.csv"
        )

def test_detect_data_errors_traditional():
    errors = detect_data_errors_traditional(
        schema_file="tests/fixtures/SAP_Customer_Master_Data.xlsx",
        data_file="tests/fixtures/test_no_err_col.csv"
    )
    assert isinstance(errors, list)
    for error in errors:
        assert isinstance(error, dict)
        assert all(key in error for key in ["field", "row", "value", "error"])

def test_detect_data_errors_llm():
    errors = detect_data_errors_llm(
        schema_file="tests/fixtures/SAP_Customer_Master_Data.xlsx",
        data_file="tests/fixtures/test_no_err_col.csv"
    )
    assert isinstance(errors, list)
    for error in errors:
        assert isinstance(error, dict)
        assert all(key in error for key in ["field", "row", "value", "error"])'''

'''def test_end_to_end_data_quality_check():
    """
    Integration test that combines describing quality issues and checking for them in the data.
    Tests the full workflow from issue detection to row identification.
    """
    # First get the natural language description of issues
    issues = describe_data_quality_issues(
        schema_file="tests/fixtures/SAP_Customer_Master_Data.xlsx",
        data_file="tests/fixtures/test_no_err_col.csv"
    )
    
    print("\nIdentified Issues:")
    for issue in issues:
        print(f"- {issue}")
    
    # Convert the issues into the required format
    formatted_issues = [
        {
            "issue": issue["issue"],
            "solution": issue["solution"]
        }
        for issue in issues
    ]
    
    # Run the checks to find problematic rows
    results = generate_and_run_data_checks(
        issues=formatted_issues,
        data_file="tests/fixtures/test_no_err_col.csv"
    )
    
    # Print detailed results for manual inspection
    print("\nDetailed Results:")
    for issue, problem_rows in results.items():
        print(f"\nIssue: {issue}")
        if isinstance(problem_rows, str) and problem_rows.startswith("Error"):
            print(f"Error occurred: {problem_rows}")
        else:
            print(f"Found {len(problem_rows)} problematic rows:")
            print(problem_rows)
            print("\nColumns with issues:")
            print(problem_rows.columns.tolist())
    
    # Verify structure and types
    assert isinstance(issues, list)
    assert all(isinstance(issue, object) for issue in issues)
    assert isinstance(results, dict)
    
    for issue, problem_rows in results.items():
        if isinstance(problem_rows, str) and problem_rows.startswith("Error"):
            continue  # Skip error messages
        assert isinstance(problem_rows, pd.DataFrame)
        assert 'row_index' in problem_rows.columns
        assert len(problem_rows.columns) > 1  # Should have more columns than just row_index''' 