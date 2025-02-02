import pytest
import os
from pathlib import Path
import json
import pandas as pd
from app.assistant_service import AssistantService
import tempfile

@pytest.fixture
def assistant_service():
    """Create an AssistantService instance for testing."""
    return AssistantService()

@pytest.fixture
def test_csv_file():
    """Create a test CSV file with some sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create a simple dataset with some intentional issues
        data = pd.DataFrame({
            'id': [1, 2, 3, None, 5],
            'name': ['John', 'Jane', '', 'Bob', 'Alice'],
            'age': [25, 'invalid', 35, 40, 45],
            'email': ['john@test.com', 'invalid-email', 'jane@test.com', 'bob@test.com', 'alice@test.com']
        })
        data.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

def test_data_validation_workflow(assistant_service, test_csv_file):
    """
    Test a complete data validation workflow using the Assistants API.
    This test verifies:
    1. File upload
    2. Assistant creation
    3. Running a conversation
    4. Getting results via file
    5. Proper cleanup
    """
    try:
        # Create assistant with test file
        assistant, file_ids = assistant_service.create_assistant_with_files(
            name="Data Validator",
            instructions="""You are a data validation assistant. Analyze the CSV file and identify data quality issues.
            Look for:
            1. Missing values
            2. Invalid data types
            3. Invalid email formats
            
            After analysis, create a CSV file named 'validation_results.csv' with these columns:
            - column: The name of the column with issues
            - issue_type: Description of the issue
            - count: Number of rows affected
            - affected_rows: Comma-separated list of row indices""",
            files=[test_csv_file]
        )
        
        assert assistant.id is not None, "Assistant creation failed"
        assert len(file_ids) == 1, "File upload failed"
        
        # Run the analysis
        response = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message="""Please analyze the CSV file and identify any data quality issues.
            Focus on:
            1. The 'age' column for non-numeric values
            2. The 'email' column for invalid email formats
            3. Any missing values
            
            Save your findings in validation_results.csv"""
        )
        
        assert response["thread_id"] is not None, "Thread creation failed"
        assert response["file_id"] is not None, "No results file was created"
        
        # Download and verify the results
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        assistant_service.download_file(response["file_id"], temp_file)
        results_df = pd.read_csv(temp_file)
        
        # Print the results for debugging
        print("\nValidation Results:")
        print(results_df.to_string())
        print("\n")
        
        # Just verify we got a readable CSV with some content
        assert len(results_df) > 0, "Results file is empty"
        print(f"Columns found in results: {list(results_df.columns)}")

    finally:
        # Clean up
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {e}")
        
        if 'assistant' in locals() and 'file_ids' in locals():
            assistant_service.cleanup_resources(assistant.id, file_ids)

def test_file_generation_and_download(assistant_service, test_csv_file):
    """
    Test file generation and download functionality.
    This verifies that:
    1. Assistant can generate files
    2. Files can be downloaded correctly
    3. Generated files have expected content
    """
    temp_file = None
    try:
        # Create assistant with test file
        assistant, file_ids = assistant_service.create_assistant_with_files(
            name="File Generator",
            instructions="""You are a file generation test assistant. When asked:
            1. Read the input CSV file
            2. Create a new file called 'summary.csv'
            3. Include columns: total_rows, total_columns, column_names
            4. Save basic statistics about the input file""",
            files=[test_csv_file]
        )
        
        assert assistant.id is not None, "Assistant creation failed"
        
        # Request file generation
        response = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message="Please analyze the CSV file and create a summary.csv file with the requested information."
        )
        
        assert response["file_id"] is not None, "No file was generated"
        
        # Download and verify the generated file
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        assistant_service.download_file(response["file_id"], temp_file)
        
        # Read and verify the file
        summary_df = pd.read_csv(temp_file)
        
        # Print the summary for debugging
        print("\nGenerated Summary:")
        print(summary_df.to_string())
        print("\n")
        
        # Verify basic content
        assert 'total_rows' in summary_df.columns, "Missing total_rows column"
        assert 'total_columns' in summary_df.columns, "Missing total_columns column"
        assert 'column_names' in summary_df.columns, "Missing column_names column"
        
        # Verify the numbers match our test file
        test_df = pd.read_csv(test_csv_file)
        assert summary_df['total_rows'].iloc[0] == len(test_df), "Row count mismatch"
        assert summary_df['total_columns'].iloc[0] == len(test_df.columns), "Column count mismatch"
        
    finally:
        # Clean up
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {e}")
        
        if 'assistant' in locals() and 'file_ids' in locals():
            assistant_service.cleanup_resources(assistant.id, file_ids)

def test_conversation_thread_management(assistant_service):
    """
    Test conversation thread management.
    This verifies that:
    1. Threads are created correctly
    2. Messages are sent and received
    3. Multiple messages in the same thread maintain context
    """
    temp_file1 = None
    temp_file2 = None
    try:
        # Create a simple assistant for testing conversations
        assistant, _ = assistant_service.create_assistant_with_files(
            name="Conversation Tester",
            instructions="""You are a test assistant. For each message:
            1. Create a file named 'message_info.csv'
            2. Write a single row with columns: message_number, content
            3. message_number should increment with each message in the thread
            4. content should be the message you received

            A new file should be created for each message you receive.
            """,
            files=[]
        )
        
        # First message
        response1 = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message="This is message 1"
        )
        
        thread_id = response1["thread_id"]
        assert response1["file_id"] is not None, "No file created for first message"
        
        # Download and verify first message file
        temp_file1 = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        assistant_service.download_file(response1["file_id"], temp_file1)
        
        # Read and verify the file
        message1_df = pd.read_csv(temp_file1)
        
        print("\nFirst Message Info:")
        print(message1_df.to_string())
        print("\n")
        
        assert message1_df['message_number'].iloc[0] == 1, "First message not properly numbered"
        assert message1_df['content'].iloc[0] == "This is message 1", "First message content mismatch"
        
        # Second message in same thread
        response2 = assistant_service.run_conversation(
            assistant_id=assistant.id,
            message="This is message 2",
            thread_id=thread_id
        )
        
        assert response2["thread_id"] == thread_id, "Thread ID not maintained"
        assert response2["file_id"] is not None, "No file created for second message"
        
        # Download and verify second message file
        temp_file2 = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        assistant_service.download_file(response2["file_id"], temp_file2)
        
        # Read and verify the file
        message2_df = pd.read_csv(temp_file2)
        
        print("\nSecond Message Info:")
        print(message2_df.to_string())
        print("\n")
        
        assert message2_df['message_number'].iloc[0] == 2, "Second message not properly numbered"
        assert message2_df['content'].iloc[0] == "This is message 2", "Second message content mismatch"
        
    finally:
        # Clean up
        for temp_file in [temp_file1, temp_file2]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file}: {e}")
        
        if 'assistant' in locals():
            assistant_service.cleanup_resources(assistant.id, []) 