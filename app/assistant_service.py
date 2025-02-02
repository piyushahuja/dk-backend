from typing import Dict, List, Optional, Any
import os
from openai import OpenAI
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class AssistantService:
    """Service for handling OpenAI Assistants API operations."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI()
        
    def create_assistant_with_files(
        self,
        name: str,
        instructions: str,
        files: List[str],
        model: str = "gpt-4o"
    ) -> tuple[Any, List[str]]:
        """
        Create an assistant with file attachments.
        
        Args:
            name: Name of the assistant
            instructions: Instructions for the assistant
            files: List of file paths to attach
            model: Model to use (default: gpt-4o)
            
        Returns:
            Tuple of (assistant object, list of file IDs)
        """
        try:
            # Upload all files
            file_ids = []
            for file_path in files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                    
                file_obj = self.client.files.create(
                    file=open(file_path, "rb"),
                    purpose="assistants"
                )
                file_ids.append(file_obj.id)
            
            # Create assistant
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                model=model,
                tools=[{"type": "code_interpreter"}],
                tool_resources={
                    "code_interpreter": {
                        "file_ids": file_ids
                    }
                }
            )
            
            return assistant, file_ids
            
        except Exception as e:
            logger.error(f"Error creating assistant: {str(e)}", exc_info=True)
            # Clean up any files that were uploaded
            for file_id in file_ids:
                try:
                    self.client.files.delete(file_id)
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up file {file_id}: {str(cleanup_error)}")
            raise
    
    def run_conversation(
        self,
        assistant_id: str,
        message: str,
        thread_id: Optional[str] = None
    ) -> Dict:
        """
        Run a conversation with an assistant and get the response.
        
        Args:
            assistant_id: ID of the assistant to use
            message: Message to send to the assistant
            thread_id: Optional existing thread ID to continue conversation
            
        Returns:
            Dict containing the assistant's response
        """
        try:
            # Create or retrieve thread
            if thread_id:
                thread = self.client.beta.threads.retrieve(thread_id)
            else:
                thread = self.client.beta.threads.create()
            
            # Send message
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            
            if run.status == 'completed':
                # Get the response
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                
                # Get file ID from message attachments if any
                file_id = None
                if messages.data[0].attachments:
                    file_id = messages.data[0].attachments[0].file_id
                
                if not file_id:
                    # Fallback to checking run steps
                    run_steps = self.client.beta.threads.runs.steps.list(
                        thread_id=thread.id,
                        run_id=run.id
                    )
                    
                    for step in run_steps.data:
                        if step.step_details.type == "tool_calls":
                            for call in step.step_details.tool_calls:
                                if call.type == "code_interpreter" and hasattr(call.code_interpreter, "outputs"):
                                    for output in call.code_interpreter.outputs:
                                        if output.type == "file":
                                            file_id = output.file_id
                                            break
                
                response = {
                    "thread_id": thread.id,
                    "message": messages.data[0].content[0].text.value,
                    "file_id": file_id
                }
                
                return response
            else:
                raise Exception(f"Assistant run failed with status: {run.status}")
                
        except Exception as e:
            logger.error(f"Error in conversation: {str(e)}", exc_info=True)
            raise
    
    def cleanup_resources(self, assistant_id: str, file_ids: List[str]):
        """
        Clean up assistant and file resources.
        
        Args:
            assistant_id: ID of the assistant to delete
            file_ids: List of file IDs to delete
        """
        try:
            # Delete assistant
            self.client.beta.assistants.delete(assistant_id)
            
            # Delete files
            for file_id in file_ids:
                self.client.files.delete(file_id)
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    
    def download_file(self, file_id: str, output_path: str):
        """
        Download a file from the API.
        
        Args:
            file_id: ID of the file to download
            output_path: Path where to save the file
        """
        try:
            file_content = self.client.files.content(file_id)
            
            with open(output_path, "wb") as f:
                f.write(file_content.read())
                
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}", exc_info=True)
            raise 