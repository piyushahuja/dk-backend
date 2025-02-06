from typing import Dict, Tuple
import threading

class AssistantManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AssistantManager, cls).__new__(cls)
                cls._instance.file_assistants: Dict[str, Tuple[str, str]] = {}  # file_id -> (assistant_id, thread_id)
            return cls._instance
    
    def get_assistant_and_thread(self, file_id: str) -> Tuple[str, str]:
        """Get existing assistant and thread IDs for a file."""
        return self.file_assistants.get(file_id, (None, None))
    
    def set_assistant_and_thread(self, file_id: str, assistant_id: str, thread_id: str):
        """Store assistant and thread IDs for a file."""
        self.file_assistants[file_id] = (assistant_id, thread_id)
    
    def remove_file(self, file_id: str):
        """Remove file mapping when cleanup is needed."""
        self.file_assistants.pop(file_id, None) 