import io
import os
import sys
import shutil
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Dict, Optional, Any, Union


class IsolatedPythonExecutor:
    """
    A class to execute Python code in isolation within a separate directory.
    
    This class provides a sandboxed directory for executing Python code
    with controlled access to files and resources. Note that this is not
    a true isolated environment - it just uses a separate directory.
    """
    
    def __init__(self, isolated_dir: str = "isolated_execution_dir"):
        """
        Initialize the IsolatedPythonExecutor.
        
        Args:
            isolated_dir: Directory name to use for isolated code execution
        """
        self.isolated_dir = isolated_dir
        self.original_dir = os.getcwd()
    
    def execute_code(self, 
                    code: str, 
                    source_dir: str = ".", 
                    files_to_copy: Optional[List[str]] = None,
                    input_data: str = "") -> Dict[str, str]:
        """
        Execute Python code in a separate directory and return the output.
        
        Args:
            code: The Python code to execute
            source_dir: Directory containing files to copy into the isolated directory
            files_to_copy: List of specific filenames to copy from source_dir
            input_data: Optional input data to provide to the executed code
            
        Returns:
            Dictionary containing execution status, output, and any errors
        """
        result = {
            "status": "executed",
            "output": "",
            "errors": ""
        }
        
        try:
            self._setup_isolated_environment(source_dir, files_to_copy)
            result = self._run_code(code, input_data)
        except Exception as e:
            result["status"] = "setup_error"
            result["errors"] = f"Error setting up isolated directory: {str(e)}\n{traceback.format_exc()}"
        finally:
            # Always attempt to restore the original directory
            try:
                os.chdir(self.original_dir)
            except Exception as e:
                if result["errors"]:
                    result["errors"] += f"\nError returning to original directory: {str(e)}"
                else:
                    result["errors"] = f"Error returning to original directory: {str(e)}"
        
        return result
    
    def _setup_isolated_environment(self, source_dir: str, files_to_copy: Optional[List[str]]) -> None:
        """
        Set up the isolated directory.
        
        Args:
            source_dir: Directory containing files to copy
            files_to_copy: List of specific filenames to copy from source_dir
        """
        # Create the isolated directory if it doesn't exist
        if not os.path.exists(self.isolated_dir):
            os.makedirs(self.isolated_dir)
        
        # Copy specified files from source directory
        copied_files = []
        if files_to_copy:
            for file_name in files_to_copy:
                source_file = os.path.join(source_dir, file_name)
                if os.path.isfile(source_file):
                    dest_file = os.path.join(self.isolated_dir, file_name)
                    shutil.copy2(source_file, dest_file)
                    copied_files.append(file_name)
                else:
                    print(f"Warning: File not found or not a file: {file_name}")
        
        if copied_files:
            print(f"Copied files to isolated directory: {', '.join(copied_files)}")
        else:
            print("No files were copied to the isolated directory.")
    
    def _run_code(self, code: str, input_data: str = "") -> Dict[str, str]:
        """
        Run the provided code in the isolated directory.
        
        Args:
            code: The Python code to execute
            input_data: Optional input data to provide to the executed code
            
        Returns:
            Dictionary containing execution status, output, and any errors
        """
        result = {
            "status": "executed",
            "output": "",
            "errors": ""
        }
        
        code_file = os.path.join(self.isolated_dir, "execute_me.py")
        
        # Save the code to execute in the isolated directory
        with open(code_file, "w") as f:
            f.write(code)
        
        # Change to the isolated directory
        os.chdir(self.isolated_dir)
        
        # Prepare for execution
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            # Handle input data if provided
            original_stdin = sys.stdin
            if input_data:
                sys.stdin = io.StringIO(input_data)
            
            # Execute the code in the isolated environment
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                # Create a restricted globals dictionary
                restricted_globals = {
                    "__builtins__": __builtins__,
                    "__name__": "__main__",
                    "__file__": code_file,
                }
                
                with open("execute_me.py", "r") as f:
                    code_to_execute = f.read()
                
                exec(code_to_execute, restricted_globals)
            
            # Restore stdin if needed
            if input_data:
                sys.stdin = original_stdin
            
            # Collect output
            result["output"] = output_buffer.getvalue()
            result["errors"] = error_buffer.getvalue()
            
        except Exception as e:
            result["status"] = "error"
            result["errors"] = f"{str(e)}\n{traceback.format_exc()}"
        
        return result
    
    def cleanup(self) -> bool:
        """
        Clean up by removing the isolated directory.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            if os.path.exists(self.isolated_dir):
                shutil.rmtree(self.isolated_dir)
                print(f"Cleaned up isolated directory: {self.isolated_dir}")
                return True
            return False
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            return False
    
    @staticmethod
    def read_code_file(filename: str) -> str:
        """
        Read code from a file.
        
        Args:
            filename: Path to the file to read
            
        Returns:
            The content of the file as a string
        """
        try:
            with open(filename, "r") as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Initialize the executor
    executor = IsolatedPythonExecutor()
    
    # Source directory containing files to copy into the isolated directory
    source_dir = "."  # Current directory, modify as needed
    target_file = "testing.py"  # File to execute
    files_to_copy = ["testing.py"]  # List of files to copy
    
    # Read code from the target file
    python_code = IsolatedPythonExecutor.read_code_file(target_file)
    
    # Execute the code in isolation
    result = executor.execute_code(python_code, source_dir, files_to_copy)
    
    print("\n=== Isolated Execution Setup ===")
    print(f"Isolated directory: {executor.isolated_dir}")
    print(f"Target file executed: {target_file}")
    print(f"Files copied: {', '.join(files_to_copy)}")
    
    print("\n=== Execution Result ===")
    print(f"Status: {result['status']}")
    print("\n=== Output ===")
    print(result['output'])
    if result['errors']:
        print("\n=== Errors ===")
        print(result['errors'])
    
    # Clean up isolated directory when done
    executor.cleanup()