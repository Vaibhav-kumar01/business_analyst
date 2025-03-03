"""
Code Execution Tool for safely running pandas code.
"""
from crewai.tools import BaseTool
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd
import io
import sys
from typing import Any, Dict, Optional
from pydantic import Field, PrivateAttr
from contextlib import redirect_stdout

class PandasExecutionTool(BaseTool):
    """Tool for safely executing pandas code."""
    
    name: str = "PandasExecutionTool"
    description: str = "Executes pandas code to analyze the dataset"
    
    # Use private attributes for non-serializable objects
    _df: pd.DataFrame = PrivateAttr()
    _dataset_name: str = PrivateAttr(default="df")
    
    def __init__(self, dataframe: pd.DataFrame, dataset_name: str = "df"):
        """
        Initialize with dataframe to analyze.
        
        Args:
            dataframe (pandas.DataFrame): The dataframe to analyze
            dataset_name (str): Variable name to use for the dataframe
        """
        super().__init__()
        self._df = dataframe
        self._dataset_name = dataset_name
    
    def _run(self, code: str) -> str:
        """
        Execute pandas code safely using LangChain's PythonAstREPLTool.
        
        Args:
            code (str): The pandas code to execute
            
        Returns:
            str: The execution results or error message
        """
        # Clean the code (remove markdown code blocks if present)
        cleaned_code = self._clean_code(code)
        
        # Create a Python REPL tool with the dataframe
        try:
            python_repl = PythonAstREPLTool()
            
            # Inject the dataframe into the global namespace
            python_repl.globals[self._dataset_name] = self._df.copy()
            python_repl.globals["pd"] = pd
            
            # Capture stdout during execution
            stdout_capture = io.StringIO()
            with redirect_stdout(stdout_capture):
                # Execute the code
                result = python_repl.run(cleaned_code)
                
            stdout_output = stdout_capture.getvalue()
            
            # Format the final output including both stdout and result
            if stdout_output and result:
                return f"{stdout_output}\n\nResult:\n{result}"
            elif stdout_output:
                return stdout_output
            else:
                return result
                
        except Exception as e:
            return f"Error executing code: {str(e)}"
    
    def _clean_code(self, code: str) -> str:
        """
        Clean code by removing markdown code blocks and language identifiers.
        
        Args:
            code (str): The code string to clean
            
        Returns:
            str: Cleaned code ready for execution
        """
        # Remove markdown code blocks if present
        if "```" in code:
            # Extract code between code blocks
            parts = code.split("```")
            if len(parts) >= 3:
                code = parts[1]
                # If language identifier is present, remove it
                if "\n" in code and code.split("\n")[0].strip() in ["python", "py"]:
                    code = "\n".join(code.split("\n")[1:])
        
        # Ensure the dataframe reference is correct
        if "df" in code and self._dataset_name != "df":
            # Replace bare "df" references, but be careful not to replace parts of other identifiers
            import re
            code = re.sub(r'\bdf\b', self._dataset_name, code)
        
        return code.strip()