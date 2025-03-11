"""
Service layer for business data analysis.
Coordinates data loading, schema management, and analysis execution.
"""
import os
import sys
import traceback
from src.core.data_manager import DataManager
from src.core.schema_registry import SchemaRegistry
from src.crew.business_analyst_crew import BusinessAnalystCrew

class BusinessAnalystService:
    """Service for handling business data analysis"""
    
    def __init__(self):
        """Initialize with core components"""
        self.data_manager = DataManager()
        self.schema_registry = SchemaRegistry()
        
        # Load default dataset if available
        self._initialize_default_dataset()
    
    def _initialize_default_dataset(self):
        """Initialize the default dataset if available"""
        default_dataset_path = os.path.join("data", "superstore.csv")
        if os.path.exists(default_dataset_path):
            self.data_manager.load_dataset(default_dataset_path)
    
    def list_datasets(self):
        """Get list of available datasets"""
        return self.data_manager.list_datasets()
    
    def load_dataset(self, dataset_path):
        """Load a new dataset"""
        return self.data_manager.load_dataset(dataset_path)
    
    def analyze_query(self, query, dataset_name=None):
        """
        Analyze a business query using the specified dataset
        
        Args:
            query: The natural language query to analyze
            dataset_name: Name of the dataset to use (uses first available if None)
            
        Returns:
            Analysis results or error message
        """
        try:
            # Get available datasets
            datasets = self.data_manager.list_datasets()
            if not datasets:
                raise ValueError("No datasets available. Please load a dataset first.")
            
            # Use specified dataset or default to first available
            if dataset_name is None:
                dataset_name = datasets[0]
            elif dataset_name not in datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            # Get dataset and schema
            df = self.data_manager.get_dataset(dataset_name)
            schema_info = self.schema_registry.format_schema_for_llm(dataset_name)
            
            # Run the analysis using CrewAI
            try:
                crew = BusinessAnalystCrew()
                result = crew.crew().kickoff(inputs={
                    "question": query, 
                    "dataset_name": dataset_name, 
                    "schema_info": schema_info
                })
                
                # Check if result is empty or None, which indicates an LLM failure
                if result is None or result == "":
                    return "Analysis failed: The AI model couldn't generate a response. This might be due to complexity of the query or a temporary issue with the AI service. Please try again with a simpler query or try later."
                
                return result
            except Exception as crew_error:
                error_details = traceback.format_exc()
                print(f"Error in CrewAI execution: {str(crew_error)}")
                print(error_details)
                
                # Provide a more helpful error message with details from the code execution
                # Find any Python output from the code execution in the error traceback
                python_output = None
                for line in error_details.split('\n'):
                    if "Tool Output:" in line:
                        python_output = line.split("Tool Output:", 1)[1].strip()
                        break
                
                error_message = f"Analysis failed: Error in AI model execution.\n\n"
                
                if python_output:
                    error_message += f"The code generated did run and produced this output:\n{python_output}\n\n"
                
                error_message += "However, the AI couldn't complete the analysis after examining the data. Please try a different or simpler query."
                
                return error_message
                
        except Exception as e:
            print(f"Error in business analysis service: {str(e)}")
            print(traceback.format_exc())
            return f"Error: {str(e)}"