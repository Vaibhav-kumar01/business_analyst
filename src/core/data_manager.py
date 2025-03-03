"""
Data Manager for handling dataset loading and access.
"""
import pandas as pd

class DataManager:
    """Handles loading and accessing datasets."""
    
    def __init__(self):
        """Initialize with empty dataset storage."""
        self.datasets = {}
    
    def load_dataset(self, dataset_path, dataset_name=None):
        """
        Load a dataset from a file path.
        
        Args:
            dataset_path (str): Path to the dataset file
            dataset_name (str, optional): Name to refer to the dataset.
                                         If None, uses filename as name.
        
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        if dataset_name is None:
            # Extract filename without extension as dataset name
            import os
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        
        # Simple CSV loading for now
        try:
            df = pd.read_csv(dataset_path)
            self.datasets[dataset_name] = df
            print(f"Loaded dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def get_dataset(self, dataset_name):
        """
        Retrieve a dataset by name.
        
        Args:
            dataset_name (str): Name of the dataset to retrieve
            
        Returns:
            pandas.DataFrame or None: The dataset if found, None otherwise
        """
        return self.datasets.get(dataset_name)
    
    def list_datasets(self):
        """
        List all available datasets.
        
        Returns:
            list: Names of available datasets
        """
        return list(self.datasets.keys())