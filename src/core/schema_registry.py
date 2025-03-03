"""
Schema Registry for storing and retrieving dataset metadata.
"""

class SchemaRegistry:
    """Manages dataset schema information."""
    
    def __init__(self):
        """Initialize with empty schema storage."""
        self.schemas = {}
    
    def register_schema(self, dataset_name, dataframe):
        """
        Extract and store basic schema information from dataframe.
        
        Args:
            dataset_name (str): Name of the dataset
            dataframe (pandas.DataFrame): The dataset to extract schema from
            
        Returns:
            dict: The extracted schema information
        """
        schema = {
            "table_name": dataset_name,
            "row_count": len(dataframe),
            "columns": []
        }
        
        # Get basic column information
        for col in dataframe.columns:
            col_info = {
                "name": col,
                "data_type": str(dataframe[col].dtype),
                "sample_values": dataframe[col].head(3).tolist()
            }
            schema["columns"].append(col_info)
        
        self.schemas[dataset_name] = schema
        return schema
    
    def get_schema(self, dataset_name):
        """
        Retrieve schema for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            dict or None: The schema if found, None otherwise
        """
        return self.schemas.get(dataset_name)
    
    def format_schema_for_llm(self, dataset_name):
        """
        Format schema information for LLM context.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            str: Formatted schema information
        """
        schema = self.get_schema(dataset_name)
        if not schema:
            return "Schema not found."
        
        formatted = f"Dataset: {schema['table_name']}\n"
        formatted += f"Total Rows: {schema['row_count']}\n\n"
        formatted += "Columns:\n"
        
        for col in schema["columns"]:
            sample_str = ", ".join(str(v) for v in col["sample_values"])
            formatted += f"- {col['name']} ({col['data_type']}): {sample_str}\n"
        
        return formatted