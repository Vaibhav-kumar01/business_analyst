"""
Main entry point for the Business Analyst application.
"""
import argparse
import os
from dotenv import load_dotenv
from src.core.data_manager import DataManager
from src.core.schema_registry import SchemaRegistry
from src.core.config_loader import ConfigLoader
from src.orchestration.crew_manager import CrewManager
from src.crew.business_analyst_crew import BusinessAnalystCrew

# Load environment variables
load_dotenv()
   

def interactive_mode(data_manager, schema_registry):
    """
    Run in interactive mode, processing queries from user input.
    
    Args:
        data_manager: The data manager instance
        schema_registry: The schema registry instance
        agent_configs: LLM configurations for each agent
    """
    print("Welcome to Business Analyst!")
    print("Enter your business questions (or 'exit' to quit)")
    
    # Get available datasets
    datasets = data_manager.list_datasets()
    if not datasets:
        print("No datasets available. Please load a dataset first.")
        return
    
    # Select dataset to use
    dataset_name = datasets[0]  # Use first dataset by default
    if len(datasets) > 1:
        print("\nAvailable datasets:")
        for i, ds in enumerate(datasets):
            print(f"{i+1}. {ds}")
        selection = input("Select dataset number (or press Enter for first dataset): ")
        if selection.strip() and selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(datasets):
                dataset_name = datasets[idx]
    
    print(f"\nUsing dataset: {dataset_name}")
    
    # Get dataset and schema
    df = data_manager.get_dataset(dataset_name)
    schema_info = schema_registry.format_schema_for_llm(dataset_name)
    
    
    # Main loop
    while True:
        question = input("\nYour question: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        if not question.strip():
            continue
        
        print("\nProcessing your question...")
        result = BusinessAnalystCrew().crew().kickoff(inputs={"question": question, "dataset_name": dataset_name, "schema_info": schema_info})
        print("\nAnswer:")
        print(result)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Business Analyst - Natural language business data analysis")
    parser.add_argument("--dataset", help="Path to dataset file")
    parser.add_argument("--question", help="Business question to analyze")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    # Initialize core components
    data_manager = DataManager()
    schema_registry = SchemaRegistry()
    
    
    # Load dataset
    dataset_path = args.dataset or os.path.join("data", "superstore.csv")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    df = data_manager.load_dataset(dataset_path)
    if df is None:
        print("Failed to load dataset.")
        return
    
    # Register schema
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    schema = schema_registry.register_schema(dataset_name, df)
    schema_info = schema_registry.format_schema_for_llm(dataset_name)
    
    
    # Process single question or enter interactive mode
    if args.question:
        print("Not supported yet")
    elif args.interactive or not args.question:
        interactive_mode(data_manager, schema_registry)

if __name__ == "__main__":
    main()