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

# Load environment variables
load_dotenv()

def setup_llm(config_name="default"):
    """
    Set up the language model based on configuration.
    
    Args:
        config_name (str): Name of the LLM configuration to use
    
    Returns:
        The configured language model
    """
    # Try to load LLM config
    config_loader = ConfigLoader()
    system_config = config_loader.get_config("system")
    llm_configs = system_config.get("llm", {})
    
    # Get specific LLM config, fall back to default if not found
    llm_config = llm_configs.get(config_name, llm_configs.get("default", {}))
    
    # Check for model type and initialize accordingly
    model_type = llm_config.get("type", "").lower()
    
    if model_type == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=llm_config.get("model", "gemini-pro"),
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                temperature=llm_config.get("temperature", 0)
            )
        except ImportError:
            print("Google Gemini not available. Install with: pip install langchain-google-genai")
    
    elif model_type == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=llm_config.get("model", "gpt-3.5-turbo"),
                temperature=llm_config.get("temperature", 0),
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                max_tokens=llm_config.get("max_tokens", 1000)
            )
        except ImportError:
            print("OpenAI not available. Install with: pip install langchain-openai")
    
    # elif model_type == "anthropic":
    #     try:
    #         from langchain_anthropic import ChatAnthropic
    #         return ChatAnthropic(
    #             model=llm_config.get("model", "claude-2"),
    #             anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    #             temperature=llm_config.get("temperature", 0)
    #         )
    #     except ImportError:
    #         print("Anthropic not available. Install with: pip install langchain-anthropic")
    
    # Default to OpenAI if no valid type specified
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
    except ImportError:
        print("Warning: No LLM available. Please install a supported LLM package.")
        return None

def setup_agent_llms():
    """
    Set up language models for each agent.
    
    Returns:
        dict: Dictionary with LLM instances for each agent
    """
    return {
        "query_interpreter": setup_llm("query_interpreter"),
        "code_generator": setup_llm("code_generator"),
        "result_explainer": setup_llm("result_explainer"),
        "default": setup_llm("default")
    }

def interactive_mode(data_manager, schema_registry, agent_configs):
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
    
    # Create crew manager
    crew_manager = CrewManager(df, schema_info, None, dataset_name)
    crew_manager.create_crew(agent_configs)
    
    # Main loop
    while True:
        question = input("\nYour question: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        if not question.strip():
            continue
        
        print("\nProcessing your question...")
        try:
            result = crew_manager.process_query(question)
            print("\nAnswer:")
            print(result)
        except Exception as e:
            print(f"Error processing query: {e}")

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
    
    # Set up LLMs for each agent
    llms = setup_agent_llms()
    
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
    
    # Create agent configs with specified LLMs
    agent_configs = {
        "query_interpreter": {"llm": llms.get("query_interpreter", llms["default"])},
        "code_generator": {"llm": llms.get("code_generator", llms["default"])},
        "result_explainer": {"llm": llms.get("result_explainer", llms["default"])}
    }
    
    # Process single question or enter interactive mode
    if args.question:
        crew_manager = CrewManager(df, schema_info, None, dataset_name)
        # Pass the agent configs to create_crew
        crew_manager.create_crew(agent_configs)
        result = crew_manager.process_query(args.question)
        print(result)
    elif args.interactive or not args.question:
        interactive_mode(data_manager, schema_registry, agent_configs)

if __name__ == "__main__":
    main()