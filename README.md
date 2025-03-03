# Business Analyst

A natural language interface that enables non-technical users to query business data through AI agents.

## Overview

Business Analyst allows non-technical users to ask questions about their data in plain English. The system uses AI agents to:

1. Interpret the business question
2. Generate appropriate data analysis code
3. Execute the code and explain the results in business-friendly language

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/business_analyst.git
cd business_analyst

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
cp .env.example .env
# Edit .env to add your API keys
```

## Configuration

The system is configured using YAML files in the `config` directory:

- `system.yaml`: General system configuration
- `agents.yaml`: Agent definitions and properties
- `tasks.yaml`: Task definitions and prompts

## Usage

### Running with default settings

```bash
python -m src.main --interactive
```

This will load the default dataset (superstore.csv) and enter interactive mode.

### Specifying a different dataset

```bash
python -m src.main --dataset path/to/your/dataset.csv --interactive
```

### Asking a single question

```bash
python -m src.main --question "What are the top 5 products by sales?"
```

## Example Questions

- "What are the top 5 products by sales?"
- "Show me monthly sales trend over time"
- "Which region has the highest profit margin?"
- "What's the average discount by product category?"
- "Who are our top 10 customers by order value?"

## Architecture

The system consists of three main layers:

1. **Core Layer**: Basic data handling and configuration
   - DataManager: Loads and provides datasets
   - SchemaRegistry: Extracts and stores dataset metadata
   - ConfigLoader: Handles YAML configuration

2. **Agent Layer**: Specialized AI agents
   - QueryInterpreterAgent: Translates business questions to analytical requirements
   - CodeGeneratorAgent: Creates pandas code for analysis
   - ResultExplainerAgent: Executes code and explains results

3. **Orchestration Layer**: Workflow management
   - CrewManager: Coordinates agents and tasks using CrewAI

## Customization

You can customize the system by:

1. Editing the YAML configuration files
2. Adding new agent types by extending BaseAgent
3. Creating custom tools for specific analysis needs

## License

MIT