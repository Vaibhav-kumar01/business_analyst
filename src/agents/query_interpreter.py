"""
Query Interpreter Agent for translating natural language to analytical requirements.
"""
from src.agents.base_agent import BaseAgent

class QueryInterpreterAgent(BaseAgent):
    """
    Agent that interprets natural language queries and translates them
    to structured analytical requirements.
    """
    
    def __init__(self, role="Query Interpreter", 
                 goal="Translate business questions into data analysis requirements", 
                 backstory="You are an expert in understanding business language and translating it into data analysis requirements.",
                 verbose=True, llm=None):
        """Initialize the Query Interpreter Agent."""
        super().__init__(role, goal, backstory, verbose, llm)
    
    @classmethod
    def from_config(cls, config, llm=None):
        """Create agent from configuration."""
        agent = super().from_config(config, llm)
        return agent