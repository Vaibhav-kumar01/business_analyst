"""
Code Generator Agent for creating pandas code based on analytical requirements.
"""
from src.agents.base_agent import BaseAgent

class CodeGeneratorAgent(BaseAgent):
    """
    Agent that generates pandas code to perform the required analysis.
    """
    
    def __init__(self, role="Code Generator", 
                 goal="Write efficient pandas code to analyze data", 
                 backstory="You are a Python and pandas expert who writes clean, efficient code to analyze business data.",
                 verbose=True, llm=None):
        """Initialize the Code Generator Agent."""
        super().__init__(role, goal, backstory, verbose, llm)
    
    @classmethod
    def from_config(cls, config, llm=None):
        """Create agent from configuration."""
        agent = super().from_config(config, llm)
        return agent