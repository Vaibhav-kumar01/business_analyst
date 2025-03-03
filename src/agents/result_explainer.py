"""
Result Explainer Agent for executing code and explaining results.
"""
from src.agents.base_agent import BaseAgent

class ResultExplainerAgent(BaseAgent):
    """
    Agent that executes pandas code and explains results in business terms.
    """
    
    def __init__(self, role="Result Explainer", 
                 goal="Execute data analysis code and explain results in business-friendly language", 
                 backstory="You are skilled at communicating technical findings to non-technical business users.",
                 verbose=True, llm=None):
        """Initialize the Result Explainer Agent."""
        super().__init__(role, goal, backstory, verbose, llm)
    
    @classmethod
    def from_config(cls, config, llm=None):
        """Create agent from configuration."""
        agent = super().from_config(config, llm)
        return agent