"""
Base Agent class for common agent functionality.
"""
from crewai import Agent

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, role, goal, backstory, verbose=True, llm=None):
        """
        Initialize base agent with common properties.
        
        Args:
            role (str): Agent's role
            goal (str): Agent's goal
            backstory (str): Agent's backstory
            verbose (bool): Whether to display verbose output
            llm: Language model to use (optional)
        """
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose
        self.llm = llm
        self.tools = []
    
    def add_tool(self, tool):
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
    
    def create_agent(self):
        """
        Create the actual CrewAI agent.
        
        Returns:
            crewai.Agent: The created agent
        """
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=self.verbose,
            llm=self.llm,
            tools=self.tools
        )
    
    @classmethod
    def from_config(cls, config, llm=None):
        """
        Create agent from configuration.
        
        Args:
            config (dict): Agent configuration
            llm: Language model to use (optional)
            
        Returns:
            BaseAgent: Instantiated agent
        """
        return cls(
            role=config.get("role", "Agent"),
            goal=config.get("goal", "Complete tasks"),
            backstory=config.get("backstory", ""),
            verbose=config.get("verbose", True),
            llm=llm
        )