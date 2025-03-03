"""
Crew Manager for orchestrating the workflow of agents.
"""
from crewai import Crew, Task
from src.agents.query_interpreter import QueryInterpreterAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.result_explainer import ResultExplainerAgent
from src.tools.code_execution import PandasExecutionTool

class CrewManager:
    """Manages the CrewAI workflow and agent interactions."""
    
    def __init__(self, dataframe, schema_info, llm=None, dataset_name="df"):
        """
        Initialize with required data.
        
        Args:
            dataframe: Dataset to analyze
            schema_info: Schema information for the dataset
            llm: Language model to use
            dataset_name: Name to use for the dataframe variable
        """
        self.df = dataframe
        self.schema_info = schema_info
        self.llm = llm
        self.dataset_name = dataset_name
        
        # Initialize agents
        self.query_interpreter = None
        self.code_generator = None
        self.result_explainer = None
        
        # Initialize crew
        self.crew = None
    
    def create_crew(self, agent_configs=None):
        """
        Create the crew with agents and tasks.
        
        Args:
            agent_configs: Optional configurations for agents
        """
        # Create agents with default configurations if none provided
        self._create_agents(agent_configs)
        
        # Create execution tool
        execution_tool = PandasExecutionTool(self.df, self.dataset_name)
        
        # Add tool to result explainer agent
        self.result_explainer.add_tool(execution_tool)
        
        # Create CrewAI agents
        query_agent = self.query_interpreter.create_agent()
        code_agent = self.code_generator.create_agent()
        result_agent = self.result_explainer.create_agent()
        
        # Create tasks
        interpret_task = Task(
            description=f"""
            Analyze the following user question and determine what data analysis needs to be performed.
            Identify relevant columns, filters, groupings, and calculations.
            
            Schema Information:
            {self.schema_info}
            
            User Question: {{question}}
            
            Provide your analysis in a structured format that identifies:
            1. Relevant columns to use
            2. Any filters or conditions
            3. Grouping requirements (if any)
            4. Calculations or aggregations needed
            5. Type of result expected (table, single value, etc.)
            """,
            agent=query_agent,
            expected_output="A structured analysis of the user's question with specific data requirements."
        )
        
        generate_code_task = Task(
            description=f"""
            Generate python pandas code to answer the user's question based on the provided analysis.
            
            Schema Information:
            {self.schema_info}
            
            User Question: {{question}}
            Analysis: {{analysis}}
            
            Write clean, efficient pandas code that uses the '{self.dataset_name}' variable for the dataframe.
            Include comments explaining your approach.
            The code should print or return the results.
            Store the final result in a variable named 'result'.
            """,
            agent=code_agent,
            expected_output="Python pandas code that answers the user's question.",
            context=[interpret_task]
        )
        
        explain_results_task = Task(
            description=f"""
            Execute the provided pandas code and explain the results in business-friendly language.
            
            User Question: {{question}}
            Analysis: {{analysis}}
            Code: {{code}}
            
            1. Execute the code using your PandasExecutionTool
            2. Format the results in a readable way
            3. Explain the findings in business terms that non-technical users can understand
            4. Highlight any interesting insights or patterns
            
            Your explanation should directly answer the original question.
            """,
            agent=result_agent,
            expected_output="Business-friendly explanation of the analysis results.",
            context=[interpret_task, generate_code_task]
        )
        
        # Create crew
        self.crew = Crew(
            agents=[query_agent, code_agent, result_agent],
            tasks=[interpret_task, generate_code_task, explain_results_task],
            verbose=True
        )
        
        return self.crew
    
    def process_query(self, question):
        """
        Process a user query through the crew workflow.
        
        Args:
            question (str): The user's natural language question
            
        Returns:
            str: The answer to the user's question
        """
        if not self.crew:
            self.create_crew()
        
        # Execute the crew process
        result = self.crew.kickoff(inputs={"question": question})
        
        return result
    
    def _create_agents(self, configs=None):
        """
        Create agents with default or provided configurations.
        
        Args:
            configs (dict, optional): Configurations for agents
        """
        # If no configs provided, use defaults
        if configs is None:
            configs = {}
        
        # Get configs for each agent
        query_config = configs.get("query_interpreter", {})
        code_config = configs.get("code_generator", {})
        result_config = configs.get("result_explainer", {})
        
        # Create agents with their specific LLMs
        self.query_interpreter = QueryInterpreterAgent.from_config(
            query_config,
            query_config.get("llm", self.llm)  # Use agent-specific LLM if provided
        )
        
        self.code_generator = CodeGeneratorAgent.from_config(
            code_config,
            code_config.get("llm", self.llm)  # Use agent-specific LLM if provided
        )
        
        self.result_explainer = ResultExplainerAgent.from_config(
            result_config,
            result_config.get("llm", self.llm)  # Use agent-specific LLM if provided
        )