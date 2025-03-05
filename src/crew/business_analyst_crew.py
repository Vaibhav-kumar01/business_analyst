# src/latest_ai_development/crew.py

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
from src.core.config_loader import ConfigLoader


load_dotenv()
# codeExecutor = PandasExecutionTool()

def setup_llm(config_name="default"):
    # Try to load LLM config
    config_loader = ConfigLoader()
    system_config = config_loader.get_config("system")
    llm_configs = system_config.get("llm", {})
    
    # Get specific LLM config, fall back to default if not found
    llm_config = llm_configs.get(config_name, llm_configs.get("default", {}))
    
    # Check for model type and initialize accordingly
    model_type = llm_config.get("type", "").lower()
    print(f"Setting up LLM for {config_name} with model type: {model_type}")
    print(f"LLM Config: {llm_config}")
    if model_type == "google":
        model=llm_config.get("model", "gemini-pro")
        print(os.environ.get("GOOGLE_API_KEY"))
        print(model)
        return LLM(
            model=llm_config.get("model", "gemini-pro"),
            api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=llm_config.get("temperature", 0)
        )
    else:
        print(f"Unsupported LLM type: {model_type}")


@CrewBase
class BusinessAnalystCrew():
  """Business Analyst crew"""

  @agent
  def query_interpreter(self) -> Agent:
    return Agent(
      config=self.agents_config['query_interpreter'],
      verbose=True,
      llm = setup_llm("query_interpreter")
    )

  @agent
  def code_generator(self) -> Agent:
    return Agent(
      config=self.agents_config['code_generator'],
      verbose=True,
      llm = setup_llm("code_generator")
    )
  
  @agent
  def result_explainer(self) -> Agent:
    return Agent(
      config=self.agents_config['result_explainer'],
      verbose=True,
      llm = setup_llm("result_explainer")
    )

  @task
  def interpret_task(self) -> Task:
    return Task(
            description=f"""
            Analyze the following user question and determine what data analysis needs to be performed.
            Identify relevant columns, filters, groupings, and calculations.
            User Question: {{question}}
            Schema Information:
            {{schema_info}}
            
            Provide your analysis in a structured format that identifies:
            1. Relevant columns to use
            2. Any filters or conditions
            3. Grouping requirements (if any)
            4. Calculations or aggregations needed
            5. Type of result expected (table, single value, etc.)
            """,
            expected_output="A structured analysis of the user's question with specific data requirements.",
            agent = self.query_interpreter()
        )

  @task
  def generate_code_task(self) -> Task:
    return Task(
            description=f"""
            Generate python pandas code to answer the user's question based on the provided analysis.
            User Question: {{question}}
                        
            Schema Information:
            {{schema_info}}
            
            Write clean, efficient pandas code that uses the '{{dataset_name}}' variable for the dataframe.
            Include comments explaining your approach.
            The code should print or return the results.
            Store the final result in a variable named 'result'.
            """,
            expected_output="Python pandas code that answers the user's question.",
            context = [self.interpret_task()],
            agent = self.code_generator()
        )

  @task
  def explain_results_task(self) -> Task:
    return Task(
            description=f"""
            Execute the provided pandas code and explain the results in business-friendly language.
            
            1. Execute the code using your PandasExecutionTool
            2. Format the results in a readable way
            3. Explain the findings in business terms that non-technical users can understand
            4. Highlight any interesting insights or patterns
            
            Your explanation should directly answer the original question.
            """,
            expected_output="Business-friendly explanation of the analysis results.",
            context = [self.interpret_task(), self.generate_code_task()]
        )
  
  @crew
  def crew(self) -> Crew:
    return Crew(
      agents=[
        self.query_interpreter(),
        self.code_generator()
        # self.result_explainer()
      ],
      tasks=[
        self.interpret_task(),
        self.generate_code_task()
        # self.explain_results_task()
      ],
      process=Process.sequential
    )