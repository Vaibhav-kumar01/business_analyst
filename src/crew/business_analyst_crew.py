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
  def data_analyst_agent(self) -> Agent:
    return Agent(
      config=self.agents_config['data_analyst_agent'],
      verbose=True,
      llm = setup_llm("data_analyst_agent")
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
            config = self.tasks_config['interpret_task'],
            agent = self.query_interpreter()
        )

  @task
  def generate_code_task(self) -> Task:
    return Task(
            config = self.tasks_config['generate_code_task'],
            agent = self.data_analyst_agent()
        )

  @task
  def explain_results_task(self) -> Task:
    return Task(
            config = self.tasks_config['explain_results_task'],
            agent = self.result_explainer()
        )
  
  @crew
  def crew(self) -> Crew:
    return Crew(
      agents=[
        self.query_interpreter(),
        self.data_analyst_agent(),
        self.result_explainer()
      ],
      tasks=[
        self.interpret_task(),
        self.generate_code_task(),
        self.explain_results_task()
      ],
      process=Process.sequential
    )