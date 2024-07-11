from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from personal_search.tools.books import LibraryTool
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

@CrewBase
class PersonalSearchCrew():
	"""PersonalSearch crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=[LibraryTool()],
			verbose=False,
			llm=llm,
    		allow_delegation=False
		)
	
	@agent
	def final_revisor(self) -> Agent:
		return Agent(
			config=self.agents_config['final_revisor'],
			verbose=False,
			llm=llm,
    		allow_delegation=False,
		)

	@task
	def research(self) -> Task:
		return Task(
			config=self.tasks_config['research'],
			agent=self.researcher(),
		)

	@task
	def final_revision(self) -> Task:
		return Task(
			config=self.tasks_config['final_revision'],
			agent=self.final_revisor(),
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the PersonalSearch crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=2,
			llm=llm,
			#full_output=True,
			#manager_agent=manager,
			#process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
