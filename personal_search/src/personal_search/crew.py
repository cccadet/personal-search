from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
from personal_search.tools.custom_tool import LibraryTool, BibleTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class PersonalSearchCrew():
	"""PersonalSearch crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=[LibraryTool(),BibleTool()],
			verbose=True,
			llm=llm,
    		allow_delegation=False,
		)

	@agent
	def revisor(self) -> Agent:
		return Agent(
			config=self.agents_config['revisor'],
			verbose=True,
			llm=llm,
    		allow_delegation=False,
		)
	
	@agent
	def commentary_expert(self) -> Agent:
		return Agent(
			config=self.agents_config['commentary_expert'],
			verbose=True,
			llm=llm,
    		allow_delegation=False,
		)
	
	@agent
	def bible_expert(self) -> Agent:
		return Agent(
			config=self.agents_config['bible_expert'],
			verbose=True,
			tools=[BibleTool()],
			llm=llm,
    		allow_delegation=False,
		)
	
	@agent
	def final_revisor(self) -> Agent:
		return Agent(
			config=self.agents_config['final_revisor'],
			verbose=True,
			llm=llm,
    		allow_delegation=False,
			output_file='Final.md'
		)

	@task
	def library_search(self) -> Task:
		return Task(
			config=self.tasks_config['library_search'],
			agent=self.librarian(),
		)

	@task
	def library_select(self) -> Task:
		return Task(
			config=self.tasks_config['library_select'],
			agent=self.revisor(),
		)
	
	@task
	def literary_commentary(self) -> Task:
		return Task(
			config=self.tasks_config['literary_commentary'],
			agent=self.commentary_expert(),
		)
	
	@task
	def bible_search(self) -> Task:
		return Task(
			config=self.tasks_config['bible_search'],
			agent=self.bible_expert(),
		)
	
	@task
	def final_revision(self) -> Task:
		return Task(
			config=self.tasks_config['final_revision'],
			agent=self.final_revisor(),
			contexts=[self.library_select(), self.literary_commentary(), self.bible_search()]
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the PersonalSearch crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=2,
			llm=llm
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
