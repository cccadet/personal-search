from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from personal_search.tools.books import LibraryTool, BibleTool
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="""You're an experienced project manager, skilled in overseeing complex projects and 
      guiding teams to success. Your role is to coordinate the efforts of the crew members, 
      ensuring that each task is completed on time and to the highest standard.""",
	verbose=True,
	llm=llm,
	allow_delegation=True,
)

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
			verbose=True,
			llm=llm,
    		allow_delegation=False,
			async_mode=True
		)
	
	@agent
	def bible_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['bible_researcher'],
			tools=[BibleTool()],
			verbose=True,
			llm=llm,
			allow_delegation=False,
			async_mode=True
		)

	@agent
	def final_revisor(self) -> Agent:
		return Agent(
			config=self.agents_config['final_revisor'],
			verbose=True,
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
	def bible_research(self) -> Task:
		return Task(
			config=self.tasks_config['bible_research'],
			agent=self.bible_researcher(),
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
			#process=Process.sequential,
			verbose=2,
			llm=llm,
			manager_agent=manager,
			process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
