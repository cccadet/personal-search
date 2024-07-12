from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from personal_search.tools.books import LibraryTool
#llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
llm = ChatOpenAI(model='gpt-4o', temperature=0)

from pydantic import BaseModel, Field
from typing import List

class TopicoNaBiblia(BaseModel):
	"""Modelo de Tópico na Bíblia"""
	titulo: str = Field(..., description="Título do tópico")
	encontrado: bool = Field(..., description="O tópico foi encontrado na Bíblia?")
	versiculo_principal: str = Field(..., description="Versículo principal do tópico")
	versiculos_relacionados: List[str] = Field(..., description="Versículos relacionados ao tópico")
	como_tratado: str = Field(..., description="Como o tópico é tratado na Bíblia")
	fontes: List[str] = Field(..., description="Fontes do tópico")

class ContextoHistorico(BaseModel):
	"""Modelo de Contexto Histórico"""
	titulo: str = Field(..., description="Título do tópico")
	periodo_historico: str = Field(..., description="Período histórico do tópico")
	localizacao_geografica: str = Field(..., description="Localização geográfica do tópico")
	contexto_politico: str = Field(..., description="Contexto político do tópico")
	contexto_social: str = Field(..., description="Contexto social do tópico")
	contexto_religioso: str = Field(..., description="Contexto religioso do tópico")
	fontes: List[str] = Field(..., description="Fontes do tópico")

class AnaliseHermeneutica(BaseModel):
	"""Modelo de Análise Hermenêutica"""
	titulo: str = Field(..., description="Título do tópico")
	hermeneutica: str = Field(..., description="Análise hermenêutica do tópico")
	interpretacoes: List[str] = Field(..., description="Principais interpretações do texto")
	dificuldades_interpretacao: str = Field(..., description="Dificuldades de interpretação do tópico")
	interpretacoes_linguas_originais: List[str] = Field(..., description="Principais análises das línguas originais")
	fontes: List[str] = Field(..., description="Fontes do tópico")
	

class AnaliseTeologica(BaseModel):
	"""Modelo de Anaĺise Teológica"""
	titulo: str = Field(..., description="Título do tópico")
	classificacao: str = Field(..., description="Classificação do tópico (Texto Bíblico, Tema, Dourtina, etc)")
	topico_na_biblia: List[str] = Field(..., description="Explica como o tópico é tratado na Bíblia")
	contexto_historico: str = Field(..., description="Contexto histórico completo do tópico")
	analise_hermeneutica: str = Field(..., description="Como o tópico é interpretado hermeneuticamente")
	aplicacoes_praticas: List[str] = Field(..., description="Quais as aplicações práticas do tópico")
	fontes: List[str] = Field(..., description="Quais as fontes utilizadas na análise")

@CrewBase
class PersonalSearchCrew():
	"""PersonalSearch crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def biblical_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['biblical_researcher'],
			tools=[SerperDevTool()],
			verbose=True,
			llm=llm,
    		allow_delegation=False,
			memory=False
		)
	
	@agent
	def historian(self) -> Agent:
		return Agent(
			config=self.agents_config['historian'],
			tools=[SerperDevTool()],
			verbose=True,
			llm=llm,
			allow_delegation=False,
			memory=False
		)
	
	@agent
	def hermeneutics_expert(self) -> Agent:
		return Agent(
			config=self.agents_config['hermeneutics_expert'],
			tools=[SerperDevTool()],
			verbose=True,
			llm=llm,
			allow_delegation=False,
			memory=False
		)
	
	@agent
	def revisor(self) -> Agent:
		return Agent(
			config=self.agents_config['revisor'],
			verbose=True,
			llm=llm,
    		allow_delegation=False,
			memory=False
		)

	@task
	def topic_in_bible(self) -> Task:
		return Task(
			config=self.tasks_config['topic_in_bible'],
			agent=self.biblical_researcher(),
			output_pydantic=TopicoNaBiblia
		)
	
	@task
	def historical_context(self) -> Task:
		return Task(
			config=self.tasks_config['historical_context'],
			agent=self.historian(),
			output_pydantic=ContextoHistorico
		)
	
	@task
	def hermeneutical_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['hermeneutical_analysis'],
			agent=self.hermeneutics_expert(),
			output_pydantic=AnaliseHermeneutica
		)
	
	@task
	def revision(self) -> Task:
		return Task(
			config=self.tasks_config['revision'],
			agent=self.revisor(),
			context=[self.topic_in_bible(), self.historical_context(), self.hermeneutical_analysis()],
			output_pydantic=AnaliseTeologica
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the PersonalSearch crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			llm=llm,
			full_output=True,
			#manager_agent=manager,
			#process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
