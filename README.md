[Procurando a versão em português? Clique aqui!](#personalsearch-crew---pt-br)

# PersonalSearch Crew

This is a study project. The goal is to create a personal search system that analyzes biblical texts and returns a report with relevant information.

This report will be divided into sections, such as:

- Topic in the Bible: Performs a search on the biblical text, related texts, etc.

- Topic in books: Searches on the topic in books loaded in the book database.

- Historical Context: Searches on the historical context, location, political and social context, etc.

- Hermeneutic Analysis: Performs a hermeneutic analysis, with main interpretations, difficulties and analysis of original languages.

- Sources

## Prerequisites

- **Ollama:** Download the Ollama installer from [Ollama Download](https://ollama.com/download) and follow the installation instructions.

- **nomic-embed-text:v1.5:** Use the `ollama pull nomic-embed-text:v1.5` command to download the text embedment model.

## Installation

Make sure you have Python >=3.10 <=3.13 installed on your system. This project uses [Poetry](https://python-poetry.org/) for dependency management and package handling, providing a seamless setup and runtime experience.

First, if you haven't already done so, install Poetry:

```bash
pip install poetry
```

Then navigate to your project directory and install the dependencies:

1. First lock the dependencies and then install them:
```bash
poetry lock
```
```bash
poetry install
```

### Configuration

1. Based on the `.env_example` file, create a `.env` file and add your OpenAI API key.

2. By default, the OPENAI_MODEL_NAME variable uses the 'gpt-4o-mini' model. If you want to use another model, change the value of this variable in the `.env` file.

3. Enter the SERPER_API_KEY in the `.env` file to use the Serper API.

4. Text embedding is done locally. If you want to use an embedding service, enter the LOCAL_EMBEDDINGS_URL and LOCAL_EMBEDDINGS_MODEL variables in the `.env` file. By default, embedding is done locally with the **nomic-embed-text:v1.5** model via Ollama.
5. Enter the VECTOR_STORE variable in the `.env` file to use Chroma's vector store.
6. Enter the PATH_LIBRARY variable in the `.env` file with the path to the books you want to load into the database.
7. Enter the YOUTUBE_CHANNELS variable in the `.env` file with the YouTube channels you want to search.
8. Enter the YOUTUBE_VIDEOS variable in the `.env` file with the YouTube videos you want to search.

## Importing data

To import the books data, run the following command in the root folder of your project:

```bash
python ingestion/books.py
```

To import the YouTube video data, run the following command in the root folder of your project:

```bash
python ingestion/youtube_video.py
```

To import the YouTube channel data, run the following command in the root folder of your project:

```bash
python ingestion/youtube_channel.py
```

## Running the project

To kickstart your AI agent team and start executing tasks, run this in the root folder of your project:

```bash
poetry run personal_search
```

This command initializes the personal-search team, setting up the agents and assigning them tasks as defined in your configuration.

This example, without modification, will run the `report.md` file with the output of a search for LLMs in the root folder.

## Understanding your team

The personal search team is comprised of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective abilities to achieve complex goals. The `config/agents.yaml` file describes the capabilities and configurations of each agent in your team.

<br><br>

# PersonalSearch Crew - Pt-br

Esse é um projeto de estudo. O objetivo é criar um sistema de busca pessoal que analise textos bíblicos e retorne um relatório com informações relevantes.

Esse relatório será dividido em seções, como: 
- Tópico na Bíblia: Realiza uma pesquisa sobre o texto bíblico, textos relacionados, etc.
- Tópico em livros: Busca sobre o tema em livros carregados na base de dados de livros.
- Contexto Histórico: Busca sobre o contexto histórico, localização, contexto político e social, etc.
- Análise Hermenêutica: Realiza uma análise hermenêutica, com principais interpretações, dificuldades e análise de línguas originais.
- Fontes

## Pré-requisitos

- **Ollama:** Baixe o instalador do Ollama em [Ollama Download](https://ollama.com/download) e siga as instruções de instalação.
- **nomic-embed-text:v1.5:** Use o comando `ollama pull nomic-embed-text:v1.5` para baixar o modelo de embbeding de texto.

## Instalação

Certifique-se de ter o Python >=3.10 <=3.13 instalado no seu sistema. Este projeto usa [Poetry](https://python-poetry.org/) para gerenciamento de dependências e manipulação de pacotes, oferecendo uma experiência de configuração e execução perfeita.

Primeiro, se você ainda não tiver feito isso, instale o Poetry:

```bash
pip install poetry
```

Em seguida, navegue até o diretório do seu projeto e instale as dependências:

1. Primeiro bloqueie as dependências e depois instale-as:
```bash
poetry lock
```
```bash
poetry install
```

### Configurações

1. Com base no arquivo `.env_example`, crie um arquivo `.env` e adicione sua chave de API do OpenAI.
2. Por padrão, a variável OPENAI_MODEL_NAME usa o modelo 'gpt-4o-mini'. Caso deseje usar outro modelo, altere o valor dessa variável no arquivo `.env`.
3. Informe a SERPER_API_KEY no arquivo `.env` para utilizar a API do Serper.
4. O embbeding de texto é feito localmente. Caso deseje usar um serviço de embbeding, informe a variável LOCAL_EMBEDDINGS_URL e LOCAL_EMBEDDINGS_MODEL no arquivo `.env`. Por padrão, o embbeding é feito localmente com o modelo **nomic-embed-text:v1.5** via Ollama.
5. Informe a variável VECTOR_STORE no arquivo `.env` para utilizar o vector store do Chroma.
6. Informe a variável PATH_LIBRARY no arquivo `.env` com o caminho dos livros que deseja carregar na base de dados.
7. Informe a variável YOUTUBE_CHANNELS no arquivo `.env` com os canais do YouTube que deseja pesquisar.
8. Informe a variável YOUTUBE_VIDEOS no arquivo `.env` com os vídeos do YouTube que deseja pesquisar.

## Importar dados

Para importar os dados dos livros, execute o comando abaixo na raiz do projeto:

```bash
python ingestion/books.py
```

Para importar os dados dos vídeos do YouTube, execute o comando abaixo na raiz do projeto:

```bash
python ingestion/youtube_video.py
```

Para importar os dados dos canais do YouTube, execute o comando abaixo na raiz do projeto:

```bash
python ingestion/youtube_channel.py
```

## Executando o projeto

Para dar o pontapé inicial na sua equipe de agentes de IA e começar a execução de tarefas, execute isto na pasta raiz do seu projeto:

```bash
poetry run personal_search
```

Este comando inicializa a equipe personal-search, montando os agentes e atribuindo a eles tarefas conforme definido na sua configuração.

Este exemplo, sem modificações, executará o arquivo `report.md` com a saída de uma pesquisa sobre LLMs na pasta raiz.

## Entendendo sua equipe

A equipe de busca pessoal é composta por vários agentes de IA, cada um com funções, objetivos e ferramentas exclusivos. Esses agentes colaboram em uma série de tarefas, definidas em `config/tasks.yaml`, alavancando suas habilidades coletivas para atingir objetivos complexos. O arquivo `config/agents.yaml` descreve os recursos e configurações de cada agente em sua equipe.