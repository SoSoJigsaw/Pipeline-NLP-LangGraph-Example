import json
import time
from langchain.agents import Message
import langchain.chains.langgraph
import langchain.utils
from newsapi import NewsApiClient
from agents import SupervisorAgent, ResearchAgent, WritingAgent


# Criação da equipe, criando instâncias para cada uma das classes definidas para os Agentes,
# depois a execução da equipe

# Cria o grafo de linguagem
langgraph = langchain.chains.langgraph.LangGraph()

# Cria os prompts
prompts = {
    "research": "Pesquise artigos de notícias sobre {topic} e extraia as principais informações.",
    "writing": "Gerar um resumo dos artigos de notícias sobre {topics}."
}

# Cria os agentes
supervisor_agent = SupervisorAgent(langgraph, prompts)
research_agents = [ResearchAgent(langgraph, prompts) for _ in range(2)]
writing_agent = WritingAgent(langgraph, prompts)

# Define os agentes de pesquisa e o agente de escrita no agente supervisor
supervisor_agent.research_agents = research_agents
supervisor_agent.writing_agent = writing_agent

# API de notícias
with open("news_api_key.json") as config_file:
    config = json.load(config_file)

newsapi = NewsApiClient(api_key=config["news_api_key"])

# Define o tópico da pesquisa
topic = "Inteligência Artificial"

# Envia a tarefa de pesquisa para o supervisor
supervisor_agent.send_message(Message(type="research_task", data={"topic": topic}))

# Aguarda a finalização da tarefa
while not supervisor_agent.is_finished():
    time.sleep(1)

# Obtém o documento final
document = supervisor_agent.results["writing_complete"].data

# Imprime o documento
print(document)
