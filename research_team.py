import functools
import json
import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END

from helper_utilities import create_agent, create_team_supervisor, agent_node
from research_team_tools import search_news


# Definição do Grafo de Linguagem do Time de pesquisa
class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

    team_members: List[str]

    next: str


# Setando API Key da OpenAI
with open("apis_key.json") as config_file:
    config = json.load(config_file)

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=config["openai_api_key"])

search_agent = create_agent(
    llm,
    [search_news],
    f"Você é um especialista em IA. Para me auxiliar na pesquisa, utilize a ferramenta de busca de notícias "
    f"para encontrar os artigos mais recentes relacionados a tecnologias de IA e a utilização de IA na sociedade."
)
search_node = functools.partial(agent_node, agent=search_agent, name="Search")


supervisor_agent = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  Search. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Search"],
)

research_graph = StateGraph(ResearchTeamState)
research_graph.add_node("Search", search_node)
research_graph.add_node("supervisor", supervisor_agent)

# Define the control flow
research_graph.add_edge("Search", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"Search": "Search", "FINISH": END},
)


research_graph.set_entry_point("supervisor")
chain = research_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


research_chain = enter_chain | chain
