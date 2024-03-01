import functools
import json
import operator
from pathlib import Path
from typing import TypedDict, Annotated, List

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from helper_utilities import create_agent, create_team_supervisor, agent_node
from research_team import research_graph
from writing_team_tools import print_results


# Document writing team graph state
class DocWritingState(TypedDict):
    # This tracks the team's conversation internally
    messages: Annotated[List[BaseMessage], operator.add]
    # This provides each worker with context on the others' skill sets
    team_members: str
    # This is how the supervisor tells langgraph who to work next
    next: str
    # This tracks the shared directory state
    current_files: str


# Setando API Key da OpenAI
with open("apis_key.json") as config_file:
    config = json.load(config_file)

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=config["openai_api_key"])

print_agent = create_agent(
    llm,
    [print_results],
    "Você é um ótimo redator capaz de criar resumos autênticos e enxutos a respeito de temáticas de IA.\n"
)
print_agent_node = functools.partial(
    agent_node, agent=print_agent, name="Printer"
)

doc_writing_supervisor = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Printer"],
)

# Create the graph here:
# Note that we have unrolled the loop for the sake of this doc
authoring_graph = StateGraph(DocWritingState)
authoring_graph.add_node("Printer", print_agent_node)
authoring_graph.add_node("supervisor", doc_writing_supervisor)

# Add the edges that always occur
authoring_graph.add_edge("Printer", "supervisor")

# Add the edges where routing applies
authoring_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Printer": "Printer",
        "FINISH": END,
    },
)

authoring_graph.set_entry_point("supervisor")
chain = research_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results


# We re-use the enter/exit functions to wrap the graph
authoring_chain = (
    functools.partial(enter_chain, members=authoring_graph.nodes)
    | authoring_graph.compile()
)
