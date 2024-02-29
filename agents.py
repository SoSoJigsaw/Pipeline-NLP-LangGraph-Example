import langchain.agents
import langchain.utils
from langchain.agents import Message


# Agente Supervisor
class SupervisorAgent(langchain.agents.SupervisedAgent):

    def __init__(self, langgraph, prompts, research_agents, writing_agent):
        super().__init__(langgraph, prompts)
        self.research_agents = research_agents
        self.writing_agent = writing_agent
        self.results = {}

    def on_message(self, message):
        if message.type == "research_results":
            self.results[message.data["topic"]] = message.data["results"]
            if len(self.results) == len(self.research_agents):
                self.send_message(Message(type="writing_task", data=self.results))
        elif message.type == "writing_complete":
            self.results = {}


# Agente de Pesquisa
class ResearchAgent(langchain.agents.ChainAgent):

    def __init__(self, langgraph, prompts):
        super().__init__(langgraph, prompts)

    def on_message(self, message):
        if message.type == "research_task":
            results = self.run_chain(message.data["topic"])
            self.send_message(Message(type="research_results", data={"topic": message.data["topic"], "results": results}))


# Agente de escrita
class WritingAgent(langchain.agents.ChainAgent):

    def __init__(self, langgraph, prompts):
        super().__init__(langgraph, prompts)

    def on_message(self, message):
        if message.type == "writing_task":
            document = self.run_chain(message.data)
            self.send_message(Message(type="writing_complete", data={"document": document}))

