from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
import json

# API tavily
with open("apis_key.json") as config_file:
    config = json.load(config_file)

search = TavilySearchAPIWrapper(tavily_api_key=config["tavily_api_key"])
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
