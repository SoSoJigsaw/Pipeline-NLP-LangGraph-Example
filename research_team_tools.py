import json
from typing import Annotated, List, Tuple, Union
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from newsapi import NewsApiClient

# API de notícias
with open("apis_key.json") as config_file:
    config = json.load(config_file)


def search_news():
    """Busca notícias relacionadas ao tópico usando a API do NewsAPI."""
    newsapi = NewsApiClient(api_key=config["news_api_key"])
    articles = newsapi.get_everything(q="Inteligência Artificial")

    return articles
