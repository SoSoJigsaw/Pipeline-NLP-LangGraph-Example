from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Annotated, List

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL


@tool
def print_results(
        points: Annotated[List[str], "Lista de resumos a respeito das noticias mais recentes de IA"]
) -> Annotated[str, "Definindo para que seja printado cada resumo."]:
    for i, point in enumerate(points):
        print(f"{i + 1}. {point}\n")
    return "Todos os resumos das principais notícias de IA da semana foram printados com sucesso"
