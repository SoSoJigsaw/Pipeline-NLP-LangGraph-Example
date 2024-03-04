from typing import Annotated, List
from langchain_core.tools import tool


@tool
def print_results(results):
    """
        Imprime os resultados da equipe de escrita.

        Args:
            results: Dicionário contendo os resultados da equipe de escrita.

        Raises:
            ValueError: Se os resultados não forem válidos.
        """
    # Valida os resultados
    if not isinstance(results, dict):
        raise ValueError("Resultados inválidos")

    # Imprime os resultados
    for key, value in results.items():
        print(f"{key}: {value}")
