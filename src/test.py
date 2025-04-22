"""Un cas d'exmple pour valider le pipeline."""

from .graphs import ARKEY_GRAPH


def parkey(x: str) -> None:
    """Effectue un tour dans le graphe de digita et en imprime la réponse."""
    rsp = ARKEY_GRAPH.invoke({"original": x})
    print(rsp["response"])  # noqa: T201

def sarkey(x: str) -> None:
    """Effectue un tour dans le graphe de digita et en imprime la réponse."""
    for c in ARKEY_GRAPH.stream({"original": x}):
        print(c)  # noqa: T201



if __name__ == "__main__":
    ARKEY_GRAPH.get_graph().draw_png(output_file_path="graph.png")
    #
    #parkey("""S'agit-il de Marneffe Joseph né en 1829 à Namur?
    # A-t-il été emprisonné et si oui combien de temps? Merci pour votre réponse""")
    #print("==" * 120)  # noqa: ERA001
    #
    #parkey("""
    #Bonjour et Meilleurs Voeux
    #Je recherche dans vos archives numérisés de Bruxelles, l'acte de mariage 2117 du 30/11/1901
    #entre WILMART Jean Eugène & PIRARD Anna Alexandrine.
    #Je ne trouve pas dernier trimestre 1901.
    #Bien à vous
    #""")
    #print("==" * 120)  # noqa: ERA001
