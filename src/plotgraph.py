"""Just plot the graph and subgraphs of the whole system."""
from .graphs import ARKEY_GRAPH  # noqa: I001


ARKEY_GRAPH.get_graph().draw_png("png/arkey.png")
for n, sub in ARKEY_GRAPH.get_subgraphs():
    sub.get_graph().draw_png(f"png/{n}.png")
