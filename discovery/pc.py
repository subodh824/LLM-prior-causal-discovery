from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from common import utils
from config import Config
import networkx as nx

def _build_graph(cg, variable_names):
    G = nx.DiGraph()
    G.add_nodes_from(variable_names)
    amat = cg.G.graph
    n = len(variable_names)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if amat[a, b] == -1 and amat[b, a] == 1:
                G.add_edge(variable_names[a], variable_names[b])
    
    return G

def _build_background_knowledge(priors, variable_names):
    nodes = {name: GraphNode(name) for name in variable_names}
    bk = BackgroundKnowledge()

    for a, b in priors.get("forbidden_edges", []):
        if a in nodes and b in nodes:
            bk.add_forbidden_by_node(nodes[a], nodes[b])

    for var, tier in priors.get("tier_ordering", []):
        if var in nodes:
            bk.add_node_to_tier(nodes[var], int(tier))

    return bk

def baseline(df, alpha, indep_test = 'fisherz'):
    variable_names = list(df.columns)
    cg = pc(df.values, alpha=alpha, indep_test=indep_test, node_names=variable_names)

    G = _build_graph(cg, variable_names)
    return G


def with_prior(df, alpha, prior, indep_test= 'fisherz'):
    variable_names = list(df.columns)
    bk = _build_background_knowledge(prior, variable_names)
    cg = pc(
        df.values, alpha=alpha, indep_test=indep_test, node_names=variable_names,
        background_knowledge=bk
    )

    G = _build_graph(cg, variable_names)
    return G

