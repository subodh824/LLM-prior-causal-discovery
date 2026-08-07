from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BICGauss
from pgmpy.estimators import ExpertKnowledge

from common import utils
from config import Config
import networkx as nx
import pandas as pd

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)

class PriorWeightedBIC(BICGauss):

    def __init__(self, data: pd.DataFrame, prior_prob: dict, lam: float, **kwargs):
        super().__init__(data, **kwargs)
        self.prior_prob = prior_prob
        self.lam = lam

    def local_score(self, variable, parents):
        base = super().local_score(variable, parents)
        prior_term = self.lam * sum(self.prior_prob.get((p, variable), 0.0) for p in parents)
        return base + prior_term


def _tier_groups(tier_ordering):
    by_tier = {}
    for name, tier in tier_ordering:
        by_tier.setdefault(int(tier), []).append(name)

    groups = []
    for tier in sorted(by_tier.keys()):
        groups.append(sorted(by_tier[tier]))
    return groups


def _expert_knowledge_from_prior(prior):
    forbidden_edges = [tuple(edge) for edge in prior.get("forbidden_edges", [])]
    temporal_order = _tier_groups(prior.get("tier_ordering", []))

    if not forbidden_edges and not temporal_order:
        return None

    return ExpertKnowledge(
        forbidden_edges=forbidden_edges or None,
        temporal_order=temporal_order or None,
    )


def _run_hill_climb(df, score, max_indegree, expert_knowledge=None):
    hc = HillClimbSearch(df)
    model = hc.estimate(
        scoring_method=score,
        max_indegree=max_indegree,
        expert_knowledge=expert_knowledge,
        show_progress=False,
    )
    G = nx.DiGraph()
    G.add_nodes_from(df.columns)
    G.add_edges_from(model.edges())

    return G


def baseline(df, max_indegree = 5):
    score = BICGauss(df)
    G = _run_hill_climb(df, score, max_indegree)
    return G


def with_prior(df, prior, lambda_weight, max_indegree = 5):
    prior_prob = {
        (cause, effect): float(prob) for cause, effect, prob in prior.get("edges_with_probability", [])
    }
    expert_knowledge = _expert_knowledge_from_prior(prior)

    score = PriorWeightedBIC(df, prior_prob=prior_prob, lam=lambda_weight)
    G = _run_hill_climb(df, score, max_indegree, expert_knowledge=expert_knowledge)
    return G