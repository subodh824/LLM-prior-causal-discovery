from pgmpy.causal_discovery.HillClimbSearch import HillClimbSearch
from pgmpy.structure_score import BICGauss

from common import utils
from config import Config
import networkx as nx

class PriorWeightedBIC(BICGauss):

    def __init__(self, data: pd.DataFrame, prior_prob: dict, lam: float, **kwargs):
        super().__init__(data, **kwargs)
        self.prior_prob = prior_prob
        self.lam = lam

    def local_score(self, variable, parents):
        base = super().local_score(variable, parents)
        prior_term = self.lam * sum(self.prior_prob.get((p, variable), 0.0) for p in parents)
        return base + prior_term


def _run_hill_climb(df, score, max_indegree):
    hc = HillClimbSearch(scoring_method=score, max_indegree=max_indegree, return_type="dag", show_progress=False)
    hc.fit(df)
    model = hc.causal_graph_
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
    score = PriorWeightedBIC(df, prior_prob=prior_prob, lam=lambda_weight)
    G = _run_hill_climb(df, score, max_indegree)
    return G