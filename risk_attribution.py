import numpy as np
import pandas as pd
import networkx as nx

from dowhy import gcm
from config import Config
from common import utils

def _shock_value(df, col):
    return float(df[col].mean() + Config.SHOCK_SDS * df[col].std())

def _fit_gcm_model(data, G):
    model = gcm.StructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(model, data)
    gcm.fit(model, data)
    return model

def _attribution_decomposition(model, outcome, anomalous_rows):
    scores = gcm.attribute_anomalies(
                model, 
                outcome, 
                anomaly_samples=anomalous_rows,
                num_distribution_samples=Config.DISTRIBUTION_SAMPLES)

    return pd.DataFrame(scores).abs().mean().sort_values(ascending=False)

def attribution_accuracy(df, G, outcome, causes, scm=None, seed=Config.RANDOM_SEED):
    if scm is None:
        return None
    
    results = []
    model = _fit_gcm_model(df, G)
    for cause in causes:
        shock_value = _shock_value(df, cause)
        # Generate Anomalous Rows
        anomalous_rows = scm.generate(Config.N_ANOMALOUS, interventions={cause: shock_value}, seed=seed).astype(float)
        
        decomposition = _attribution_decomposition(model, outcome, anomalous_rows)
        rank = 99 if cause not in decomposition.index else list(decomposition.index).index(cause) + 1
        results.append({ 
            "injected_cause": cause,
            "rank": rank, 
            "top1": rank == 1, 
            "top3": rank <= 3, 
        })
    return results

def _edge_strengths(model, df, target):
    target_variance = float(df[target].var())
    strengths = sorted(gcm.arrow_strength(model, target).items(),
                       key=lambda kv: -kv[1])
    return [
        {"source": str(u), "target": str(v), "strength": float(s),
         "share_of_variance_pct":
             round(float(s) / target_variance * 100, 1)
             if target_variance > 0 else None}
        for (u, v), s in strengths
    ]


def _drivers_for(model, df, G, outcome):
    drivers = _edge_strengths(model, df, outcome)

    parents = list(G.predecessors(outcome))
    if len(parents) == 1 and G.in_degree(parents[0]) > 0:
        drivers += _edge_strengths(model, df, parents[0])
    return drivers

def attribution(df, G, outcome, worst_fraction=0.05):
    model = _fit_gcm_model(df, G)
    cutoff = df[outcome].quantile(1 - worst_fraction)
    worst_rows = df[df[outcome] >= cutoff]

    decomposition = _attribution_decomposition(model, outcome, worst_rows)
    decomposition = (decomposition / decomposition.sum() * 100).round(1)
    decomposition = decomposition.sort_values(ascending=False)

    strengths = gcm.arrow_strength(model, outcome)

    drivers = sorted(strengths.items(), key=lambda kv: -kv[1])

    outcome_variance = float(df[outcome].var())

    decomposition_json = [
        {"node": str(node), "share_pct": float(share)}
        for node, share in decomposition.items()
    ]

    drivers_json = _drivers_for(model, df, G, outcome)
    return {"decomposition": decomposition_json, "drivers": drivers_json}

def pick_best_runs(runs, exclude_types=["perfect"]):
    candidates = {}
    for run in runs:
        if run["type"].lower() in exclude_types or run["shd"] is None:
            continue
        candidates.setdefault(run["algo"], []).append(run)

    return [ min(runs, key=lambda r: (-r["n_rows"], r["shd"])) for algo, runs in candidates.items()]

def generate_attribution_report(all_runs, data_dir, outcome):
    best_runs = pick_best_runs(all_runs)
    summary = []
    # Load dataset
    for run in best_runs:
        print(f"\tProcessing run | {run['name']}")
        df = pd.read_csv(data_dir + '/' + run['name'])
        G = utils.convert_json_to_graph(run['dag_json'])
        run['attribution'] = attribution(df, G, outcome)
        summary.append(run)

    return summary


