from config import Config
import json
import os
import time
import numpy as np
import pandas as pd
import networkx as nx

from discovery import pc
from discovery import hill_climb

from common import utils

def _skeleton(G):
    return {frozenset(e) for e in G.edges()}

def evaluate_graph(true_G, est_G, directed=True):
    true_edges = set(true_G.edges())
    est_edges = set(est_G.edges())
    true_skel = _skeleton(true_G)
    est_skel = _skeleton(est_G)

    missing = true_skel - est_skel
    extra = est_skel - true_skel
    common = true_skel & est_skel

    wrong_direction = 0
    for fs in common:
        a, b = tuple(fs)
        true_dir_ab = (a, b) in true_edges
        est_dir_ab = (a, b) in est_edges
        if true_dir_ab != est_dir_ab:
            wrong_direction += 1

    shd = len(missing) + len(extra) + wrong_direction

    true_set = true_edges if directed else true_skel
    est_set = est_edges if directed else est_skel

    tp = len(true_set & est_set)
    fp = len(est_set - true_set)
    fn = len(true_set - est_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "shd": shd,
        "precision": utils.safe_round(precision,2),
        "recall": utils.safe_round(recall,2),
        "f1": utils.safe_round(f1,2),
    }

def run_causal_discovery_pipeline(df, data_dict, all_priors):

    prev = time.time()

    results = []

    # ===== PC
    print("Running PC Baseline")
    results.append({ 
        'algo': 'PC',
        'type': 'Baseline',
        'prior_name': None,
        'dag': pc.baseline(df, 0.05)
    })

    for prior_type, prior in all_priors.items():
        print(f"Running PC with {prior_type} prior")
        results.append({ 
            'algo': 'PC',
            'type': prior_type,
            'prior_name': prior['name'],
            'dag': pc.with_prior(df, 0.05, prior)
        })


    # ======= Hill Climb 
    print("Running HillClimb Baseline")
    results.append({ 
        'algo': 'HC',
        'type': 'Baseline',
        'prior_name': None,
        'dag': hill_climb.baseline(df)
    })

    # lambda_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lambda_grid = [0.1, 0.5, 0.9]
    for prior_type, prior in all_priors.items():
        for lamb in lambda_grid:
            print(f"Running HillClimb with {prior_type} prior. Lambda = {lamb}")
            results.append({ 
                'algo': 'HC',
                'type': prior_type,
                'prior_name': prior['name'],
                'dag': hill_climb.with_prior(df, prior, lamb),
                'params': {
                    'lambda': lamb
                }
            })
    
    print(f"Took {time.time()-prev:.1f} secs\n")
    
    return results

def generate_discovery_report(results):
    METRICS = ["shd", "f1", "precision", "recall", "n_edges_est"]
    rows = []
    for result in results:
        for r in result['runs']:
            lam = (r.get("params") or {}).get("lambda", np.nan)
            rows.append({
                'data': result['name'],
                "algo": r["algo"],
                "type": r["type"],
                "lambda": lam,
                "seed": result['seed'],
                "n_rows": result['num_rows'],
                "shd": r["shd"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "n_edges_est": r["dag"].number_of_edges() if r.get("dag") is not None else np.nan,
            })

    df = pd.DataFrame(rows)
    all_runs = df.replace({np.nan: None}).to_dict(orient="records")

    group_cols = ["algo", "type", "lambda", "n_rows"]
    summary = []
    for keys, group in df.groupby(group_cols, dropna=False):
        algo, algo_type, lam, n_rows = keys
        entry = {
            "algo": algo,
            "type": algo_type,
            "lambda": None if pd.isna(lam) else lam,
            "n_rows": n_rows,
            "n_seeds": group["seed"].nunique()
        }
        for metric in METRICS:
            entry[metric] = {
                "mean": utils.safe_round(group[metric].mean(), 2),
                "ci95": utils.safe_round(utils.ci95(group[metric]), 2),
            }
        summary.append(entry)

    return all_runs, summary



