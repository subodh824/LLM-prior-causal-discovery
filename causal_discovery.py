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
    if true_G is None:
        return {
            "shd": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "n_edges_est": est_G.number_of_edges()
        }

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
        "n_edges_est": est_G.number_of_edges()
    }

def run_pipeline(df, data_dict, all_priors):
    print("\tRunning causal discovery...")
    prev = time.time()

    results = []

    # ===== PC
    print("\tRunning PC Baseline")
    results.append({ 
        'algo': 'PC',
        'type': 'Baseline',
        'prior_name': None,
        'dag': pc.baseline(df, 0.05)
    })

    for prior_type, prior in all_priors.items():
        print(f"\tRunning PC with {prior_type} prior")
        results.append({ 
            'algo': 'PC',
            'type': prior_type,
            'prior_name': prior['name'],
            'dag': pc.with_prior(df, 0.05, prior)
        })

    # ======= Hill Climb 
    print("\tRunning HillClimb Baseline")
    results.append({ 
        'algo': 'HC',
        'type': 'Baseline',
        'prior_name': None,
        'dag': hill_climb.baseline(df)
    })

    lambda_grid = [0.1, 0.5, 0.9]
    for prior_type, prior in all_priors.items():
        for lamb in lambda_grid:
            print(f"\tRunning HillClimb with {prior_type} prior. Lambda = {lamb}")
            results.append({ 
                'algo': 'HC',
                'type': prior_type,
                'prior_name': prior['name'],
                'dag': hill_climb.with_prior(df, prior, lamb),
                'params': {
                    'lambda': lamb
                }
            })
    
    print(f"\tCausal Discovery took {time.time()-prev:.1f} secs\n")
    return results

def generate_discovery_summary(runs):
    METRICS = ["shd", "f1", "precision", "recall"]

    df = pd.DataFrame(runs)
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

    return summary



