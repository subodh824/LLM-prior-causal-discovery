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

def calculate_shd(true_G, est_G):
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

    return len(missing) + len(extra) + wrong_direction

def precision_recall_f1(true_G, est_G, directed=True):
    if directed:
        true_set = set(true_G.edges())
        est_set = set(est_G.edges())
    else:
        true_set = _skeleton(true_G)
        est_set = _skeleton(est_G)

    tp = len(true_set & est_set)
    fp = len(est_set - true_set)
    fn = len(true_set - est_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def run_causal_discovery_pipeline(df, data_dict, perfect_prior, random_prior, llm_prior):

    prev = time.time()

    results = []

    # ===== PC
    print("Running PC Baseline")
    results.append({ 
        'algo': 'pc_baseline',
        'dag': pc.baseline(df, 0.05)
    })

    print("Running PC with perfect priors")
    results.append({ 
        'algo': 'pc_perfect_prior',
        'dag': pc.with_prior(df, 0.05, perfect_prior)
    })

    print("Running PC with random priors")
    results.append({ 
        'algo': 'pc_random_prior',
        'dag': pc.with_prior(df, 0.05, random_prior)
    })

    print("Running PC with LLM priors")
    results.append({ 
        'algo': 'pc_llm_prior',
        'dag': pc.with_prior(df, 0.05, llm_prior)
    })

    # ======= Hill Climb 
    print("Running HillClimb Baseline")
    results.append({ 
        'algo': 'hc_baseline',
        'dag': hill_climb.baseline(df)
    })

    lambda_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("Running HillClimb with perfect priors")
    for lamb in lambda_grid:
        results.append({ 
            'algo': 'hc_perfect_prior',
            'dag': hill_climb.with_prior(df, perfect_prior, lamb),
            'params': {
                'lambda': lamb
            }
        })

    print("Running HillClimb with random priors")
    for lamb in lambda_grid:
        results.append({ 
            'algo': 'hc_random_prior',
            'dag': hill_climb.with_prior(df, random_prior, lamb),
            'params': {
                'lambda': lamb
            }
        })
    
    print("Running HillClimb with LLM priors")
    for lamb in lambda_grid:
        results.append({ 
            'algo': 'hc_llm',
            'dag': hill_climb.with_prior(df, llm_prior, lamb),
            'params': {
                'lambda': lamb
            }
        })
    
    print(f"Took {time.time()-prev:.1f} secs\n")
    
    return results

def generate_discovery_report(results):
    
    rows = []
    for file, runs in results.items():
        file_splits = file.split("_")
        seed, n = file_splits[2], file_splits[3]
        for r in runs:
            rows.append({
                'data': file,
                "seed": seed, 
                "n_rows": n,
                "algo": r["algo"],
                "lambda": (r.get("params") or {}).get("lambda", np.nan),
                "shd": r["shd"],
                "precision": r["edges"]["precision"],
                "recall": r["edges"]["recall"],
                "f1": r["edges"]["f1"],
                "n_edges_est": (r["dag"].number_of_edges() if r.get("dag") is not None else np.nan),
            })
    df = pd.DataFrame(rows)
    df["condition"] = df.apply(lambda x: x["algo"] if np.isnan(x["lambda"]) else f"{x['algo']}(lambda={x['lambda']:g})", axis=1)
    
    # SHD Evaluation
    df_shd_comp = (df.groupby(["condition", "n_rows"])['shd']
           .agg(mean="mean", ci95=utils.ci95, n_seeds="count")
           .reset_index())

    df_shd_comp = df_shd_comp.pivot(index="condition", columns="n_rows", values=["mean", "ci95"])
    df_shd_comp.columns = df_shd_comp.columns.swaplevel(0, 1)
    df_shd_comp = df_shd_comp.sort_index(axis=1, level=0)
    df_shd_comp.columns = [f'{n_rows}_{metric}' for n_rows, metric in df_shd_comp.columns]

    shd_results = df_shd_comp.reset_index().replace({np.nan: None}).to_dict(orient='records')
    
    # Metrics Evaluation
    df_metrics = []
    for metric in ["f1", "precision", "recall", "n_edges_est"]:
        df_temp = (df.groupby("condition")[metric]
             .agg(**{f"{metric}_mean": "mean", f"{metric}_ci95": utils.ci95}))
        df_metrics.append(df_temp)
    df_metrics = pd.concat(df_metrics, axis=1).round(3)
    df_metrics.reindex()
    metrics_result = df_metrics.reset_index().replace({np.nan: None}).to_dict(orient='records')

    return shd_results, metrics_result


