from config import Config
import json
import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

from discovery import pc
from discovery import hill_climb

from common import utils
from dowhy.gcm.falsify import falsify_graph, FalsifyConst
from dowhy.gcm.independence_test import regression_based


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

def falsification_check(df, G, n_permutations=50, significance_level=0.05, subsample=5000, seed=0):
    cols = [c for c in G.nodes() if c in df.columns]
    data = df[cols].select_dtypes(include=[np.number])
    subG = G.subgraph(list(data.columns)).copy()

    if subsample and len(data) > subsample:
        data = data.sample(subsample, random_state=seed).reset_index(drop=True)

    print(f"\t Falsification test | {len(data)} rows | {n_permutations} permutations")
    result = falsify_graph(subG, data, n_permutations=n_permutations, significance_level=significance_level, conditional_independence_test=regression_based, show_progress_bar=True)

    if result is None:
        return None

    falsified = bool(getattr(result, "falsified"))
    out = {
        "consistent_with_data": (not falsified),
        "falsified": falsified,
        "falsifiable": bool(getattr(result, "falsifiable", None)),
        "p_value_lmc": None,
        "p_value_tpa": None,
        "given_violations": None,
        "n_tests": None,
        "n_nodes": subG.number_of_nodes(),
        "n_edges": subG.number_of_edges(),
        "n_rows_used": len(data),
    }

    summary = getattr(result, "summary", {}) or {}
    lmc = summary.get(FalsifyConst.VALIDATE_LMC)
    tpa = summary.get(FalsifyConst.VALIDATE_TPA)
    if isinstance(lmc, dict):
        out["p_value_lmc"] = float(lmc.get(FalsifyConst.P_VALUE)) if lmc.get(FalsifyConst.P_VALUE) is not None else None
        out["given_violations"] = lmc.get(FalsifyConst.GIVEN_VIOLATIONS)
        out["n_tests"] = lmc.get(FalsifyConst.N_TESTS)
    if isinstance(tpa, dict) and tpa.get(FalsifyConst.P_VALUE) is not None:
        out["p_value_tpa"] = float(tpa.get(FalsifyConst.P_VALUE))
    return out

def bootstrap_stability(df, algo, prior=None, lamb=None, n_bootstrap=50, threshold=0.8, seed=0, sample_rows=5000):
   
    rng = np.random.default_rng(seed)
    n = len(df)
    edge_counter = Counter()
    node_counter = Counter()
    completed = 0

    def discovery_fn(df, algo, prior=None, lamb=None):
        if  algo == 'PC':
            if prior is None:
                return pc.baseline(df, 0.05)
            else:
                return pc.with_prior(df, 0.05, prior)
        else:
            if prior is None:
                return hill_climb.baseline(df)
            else:
                return hill_climb.with_prior(df, prior, lamb)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=min(sample_rows, n))
        resample = df.iloc[idx].reset_index(drop=True)
        
        G = discovery_fn(resample, algo, prior, lamb)
        completed += 1
        for node in G.nodes():
            node_counter[node] += 1
        for u, v in set(G.edges()):
            edge_counter[(u, v)] += 1
        if (i + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"\t  [bootstrap {i+1}/{n_bootstrap}] done")

    denom = max(completed, 1)
    edge_freq = {f"{u}->{v}": c / denom for (u, v), c in edge_counter.items()}
    node_freq = {n_: c / denom for n_, c in node_counter.items()}
    stable = sorted([f"{u}->{v}" for (u, v), c in edge_counter.items()
                     if c / denom >= threshold])

    return {
        "edge_frequency": dict(sorted(edge_freq.items(), key=lambda kv: -kv[1])),
        "stable_edges": stable,
        "node_frequency": node_freq,
        "n_bootstrap": completed,
        "threshold": threshold,
    }

def read_prior(prior_dir, run):
    if run['type'] in  ['random'] or run['type'] in Config.LLM.keys():
        priors_list = utils.read_json(prior_dir + '/' + f'{run["type"]}_priors_list.json')
        prior_idx = int(run['prior_name'].split("_")[-1])
        return priors_list[prior_idx]

    if "_consensus" in run["type"]:
        return utils.read_json(prior_dir + '/' + f"{run['type']}_prior.json")

    return None
    

def generate_validation_summary(all_runs, data_dir, prior_dir):
    summary = []
    best_runs = utils.pick_best_runs(all_runs)
    print(f"\tPicked {len(best_runs)} best runs of all {len(all_runs)} runs")
    for i, run in enumerate(best_runs):
        print(f"=" * 50)
        print(f"\t [{i+1} / {len(best_runs)}] Processing validation on {run['name']} | {run['algo']} | {run['type']} | {run['prior_name']}")
        df = pd.read_csv(data_dir + '/' + run['name'])
        G = utils.convert_json_to_graph(run['dag_json'])
        
        print("\t Checking falsification ..")
        run['falsification'] = falsification_check(df, G)

        #read prior 
        prior = read_prior(prior_dir, run)
        print("\t Checking bootstrap stability ..")
        run['bootstrap_stability'] = bootstrap_stability(df, run['algo'], prior, run['lambda'])
        summary.append(run)
    return summary


def run(leg):
    print(f"Causal Discovery report for {leg}...")
    exp_dir = f'{Config.EXPERIMENTS_DIR}/{leg}'
    all_runs = utils.read_json(exp_dir + '/all_runs.json')

    data_dir =  f'{exp_dir}/data'
    prior_dir =  f'{exp_dir}/priors'

    print("Writing discovery summary...")
    discovery_summary = generate_discovery_summary(all_runs)
    utils.write_json(discovery_summary, exp_dir + '/discovery_summary.json')

    if leg =="dataco":
        print("Writing validation summary ...")
        validation_summary = generate_validation_summary(all_runs, data_dir, prior_dir)
        utils.write_json(validation_summary, exp_dir + '/validation_summary.json')
    

if __name__ == "__main__":
    args = utils.parse_args()

    print("Starting Causal Discovery reports ...")
    if args.leg is not None:
        run(args.leg)
    else:
        for leg in Config.SUPPLY_CHAIN_LEGS:
            run(leg)

    print("Done.")