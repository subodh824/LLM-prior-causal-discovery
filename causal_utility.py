import numpy as np
import pandas as pd
import networkx as nx

from dowhy import gcm
from config import Config
from common import utils

from dowhy.gcm.config import disable_progress_bars
disable_progress_bars()

def _shock_value(df, col):
    return float(df[col].mean() + Config.SHOCK_SDS * df[col].std())

def _fit_gcm_model(data, G):
    model = gcm.StructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(model, data)
    gcm.fit(model, data)
    return model


def _attribution_decomposition(model, target, anomalous_rows):
    scores = gcm.attribute_anomalies(
        model, target,
        anomaly_samples=anomalous_rows,
        num_distribution_samples=Config.DISTRIBUTION_SAMPLES)
    return pd.DataFrame(scores).abs().mean().sort_values(ascending=False)


def injected_cause_recovery(df, G, target, causes, scm=None, seed=Config.RANDOM_SEED):
    if scm is None or len(causes) == 0:
        return []

    print(f"\t\t Running injected cause recovery..")
    results = []
    model = _fit_gcm_model(df, G)
    for cause in causes:
        shock_value = _shock_value(df, cause)
        anomalous_rows = scm.generate(
            Config.N_ANOMALOUS,
            interventions={cause: shock_value},
            seed=seed).astype(float)

        decomposition = _attribution_decomposition(model, target, anomalous_rows)
        if cause in decomposition.index:
            rank = list(decomposition.index).index(cause) + 1
        else:
            rank = 99
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


def _drivers_for(model, df, G, target):
    drivers = _edge_strengths(model, df, target)
    parents = list(G.predecessors(target))
    if len(parents) == 1 and G.in_degree(parents[0]) > 0:
        drivers += _edge_strengths(model, df, parents[0])
    return drivers


def attribution(df, model, G, target, worst_fraction=0.05):
    print(f"\t\t Running attribution on {target} ..")
    cutoff = df[target].quantile(1 - worst_fraction)
    worst_rows = df[df[target] >= cutoff]

    decomposition = _attribution_decomposition(model, target, worst_rows)
    decomposition = (decomposition / decomposition.sum() * 100).round(1)
    decomposition = decomposition.sort_values(ascending=False)

    decomposition_json = [
        {"node": str(node), "share_pct": float(share)}
        for node, share in decomposition.items()
    ]

    parents = list(G.predecessors(target))
    if parents:
        drivers_json = _drivers_for(model, df, G, target)
    else:
        drivers_json = []

    return {"decomposition": decomposition_json, "drivers": drivers_json}


def _select_levers(G, target, actionables, prefer_upstream=True):
    ancestors = nx.ancestors(G, target)

    levers = []
    for var in actionables:
        if var in G and var in ancestors:
            try:
                hops = nx.shortest_path_length(G, var, target)
            except nx.NetworkXNoPath:
                continue
            levers.append({"lever": var, "hops_to_target": hops})

    if prefer_upstream:
        levers.sort(key=lambda d: -d["hops_to_target"])
    return levers


def _new_value(df, lever, q=0.10):
    return float(df[lever].quantile(q))


def _delta_value(df, lever, n_std=1.0):
    return float(-n_std * df[lever].std())


def intervention_effect(df, model, target, lever, new_value):
    baseline_mean = df[target].mean()
    samples = gcm.interventional_samples(
        model,
        interventions={lever: lambda x: new_value},
        num_samples_to_draw=len(df))
    intervened_mean = samples[target].mean()
    return {
        "target": target, "new_value": new_value,
        "baseline_mean": float(baseline_mean),
        "intervened_mean": float(intervened_mean),
        "effect": float(intervened_mean - baseline_mean),
    }


def shift_effect(df, model, target, lever, delta):
    baseline_mean = df[target].mean()
    samples = gcm.interventional_samples(
        model,
        interventions={lever: lambda x: x + delta},
        num_samples_to_draw=len(df))
    intervened_mean = samples[target].mean()
    return {
        "target": target, "delta": delta,
        "baseline_mean": float(baseline_mean),
        "intervened_mean": float(intervened_mean),
        "effect": float(intervened_mean - baseline_mean),
    }

def _col_stats(df, column):
    col = df[column]
    return {
        "mean": float(col.mean()),
        "median": float(col.median()),
        "std": float(col.std()),
        "min": float(col.min()),
        "max": float(col.max()),
        "p10": float(col.quantile(0.10)),
        "p90": float(col.quantile(0.90)),
        "current_value": float(col.median()),
    }

def generate_utility_report(all_runs, data_dir, target, causes, actionables, scm=None):
    summary = []
    best_runs = utils.pick_best_runs(all_runs)
    print(f"\tPicked {len(best_runs)} best runs of all {len(all_runs)} runs")
    for i, run in enumerate(best_runs):
        print(f"=" * 50)
        print(f"\t [{i+1} / {len(best_runs)}] Processing downstream utility run | {run['name']} | {run['algo']} | {run['type']} | {run['prior_name']}")
        df = pd.read_csv(data_dir + '/' + run['name'])
        G = utils.convert_json_to_graph(run['dag_json'])
        print("\t\t Fitting model on data ..")
        model = _fit_gcm_model(df, G)

        run['attribution'] = attribution(df, model, G, target)
        run['injected_cause_recovery'] = injected_cause_recovery(df, G, target, causes, scm)
        run['target'] = target
        run['target_stats'] = _col_stats(df, target)
        run['interventions'] = []
        levers = _select_levers(G, target, actionables)
        print(f"\t\t Intervention Levers Count: [{ len(levers)}]")
        if len(levers) > 0:
            print(f"\t\t Running intervention ..")
            for entry in levers:
                print(f"\t\t\t intervention lever : {entry['lever']}")
                lever = entry["lever"]
                new_value = _new_value(df, lever)
                delta = _delta_value(df, lever)
                run['interventions'].append({
                    "lever": lever,
                    "hops_to_target": entry["hops_to_target"],
                    "stats": _col_stats(df, lever),
                    "intervention": intervention_effect(df, model, target, lever, new_value),
                    "shift": shift_effect(df, model, target, lever, delta),
                })
        summary.append(run)
    return summary


def run(leg):
    print(f"Causal Utility pipeline for {leg}...")
    exp_dir = f'{Config.EXPERIMENTS_DIR}/{leg}'
    all_runs = utils.read_json(exp_dir + '/all_runs.json')

    data_dir =  f'{exp_dir}/data'

    scm = utils.get_scm(leg)

    downstream_config_filepath = f'{exp_dir}/metadata/downstream_config.json'
    downstream_config = utils.read_json(downstream_config_filepath)
    target = downstream_config["target"]
    causes = downstream_config["causes"]
    actionables = downstream_config["actionables"]

    utility_summary = generate_utility_report(all_runs, data_dir, target, causes, actionables, scm)
    
    print("Writing utility report...")
    utils.write_json(utility_summary, exp_dir + '/utility_summary.json')

    

if __name__ == "__main__":
    args = utils.parse_args()

    print("Starting Causal Utility pipeline...")
    if args.leg is not None:
        run(args.leg)
    else:
        for leg in Config.SUPPLY_CHAIN_LEGS:
            run(leg)

    print("Done.")
    