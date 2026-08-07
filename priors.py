from config import Config
from common import utils
import random
import json
from itertools import combinations, chain
from collections import Counter, defaultdict
from scipy import stats
from statistics import mean, stdev
import numpy as np
import networkx as nx
import llms as LLMs

def _pair_probs(prior, u, v):
    p_uv = float(prior.get((u, v), 0.0))
    p_vu = float(prior.get((v, u), 0.0))
    p_none = max(0.0, 1.0 - p_uv - p_vu)
    total = p_uv + p_vu + p_none
    if total <= Config.EPS:                      
        return (1 / 3, 1 / 3, 1 / 3)     
    return (p_uv / total, p_vu / total, p_none / total)

def generate_llm_prior(llm, data_dict, name="llm_prior"):
    cfg = Config.LLM[llm]

    if llm.startswith("ollama"):
        prior = LLMs._call_ollama(cfg, data_dict)
    elif llm.startswith("groq"):
        prior = LLMs._call_groq(cfg, data_dict)
    elif llm.startswith("gemini"):
        prior = LLMs._call_gemini(cfg, data_dict)
    elif llm.startswith("cerebras"):
        prior = LLMs._call_cerebras(cfg, data_dict)

    if not prior:
        return None
    
    prior['parsed']['name'] = name
    prior['parsed']['type'] = llm
    return prior['parsed']

def generate_random_prior(data_dict, name="random_prior", seed=Config.RANDOM_SEED):
    rng = random.Random(seed)

    variables = list(data_dict.keys())
    edges_with_probability = []
    for i, a in enumerate(variables):
        for b in variables:
            if a == b:
                continue
            if rng.random() < 0.15:
                edges_with_probability.append([a, b, round(rng.random(), 3)])

    tier_ordering = [[v, rng.randint(0, 3)] for v in variables]

    return {
        "name": name,
        "type": 'random',
        "edges_with_probability": edges_with_probability,
        "forbidden_edges": [],
        "tier_ordering": tier_ordering,
        "rationale": {"_": "RANDOM CONTROL"},
    }

def generate_perfect_prior(ref_G, name="perfect_prior", prob=Config.PERFECT_EDGE_PROB):
    tier = {v: 0 for v in ref_G.nodes()}
    for v in nx.topological_sort(ref_G):
        for _, child in ref_G.out_edges(v):
            tier[child] = max(tier[child], tier[v] + 1)
 
    return {
        "name": "perfect_prior",
        "type": "perfect",
        "edges_with_probability": [[u, v, prob] for u, v in ref_G.edges()],
        "forbidden_edges": [[v, u] for u, v in ref_G.edges()], 
        "tier_ordering": [[v, tier[v]] for v in ref_G.nodes()],
        "rationale": {f"_": "PERFECT PRIOR"},
    }

def generate_priors(data_dict, output_dir, K=3, ref_G=None):

    # Perfect prior
    if ref_G:
        print(" Writing perfect prior")
        perfect_prior = generate_perfect_prior(ref_G)
        utils.write_json(perfect_prior, output_dir + f'/perfect_prior.json')
        
    # Random prior
    print(f" Writing {K} random priors")
    random_priors = []
    for k in range(K):
        random_priors.append(generate_random_prior(data_dict, f"random_prior_{k}", k))
    utils.write_json(random_priors, output_dir + f'/random_priors_list.json')

    # LLM prior
    for llm, val in Config.LLM.items():
        print(f" Writing {K} {llm} priors")
        llm_priors = []
        for k in range(K):
            prior = generate_llm_prior(llm, data_dict, f"{llm}_prior_{k}")
            if prior:
                llm_priors.append(prior)
            else:
                break
        
        # Create LLM Consensus prior
        if len(llm_priors) > 0:
            utils.write_json(llm_priors, output_dir + f'/{llm}_priors_list.json')
            llm_consensus_prior = create_consensus_prior(llm_priors, data_dict)
            utils.write_json(llm_consensus_prior, output_dir + f'/{llm}_consensus_prior.json')

def format_prior(prior):
    acc = defaultdict(list)
    for src, tgt, prob in prior["edges_with_probability"]:
        acc[(str(src), str(tgt))].append(float(prob))
    
    result = {}
    for key, vals in acc.items():
        result[key] = sum(vals) / len(vals)
    return result

def create_consensus_prior(priors, data_dict):
    nodes = list(data_dict.keys())
    
    all_priors = []
    for prior in priors:
        all_priors.append(format_prior(prior))

    consensus = {}
    for u, v in combinations(nodes, 2):
        trip = [_pair_probs(p, u, v) for p in all_priors]
        k = len(trip)
        consensus[(u, v)] = sum(t[0] for t in trip) / k
        consensus[(v, u)] = sum(t[1] for t in trip) / k
 
    edges = [[u, v, round(p, 6)] for (u, v), p in sorted(consensus.items()) if p > 0]

    forbid_counts = Counter(
        (src, tgt) for p in priors
        for src, tgt in p.get("forbidden_edges", [])
    )
    forbidden = [[u, v] for (u, v), c in sorted(forbid_counts.items())
                 if c >= (len(priors) / 2)]

    tier_lists = defaultdict(list)
    for p in priors:
        for var, tier in p.get("tier_ordering", []):
            tier_lists[var].append(float(tier))

    tier_ordering = [[var, round(sum(ts) / len(ts))] for var, ts in sorted(tier_lists.items())]
 
    return {
        "name": f"{priors[0]['type']}_consensus_prior",
        "type": priors[0]['type'],
        "edges_with_probability": edges,
        "forbidden_edges": forbidden,
        "tier_ordering": tier_ordering,
        "rationale": {"_": f"consensus of {len(priors)} priors; forbidden = majority vote; tiers = rounded mean"}
    }
 
def fco(prior, ref_G):
    prior = format_prior(prior)
    scores = []
    for u, v in ref_G.edges():
        p_uv, p_vu, _ = _pair_probs(prior, u, v)
        if abs(p_uv - p_vu) <= Config.EPS:
            scores.append(0.5)
        else:
            scores.append(1.0 if p_uv > p_vu else 0.0)
    return round(sum(scores) / len(scores), 2) if scores else None

def tere(prior, ref_G, as_share=True):
    prior = format_prior(prior)
    num = sum(_pair_probs(prior, u, v)[0] for u, v in ref_G.edges())
    den = sum(_pair_probs(prior, u, v)[1] for u, v in ref_G.edges())
    if num + den <= Config.EPS:
        return None
    share = num / (num + den)
    return round(share,2) if as_share else round(share / max(1.0 - share, Config.EPS),2)

def tene(prior, ref_G, as_share=True):
    prior = format_prior(prior)
    num = sum(_pair_probs(prior, u, v)[0] for u, v in ref_G.edges())
    den = sum(_pair_probs(prior, u, v)[2] for u, v in ref_G.edges())
    if num + den <= Config.EPS:
        return None
    share = num / (num + den)
    return round(share,2) if as_share else round(share / max(1.0 - share, Config.EPS),2)

def lod(prior, data_dict):
    pass


def evaluate_prior(prior, data_dict, ref_G=None):
    return {
        'name': prior['name'],
        'type': prior['type'],
        'fco': fco(prior, ref_G) if ref_G else None,
        'tere': tere(prior, ref_G) if ref_G else None,
        'tene': tene(prior, ref_G) if ref_G else None,
    }

def generate_priors_report(data_dict, all_priors, ref_G=None):
    all_priors_eval = {}
    for prior_type, priors in all_priors.items():
        if prior_type not in all_priors_eval:
            all_priors_eval[prior_type] = []
        
        if isinstance(priors, list):
            for prior in priors:
                all_priors_eval[prior_type].append(evaluate_prior(prior, data_dict, ref_G))
        elif isinstance(priors, dict):
            all_priors_eval[prior_type].append(evaluate_prior(priors, data_dict, ref_G))

    # Overall  Summary
    summary = []
    for prior_type, priors_eval in all_priors_eval.items():
        summary.append({
            'name': "Overall",
            'type': prior_type,
            'count': len(priors_eval),
            'fco' : round(mean(p['fco'] for p in priors_eval if p['fco'] is not None),2) if ref_G else None,
            'fco_ci95' : utils.ci95(list(p['fco'] for p in priors_eval if p['fco'] is not None)) if ref_G else None,
            'tere' : round(mean(p['tere'] for p in priors_eval if p['tere'] is not None),2) if ref_G else None, 
            'tere_ci95' : utils.ci95(list(p['tere'] for p in priors_eval if p['tere'] is not None)) if ref_G else None,
            'tene' : round(mean(p['tene'] for p in priors_eval if p['tene'] is not None),2) if ref_G else None,
            'tene_ci95' : utils.ci95(list(p['tene'] for p in priors_eval if p['tene'] is not None)) if ref_G else None,
            'lod': None
        })

    return summary, list(chain.from_iterable(all_priors_eval.values()))
    


    
    

