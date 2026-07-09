from config import Config
from common import utils
import random
import json
from itertools import combinations
from collections import Counter, defaultdict
from scipy import stats
from statistics import mean, stdev
import numpy as np
import networkx as nx

SYSTEM_PROMPT = '''You are a supply chain operations research expert helping \
propose a causal structure prior for causal discovery. You will be given a \
list of variables (names + descriptions only, no data) from a supply chain \
dataset. Using domain knowledge of supply chain, logistics, and operations \
management literature, propose plausible directed causal relationships \
among these variables.

Respond with STRICT JSON only, no markdown fences, no commentary, matching \
exactly this schema:
{
  "edges_with_probability": [["cause_var", "effect_var", 0.0_to_1.0], ...],
  "forbidden_edges": [["a", "b"], ...],
  "tier_ordering": [["var", tier_integer], ...],
  "rationale": {"cause->effect": "one sentence justification", ...}
}

Rules:
- Only use variable names exactly as given.
- "edges_with_probability": your belief (0-1) that a directed causal edge \
cause->effect exists. Include only edges you consider plausible (probability >= 0.05).
- "forbidden_edges": edges you are confident CANNOT be causal (e.g. effect \
preceding cause, or no plausible mechanism).
- "tier_ordering": assign every variable to an integer tier representing \
its rough causal depth (0 = exogenous/root cause, higher = more downstream \
effect). Used as a topological-order constraint.
- "rationale": one-sentence justification for each edge in edges_with_probability, \
keyed as "cause->effect".'''

def _pair_probs(prior, u, v):
    p_uv = float(prior.get((u, v), 0.0))
    p_vu = float(prior.get((v, u), 0.0))
    p_none = max(0.0, 1.0 - p_uv - p_vu)
    total = p_uv + p_vu + p_none
    if total <= Config.EPS:                      
        return (1 / 3, 1 / 3, 1 / 3)     
    return (p_uv / total, p_vu / total, p_none / total)

def _build_user_prompt(data_dict):
    var_lines = []
    for name, info in data_dict.items():
        desc = info.get("description", "")
        var_lines.append(f"- {name}: {desc}")
    return (
        "Variables in this supply chain dataset:\n\n"
        + "\n".join(var_lines)
        + "\n\nPropose the causal structure prior as specified in the system prompt."
    )

def _call_groq(data_dict):
    import requests

    api_key = Config.LLM['groq']['api_key']
    user_prompt = _build_user_prompt(data_dict)

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": Config.LLM['groq']['model'], 
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": Config.LLM['groq']["temperature"],
                "max_tokens": Config.LLM['groq']["max_tokens"],
            },
        )
        response.raise_for_status()
        
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach Groq API: {e}")

    raw_text = response.json()["choices"][0]["message"]["content"].strip()
    try:
        text = raw_text
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):text.rfind("}") + 1]
        parsed = json.loads(text)
        return {"raw_response": raw_text, "parsed": parsed}
    except (json.JSONDecodeError, ValueError) as e:
        raise

def _call_ollama(data_dict):
    import requests

    host = Config.LLM['ollama']['host']
    user_prompt = _build_user_prompt(data_dict)

    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": 'llama3.1:8b',
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": Config.LLM['ollama']["temperature"],
                    "num_predict": Config.LLM['ollama']["max_tokens"],
                },
            }
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach Ollama at {host}")

    
    raw_text = response.json()["message"]["content"].strip()
    try:
        text = raw_text
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):text.rfind("}") + 1]
        parsed = json.loads(text)
        return {"raw_response": raw_text, "parsed": parsed}
    except (json.JSONDecodeError, ValueError) as e:
        raise

def generate_llm_prior(data_dict, name="llm_prior"):
    priors = _call_groq(data_dict)

    priors['parsed']['name'] = name
    priors['parsed']['type'] = 'llm'
    return priors['parsed']

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
        "type": "perfect",
        "name": "perfect_prior",
        "edges_with_probability": [[u, v, prob] for u, v in ref_G.edges()],
        "forbidden_edges": [[v, u] for u, v in ref_G.edges()], 
        "tier_ordering": [[v, tier[v]] for v in ref_G.nodes()],
        "rationale": {f"_": "PERFECT PRIOR"},
    }

def generate_priors(data_dict, output_dir, K=5, ref_G=None):

    # Perfect prior
    if ref_G:
        perfect_prior = generate_perfect_prior(ref_G)
        utils.write_json(perfect_prior, output_dir + f'/perfect_prior.json')

    # Random prior
    random_priors = []
    for k in range(K):
        random_priors.append(generate_random_prior(data_dict, f"random_prior_{k}", k))
    utils.write_json(random_priors, output_dir + f'/random_priors_list.json')

    # LLM prior
    llm_priors = []
    for k in range(K):
        llm_priors.append(generate_llm_prior(data_dict, f"llm_prior_{k}"))
        if k >= 2:
            break
    utils.write_json(llm_priors, output_dir + f'/llm_priors_list.json')

    # Create LLM Consensus prior
    llm_consensus_prior = create_consensus_prior(llm_priors, data_dict)
    utils.write_json(llm_consensus_prior, output_dir + f'/llm_consensus_prior.json')

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
        "name": "llm_consensus_prior",
        "type": "llm_consensus",
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
        'fco': fco(prior, ref_G),
        'tere': tere(prior, ref_G),
        'tene': tene(prior, ref_G),
    }

def generate_priors_report(data_dict, perfect_prior, random_priors, llm_priors, ref_G=None):
    
    # Perfect prior 
    perfect_prior_eval = evaluate_prior(perfect_prior, data_dict, ref_G)

    # Random priors
    random_priors_eval = []
    for random_prior in random_priors:
        random_priors_eval.append(evaluate_prior(random_prior, data_dict, ref_G))

    # LLM Priors
    llm_priors_eval = []
    for llm_prior in llm_priors:
        llm_priors_eval.append(evaluate_prior(llm_prior, data_dict, ref_G))


    all_priors_eval = [perfect_prior_eval] + random_priors_eval + llm_priors_eval

    # Overall  Summary
    summary = []
    
    summary.append({
        'name': perfect_prior_eval['name'],
        'type': perfect_prior_eval['type'],
        'fco': perfect_prior_eval['fco'],
        'tere': perfect_prior_eval['tere'],
        'tene': perfect_prior_eval['tene'],
        'lod': None
    })

    summary.append({
        'name': 'random_overall',
        'type': 'random',
        'count': len(random_priors_eval),
        'fco' : round(mean(p['fco'] for p in random_priors_eval),2),
        'fco_ci95' : utils.ci95(list(p['fco'] for p in random_priors_eval)),
        'tere' : round(mean(p['tere'] for p in random_priors_eval),2),
        'tere_ci95' : utils.ci95(list(p['tere'] for p in random_priors_eval)),
        'tene' : round(mean(p['tene'] for p in random_priors_eval),2),
        'tene_ci95' : utils.ci95(list(p['tene'] for p in random_priors_eval))
    })

    summary.append({
        'name': 'llm_overall',
        'type': 'llm',
        'count': len(llm_priors_eval),
        'fco' : round(mean(p['fco'] for p in llm_priors_eval),2),
        'fco_ci95' : utils.ci95(list(p['fco'] for p in llm_priors_eval)),
        'tere' : round(mean(p['tere'] for p in llm_priors_eval),2),
        'tere_ci95' : utils.ci95(list(p['tere'] for p in llm_priors_eval)),
        'tene' : round(mean(p['tene'] for p in llm_priors_eval),2),
        'tene_ci95' : utils.ci95(list(p['tene'] for p in llm_priors_eval))
    })

    return summary, all_priors_eval
    


    
    

