from config import Config
from common import utils
import random
import json
from itertools import combinations

 
def _pair_probs(prior, u, v):
    p_uv = float(prior.get((u, v), 0.0))
    p_vu = float(prior.get((v, u), 0.0))
    p_none = max(0.0, 1.0 - p_uv - p_vu)
    total = p_uv + p_vu + p_none
    if total <= Config.EPS:                      
        return (1 / 3, 1 / 3, 1 / 3)     
    return (p_uv / total, p_vu / total, p_none / total)

def generate_random_prior(data_dict):
    rng = random.Random(Config.RANDOM_SEED)

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
        "edges_with_probability": edges_with_probability,
        "forbidden_edges": [],
        "tier_ordering": tier_ordering,
        "rationale": {f"{a}->{b}": "RANDOM CONTROL — no rationale, generated without LLM" for a, b, _ in edges_with_probability},
    }

def generate_perfect_prior(data_dict):
    return generate_random_prior(data_dict)

def create_consensus_prior(priors, data_dict):
    nodes = list(data_dict.keys())
    out = {}
    for u, v in combinations(nodes, 2):
        trip = [_pair_probs(p, u, v) for p in priors] 
        k = len(trip)
        mean_uv = sum(t[0] for t in trip) / k
        mean_vu = sum(t[1] for t in trip) / k
        out[(u, v)] = mean_uv
        out[(v, u)] = mean_vu
    return out

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
    print(raw_text)
    try:
        text = raw_text
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):text.rfind("}") + 1]
        parsed = json.loads(text)
        return {"raw_response": raw_text, "parsed": parsed}
    except (json.JSONDecodeError, ValueError) as e:
        raise

def generate_llm_prior(data_dict):
    priors = _call_ollama(data_dict)

    return priors['parsed']


def generate_priors(data_dict, output_dir, K=10):

    # Perfect prior
    perfect_prior = generate_perfect_prior(data_dict)
    utils.write_json(perfect_prior, output_dir + f'/perfect_prior.json')

    # Random prior
    output_dir_random = output_dir + '/random'
    utils.create_dir_if_not_exists(output_dir_random)
    for k in range(K):
        random_prior = generate_random_prior(data_dict)
        utils.write_json(random_prior, output_dir_random + f'/random_prior_{k}.json')

    # LLM Prior
    llm_priors = []
    output_dir_llm = output_dir + '/llm'
    utils.create_dir_if_not_exists(output_dir_llm)
    for k in range(K):
        # llm_prior = generate_llm_prior(data_dict)
        llm_prior = generate_random_prior(data_dict)
        utils.write_json(llm_prior, output_dir_llm + f'/llm_prior_{k}.json')
        llm_priors.append(llm_prior)

    # Create LLM Consensus prior
    llm_consensus_prior = create_consensus_prior(llm_priors, data_dict)
    utils.write_json(llm_prior, output_dir + f'/llm_consensus_prior.json')