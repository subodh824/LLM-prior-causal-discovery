import json
import requests
from config import Config


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
  "tier_ordering": [["var", tier_integer], ...]
}

Rules:
- Only use variable names exactly as given.
- "edges_with_probability": your belief (0-1) that a directed causal edge \
cause->effect exists. Include only edges you consider plausible (probability >= 0.05).
- "forbidden_edges": edges you are confident CANNOT be causal (e.g. effect \
preceding cause, or no plausible mechanism).
- "tier_ordering": assign every variable to an integer tier representing \
its rough causal depth (0 = exogenous/root cause, higher = more downstream \
effect). Used as a topological-order constraint.'''

def _extract_json(raw_text):
    text = raw_text
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):text.rfind("}") + 1]
    return json.loads(text)

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


def _call_groq(cfg, data_dict):
    user_prompt = _build_user_prompt(data_dict)
    try:
        response = requests.post(
            f"{cfg['host']}",
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": cfg["temperature"],
                "max_tokens": cfg["max_tokens"],
            },
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not reach Groq API: {e}")
        return None

    raw_text = response.json()["choices"][0]["message"]["content"].strip()
    return {"raw_response": raw_text, "parsed": _extract_json(raw_text)}


def _call_ollama(cfg, data_dict):
    user_prompt = _build_user_prompt(data_dict)
    try:
        response = requests.post(
            f"{cfg['host']}",
            json={
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": cfg["temperature"],
                    "num_predict": cfg["max_tokens"],
                },
            },
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not reach ollama API: {e}")
        return None

    raw_text = response.json()["message"]["content"].strip()
    return {"raw_response": raw_text, "parsed": _extract_json(raw_text)}


def _call_cerebras(cfg, data_dict):
    user_prompt = _build_user_prompt(data_dict)
    try:
        response = requests.post(
            f"{cfg['host']}",
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "stream": False,
                "temperature": cfg["temperature"],
                "max_tokens": cfg["max_tokens"],
            },
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not reach cerebras API: {e}")
        return None

    choice = response.json()["choices"][0]
    raw_text = (choice["message"].get("content") or "").strip()
    if not raw_text:
        print(f"Cerebras returned empty content (finish_reason={choice.get('finish_reason')}).")
        return None
    return {"raw_response": raw_text, "parsed": _extract_json(raw_text)}


def _call_gemini(cfg, data_dict):
    user_prompt = _build_user_prompt(data_dict)
    try:
        response = requests.post(
            f"{cfg['host']}",
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": cfg["temperature"],
                "max_tokens": cfg["max_tokens"],
            },
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not reach Gemini API: {e}")
        return None

    choice = response.json()["choices"][0]
    raw_text = (choice["message"].get("content") or "").strip()
    if not raw_text:
        print(f"Gemini returned empty content (finish_reason={choice.get('finish_reason')}).")
        return None
    return {"raw_response": raw_text, "parsed": _extract_json(raw_text)}
