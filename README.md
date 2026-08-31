# LLM-prior-causal-discovery

Causal priors elicited from large language models, used to **guide** rather than replace statistical causal discovery, applied to supply-chain delivery-risk attribution.

Observational data can often establish *which* operational variables are related to delivery delay but not *which way* the causality runs — within a Markov equivalence class the orientation of many edges is unidentifiable. This codebase elicits a causal prior from an LLM using variable names and descriptions alone, injects it into two structurally different discovery algorithms, and evaluates the recovered graph on three synthetic benchmarks with known structure and on the public DataCo supply-chain dataset, where no ground truth exists.

---

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```
pandas · numpy · matplotlib · seaborn · networkx · scikit-learn
causal-learn · pgmpy · dowhy · requests
numba==0.59.1 · llvmlite==0.42.0
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The two open-weight models are served locally through [Ollama](https://ollama.com) on `http://localhost:11434`:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:1.5b
```

Gemini (`gemini-2.5-flash`) and Cerebras (`gpt-oss-120b`) are called over HTTP and need API keys set in `config.py` (see the warning above).

---

## Quickstart

```bash
# 1. Generate synthetic data + metadata for all legs, and prepare DataCo
python data_setup.py

# 2. Run the full pipeline for every leg
python run.py

# or one leg at a time
python run.py --leg distributor_to_customer
python run.py --leg supplier_to_manufacturer
python run.py --leg manufacturer_to_distributor
python run.py --leg dataco
```

`experiments/` in this repository already contains a complete set of outputs, so the analysis can be inspected without re-running anything. 
---

## What the pipeline does

**Stage 1 — Prior elicitation** (`priors.py`, `llms.py`)

Each model is shown only the variable names and descriptions from `metadata/data_dictionary.json` — never the data — and returns JSON with three fields:

| Field | Meaning |
|---|---|
| `edges_with_probability` | directed relationships the model believes plausible, each with a confidence |
| `forbidden_edges` | relationships it considers implausible |
| `tier_ordering` | a coarse ordering of variables by causal precedence |

Three independent draws are taken per model (`K=3`). `create_consensus_prior` averages edge probabilities across draws, keeps a forbidden edge when at least two draws agree, and averages and rounds the tiers. Both single-draw and consensus priors are evaluated.

Two reference priors bracket the LLM conditions: a **perfect prior** derived from the ground-truth DAG (`PERFECT_EDGE_PROB = 0.9`, reverse edges forbidden, tiers from a topological sort) and a **random prior** (low-probability random edges, random confidences, random tiers; three drawn per leg).

**Stage 2 — Prior-guided discovery** (`causal_discovery.py`, `discovery/`)

Two learners, each run as a data-only baseline and with every prior:

- `discovery/hill_climb.py` — greedy hill-climbing, Gaussian BIC, `max_indegree=5`. The BIC objective is extended with a prior-weighted edge term controlled by λ, tested at **0.1, 0.5 and 0.9**.
- `discovery/pc.py` — PC with Fisher-z tests at α = 0.05. Edges with probability ≥ 0.8 become required, forbidden edges become prohibitions.

**Stage 3 — Attribution and intervention** (`causal_utility.py`)

A DoWhy GCM structural causal model is fitted to each recovered graph. Two distinct quantities are computed and should not be conflated:

- `_attribution_decomposition` — anomaly attribution over the worst 5% of outcomes, normalised to 100%.
- `_edge_strengths` — `gcm.arrow_strength` divided by target variance. **Not normalised and not additive**, so a single driver can exceed 100%.

`injected_cause_recovery` shocks a known cause in the true SCM (`SHOCK_SDS = 3.0`, `N_ANOMALOUS = 25`) and checks whether it surfaces among the top-ranked drivers. It assigns rank 1 or rank 99, so **top-1 and top-3 are identical by construction**.

**Stage 4 — Evaluation** (`priors.py`, `causal_discovery.py`)

- *Prior quality*, against the ground-truth DAG: `fco` (fraction correctly oriented), `tere` (coverage — true edges recalled in any orientation), `tene`.
- *Structure recovery*: SHD, directed precision, recall, F1, averaged over 15 seeds with 95% CIs.
- *Ground-truth-free validation*, DataCo only: `falsification_check` (50 permutations, α = 0.05, 5,000-row subsample) and `bootstrap_stability` (50 resamples, 0.8 recurrence threshold).

---

## Repository layout

```
run.py                     pipeline entry point (--leg optional)
data_setup.py              generates synthetic data + metadata; prepares DataCo
config.py                  seeds, model endpoints, hyper-parameters 
priors.py                  elicitation, consensus, perfect/random priors, FCO/TERE/TENE
llms.py                    HTTP clients for Ollama / Gemini / Cerebras
causal_discovery.py        run_pipeline, graph evaluation, falsification, bootstrap
causal_utility.py          GCM fitting, attribution, injected-cause recovery, intervention
discovery/hill_climb.py    score-based learner + prior integration
discovery/pc.py            constraint-based learner + prior integration
dags/                      per-leg SCM definitions and DataCo preparation
common/scm.py              SCM primitives (linear, concave, threshold mechanisms)
common/utils.py            IO, graph conversion, CI, run selection, CLI
data/                      DataCoSupplyChainDataset.csv (CC-BY-4.0)
experiments/<leg>/         all committed outputs
dashboard/                 static HTML result browser
```

### Experiment outputs

Each `experiments/<leg>/` directory contains:

| File | Contents |
|---|---|
| `metadata/data_dictionary.json` | variable names and descriptions shown to the LLMs |
| `metadata/ref_dag.json` | ground-truth DAG (synthetic legs only) |
| `metadata/downstream_config.json` | target, injected causes, actionable levers |
| `data/` | generated CSVs, `syn_data_<seed>_<n>.csv` |
| `priors/` | every elicited prior, per model, plus consensus, random and perfect |
| `results/` | per-dataset run records |
| `all_runs.json` | flattened runs - algo, type, prior, λ, SHD, precision, recall, F1, DAG |
| `priors_summary.json`, `priors_all.json` | prior-quality diagnostics |
| `discovery_summary.json` | aggregated structure recovery |
| `utility_summary.json` | attribution, injected-cause recovery, intervention effects |
| `validation_summary.json` | falsification and bootstrap results (`dataco`, `dataco_renamed`) |

Legs: `distributor_to_customer` (16 variables), `supplier_to_manufacturer` (10 variables, 11 edges), `manufacturer_to_distributor` (9 variables, 9 edges), `dataco` (62,897 orders, 11 variables), `dataco_renamed` (contamination control with relabelled variables).

**Design grid:** 15 seeds `[3, 42, 5, 7, 11, 17, 23, 29, 41, 53, 67, 79, 89, 97, 101]` × sample sizes `[250, 1000, 5000]` × {HC, PC} × {baseline, perfect, random, 4 models × (single, consensus)} × λ ∈ {0.1, 0.5, 0.9} for HC.

---
## Dashboard

`dashboard/` is a static result browser. It reads the committed JSON in `experiments/` with `fetch()`

Serve from the **repository root**, not from `dashboard/`, because the pages request `../experiments/`:

```bash
cd LLM-prior-causal-discovery
python3 -m http.server 8000
# then open http://localhost:8000/dashboard/index.html
```
---

## Licence and data

DataCo Smart Supply Chain dataset — Constante, Silva and Pereira (2019), CC-BY-4.0. Synthetic benchmark structures are grounded in operations-management literature; per-edge sources are listed in the accompanying write-up.
