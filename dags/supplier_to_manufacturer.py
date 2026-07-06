import networkx as nx
from common import scm, utils
import pandas as pd
import numpy as np
import config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

NODES = [
    "geopolitical_disruption",
    "supplier_financial_health",
    "supplier_distance",
    "supplier_risk_score",
    "po_cycle_time",
    "lead_time_variability",
    "safety_stock_held",
    "raw_material_delay",
    "production_schedule_instability",
    "inventory_holding_cost",
]

EDGES = [
    ("geopolitical_disruption", "lead_time_variability"),
    ("geopolitical_disruption", "po_cycle_time"),
    ("supplier_financial_health", "supplier_risk_score"),
    ("supplier_distance", "lead_time_variability"),
    ("supplier_risk_score", "lead_time_variability"),
    ("po_cycle_time", "lead_time_variability"),
    ("lead_time_variability", "safety_stock_held"),
    ("lead_time_variability", "raw_material_delay"),
    ("safety_stock_held", "raw_material_delay"),
    ("raw_material_delay", "production_schedule_instability"),
    ("raw_material_delay", "inventory_holding_cost"),
]


def build_data_dictionary(output_path):
    data_dict = {
        "geopolitical_disruption":         {"type": "float", "range": "[0,1]",    "description": "Index of geopolitical/tariff-regime disruption severity affecting the supplier base.", "distribution": "Beta(2,5)"},
        "supplier_financial_health":       {"type": "float", "range": "[0,1]",    "description": "Composite financial-health score of the supplier (higher = healthier).", "distribution": "Beta(5,2)"},
        "supplier_distance":               {"type": "float", "range": "[0,inf)",  "description": "Geographic distance (km) from supplier to buyer facility.", "distribution": "Gamma(shape=2, scale=500)"},
        "supplier_risk_score":             {"type": "float", "range": "[0,100]",  "description": "Composite supplier risk score (higher = riskier).", "distribution": "SCM: f(supplier_financial_health) + noise"},
        "po_cycle_time":                   {"type": "float", "range": "[1,inf)",  "description": "Purchase order cycle time in days, from issue to supplier acknowledgment.", "distribution": "SCM: f(geopolitical_disruption) + noise"},
        "lead_time_variability":           {"type": "float", "range": "[0,inf)",  "description": "Variability (std-dev-like measure, days) in observed delivery lead times.", "distribution": "SCM: f(geopolitical_disruption, supplier_distance, supplier_risk_score, po_cycle_time) + noise"},
        "safety_stock_held":               {"type": "float", "range": "[0,inf)",  "description": "Units of safety stock held as a buffer against lead-time variability.", "distribution": "SCM: f(lead_time_variability) + noise"},
        "raw_material_delay":              {"type": "float", "range": "[0,inf)",  "description": "Observed delay (days) in raw material delivery.", "distribution": "SCM: f(lead_time_variability, safety_stock_held) + noise"},
        "production_schedule_instability": {"type": "float", "range": "[0,100]",  "description": "Index of production schedule disruption/instability.", "distribution": "SCM: f(raw_material_delay) + noise"},
        "inventory_holding_cost":          {"type": "float", "range": "[0,inf)",  "description": "Monthly inventory holding cost in $k.", "distribution": "SCM: f(raw_material_delay) + noise"},
    }

    if output_path:
        utils.write_json(data_dict, output_path)
    return data_dict

def build_ref_graph(output_path):
    return utils.build_graph(NODES, EDGES, output_path, 'Supplier-to-Manufacturer Ground Truth DAG')


def generate(n, seed):
    rng = utils.make_rng(seed)
    d = {}

    # Root nodes
    # geopolitical_disruption: Beta(2,5) — skewed toward low/moderate disruption;
    # severe shocks are rare (tariff-regime severity index, Tang 2006).
    d["geopolitical_disruption"] = rng.beta(2, 5, n)

    # supplier_financial_health: Beta(5,2) — most suppliers reasonably healthy;
    # left tail captures financially distressed minority (D&B health indices).
    d["supplier_financial_health"] = rng.beta(5, 2, n)

    # supplier_distance: Gamma(2,500) km — right-skewed; most suppliers regional,
    # occasional far-flung global suppliers.
    d["supplier_distance"] = rng.gamma(2, 500, n)

    # Non-root nodes in topological order
    d["supplier_risk_score"] = np.clip(
        scm.linear_combo(
            {"supplier_financial_health": d["supplier_financial_health"]},
            {"supplier_financial_health": -90.0}, intercept=95.0, noise_std=5.0, rng=rng,
        ), 0, 100,
    )

    d["po_cycle_time"] = np.clip(
        scm.linear_combo(
            {"geopolitical_disruption": d["geopolitical_disruption"]},
            {"geopolitical_disruption": 25.0}, intercept=10.0, noise_std=2.0, rng=rng,
        ), 1, None,
    )

    d["lead_time_variability"] = np.clip(
        scm.linear_combo(
            {
                "geopolitical_disruption": d["geopolitical_disruption"],
                "supplier_distance":       d["supplier_distance"],
                "supplier_risk_score":     d["supplier_risk_score"],
                "po_cycle_time":           d["po_cycle_time"],
            },
            {
                "geopolitical_disruption": 8.0,
                "supplier_distance":       0.004,
                "supplier_risk_score":     0.05,
                "po_cycle_time":           0.15,
            },
            intercept=1.0, noise_std=1.5, rng=rng,
        ), 0, None,
    )

    d["safety_stock_held"] = np.clip(
        scm.linear_combo(
            {"lead_time_variability": d["lead_time_variability"]},
            {"lead_time_variability": 15.0}, intercept=50.0, noise_std=10.0, rng=rng,
        ), 0, None,
    )

    # safety_stock_held moderates raw_material_delay (Silver et al. 1998)
    d["raw_material_delay"] = np.clip(
        scm.linear_combo(
            {
                "lead_time_variability": d["lead_time_variability"],
                "safety_stock_held":     d["safety_stock_held"],
            },
            {"lead_time_variability": 0.6, "safety_stock_held": -0.03},
            intercept=1.0, noise_std=1.0, rng=rng,
        ), 0, None,
    )

    d["production_schedule_instability"] = np.clip(
        scm.linear_combo(
            {"raw_material_delay": d["raw_material_delay"]},
            {"raw_material_delay": 4.0}, intercept=5.0, noise_std=5.0, rng=rng,
        ), 0, 100,
    )

    d["inventory_holding_cost"] = np.clip(
        scm.linear_combo(
            {"raw_material_delay": d["raw_material_delay"]},
            {"raw_material_delay": 2.5}, intercept=10.0, noise_std=3.0, rng=rng,
        ), 0, None,
    )

    return pd.DataFrame(d)