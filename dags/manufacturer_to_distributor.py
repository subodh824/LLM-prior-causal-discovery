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
    "machine_downtime",
    "demand_volatility",
    "material_delay",
    "production_schedule_stability",
    "production_volume",
    "factory_issues",
    "vehicle_fill_rate",
    "sales_order_quantity",
    "delivery_to_distributor",
]

EDGES = [
    ("machine_downtime",              "production_schedule_stability"),
    ("demand_volatility",             "production_schedule_stability"),
    ("demand_volatility",             "production_volume"),
    ("material_delay",                "production_volume"),
    ("production_schedule_stability", "factory_issues"),
    ("production_volume",             "vehicle_fill_rate"),
    ("factory_issues",                "delivery_to_distributor"),
    ("vehicle_fill_rate",             "delivery_to_distributor"),
    ("sales_order_quantity",          "delivery_to_distributor"),
]

DATA_DICTIONARY = {
    "machine_downtime":              {"type": "float", "range": "[0,inf)",  "description": "Weekly machine downtime in hours.",                                        "distribution": "Gamma(shape=1.5, scale=3.0)"},
    "demand_volatility":             {"type": "float", "range": "[0,1]",    "description": "Index of downstream demand volatility.",                                    "distribution": "Beta(2,2)"},
    "material_delay":                {"type": "float", "range": "[0,inf)",  "description": "Input material delay in days.",                                             "distribution": "Gamma(shape=2.0, scale=1.5)"},
    "production_schedule_stability": {"type": "float", "range": "[0,100]",  "description": "Index of production schedule stability (higher = more stable).",           "distribution": "SCM: f(machine_downtime, demand_volatility) + noise"},
    "production_volume":             {"type": "float", "range": "[0,inf)",  "description": "Units produced.",                                                           "distribution": "SCM: f(demand_volatility, material_delay) + noise"},
    "factory_issues":                {"type": "float", "range": "[0,100]",  "description": "Index of factory-floor operational issues.",                                "distribution": "SCM: f(production_schedule_stability) + noise"},
    "vehicle_fill_rate":             {"type": "float", "range": "[0,1]",    "description": "Fraction of outbound vehicle capacity utilized.",                           "distribution": "SCM: f(production_volume) + noise"},
    "sales_order_quantity":          {"type": "float", "range": "[0,inf)",  "description": "Units ordered by distributor/customer.",                                    "distribution": "Gamma(shape=3.0, scale=100.0)"},
    "delivery_to_distributor":       {"type": "float", "range": "[0,inf)",  "description": "Delivery time to distributor in days.",                                     "distribution": "SCM: f(factory_issues, vehicle_fill_rate, sales_order_quantity) + noise"},
}

def build_data_dictionary(output_dir):
    data_dict = {
        'leg': 'manufacturer_to_distributor',
        'natural': []
    }
    for col, info in DATA_DICTIONARY.items():
        data_dict['natural'].append({ 'name': col } | info)

    if output_dir:
        utils.write_json(data_dict, output_dir +  '/data_dictionary.json')
    return data_dict

def build_downstream_config(output_dir=None):
    cfg = {
        "target": "delivery_to_distributor",
        "causes": ["machine_downtime", "demand_volatility"],
        "actionables": [
            "production_schedule_stability",
            "production_volume",
        ]
    }

    if output_dir:
        utils.write_json(cfg, output_dir + "/downstream_config.json")
    return cfg


def build_ref_graph(output_dir):
    G = utils.build_graph(NODES, EDGES)
    png_bytes, G_json, pos = utils.draw_graph(G, 'Manufacturer-to-Distributor Ground Truth DAG')
    utils.write_to_file(png_bytes, output_dir + '/ref_dag.png')
    utils.write_json(G_json, output_dir + '/ref_dag.json')
    return G

def generate(n, seed, interventions=None):
    rng = utils.make_rng(seed)

    do = interventions or {}

    def value(name, computed):
        if name in do:
            return np.full(n, float(do[name]))
        return computed

    d = {}

    # Root nodes
    # machine_downtime (hours/week): Gamma — most weeks low downtime, occasional spikes.
    d["machine_downtime"] = value("machine_downtime", rng.gamma(1.5, 3.0, n))

    # demand_volatility (index 0-1): Beta(2,2) — symmetric, moderate volatility typical.
    d["demand_volatility"] = value("demand_volatility", rng.beta(2, 2, n))

    # material_delay (days): Gamma — right-skewed, mostly small delays.
    d["material_delay"] = value("material_delay", rng.gamma(2.0, 1.5, n))

    # sales_order_quantity (units): Gamma — right-skewed order sizes.
    d["sales_order_quantity"] = value("sales_order_quantity", rng.gamma(3.0, 100.0, n))

    # Non-root nodes in topological order
    d["production_schedule_stability"] = value("production_schedule_stability", np.clip(
        scm.linear_combo(
            {"machine_downtime": d["machine_downtime"], "demand_volatility": d["demand_volatility"]},
            {"machine_downtime": -3.0, "demand_volatility": -25.0},
            intercept=95.0, noise_std=5.0, rng=rng,
        ), 0, 100,
    ))

    d["production_volume"] = value("production_volume", np.clip(
        scm.linear_combo(
            {"demand_volatility": d["demand_volatility"], "material_delay": d["material_delay"]},
            {"demand_volatility": -150.0, "material_delay": -40.0},
            intercept=1000.0, noise_std=60.0, rng=rng,
        ), 0, None,
    ))

    d["factory_issues"] = value("factory_issues", np.clip(
        scm.linear_combo(
            {"production_schedule_stability": d["production_schedule_stability"]},
            {"production_schedule_stability": -0.7},
            intercept=70.0, noise_std=6.0, rng=rng,
        ), 0, 100,
    ))

    d["vehicle_fill_rate"] = value("vehicle_fill_rate", np.clip(
        scm.linear_combo(
            {"production_volume": d["production_volume"]},
            {"production_volume": 0.0006},
            intercept=0.3, noise_std=0.05, rng=rng,
        ), 0, 1,
    ))

    d["delivery_to_distributor"] = value("delivery_to_distributor", np.clip(
        scm.linear_combo(
            {
                "factory_issues":       d["factory_issues"],
                "vehicle_fill_rate":    d["vehicle_fill_rate"],
                "sales_order_quantity": d["sales_order_quantity"],
            },
            {"factory_issues": 0.08, "vehicle_fill_rate": -3.0, "sales_order_quantity": 0.002},
            intercept=3.0, noise_std=1.0, rng=rng,
        ), 0, None,
    ))

    return pd.DataFrame(d)