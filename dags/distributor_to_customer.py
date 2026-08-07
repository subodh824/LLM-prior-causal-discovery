import networkx as nx
from common import scm, utils
import pandas as pd
import numpy as np
from config import Config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

NODES = [
    "order_volume_peak_season",
    "weather_severity",
    "geographic_distance",
    "order_item_quantity",
    "order_processing_time",
    "fulfillment_center_congestion",
    "shipping_mode",
    "scheduled_transit_time",
    "traffic_congestion",
    "actual_transit_time",
    "delivery_delay",
    "late_delivery_risk",
    "order_discount_rate",
    "customer_satisfaction",
    "profit_ratio",
    "churn_repeat_purchase_risk",
]

EDGES = [
    ("order_volume_peak_season",      "order_item_quantity"),
    ("order_volume_peak_season",      "fulfillment_center_congestion"),
    ("weather_severity",              "actual_transit_time"),
    ("weather_severity",              "traffic_congestion"),
    ("geographic_distance",           "actual_transit_time"),
    ("geographic_distance",           "shipping_mode"),
    ("order_item_quantity",           "fulfillment_center_congestion"),
    ("order_processing_time",         "delivery_delay"),
    ("shipping_mode",                 "scheduled_transit_time"),
    ("shipping_mode",                 "actual_transit_time"),
    ("fulfillment_center_congestion", "scheduled_transit_time"),
    ("traffic_congestion",            "actual_transit_time"),
    ("scheduled_transit_time",        "delivery_delay"),
    ("actual_transit_time",           "delivery_delay"),
    ("delivery_delay",                "late_delivery_risk"),
    ("delivery_delay",                "customer_satisfaction"),
    ("customer_satisfaction",         "churn_repeat_purchase_risk"),
    ("order_discount_rate",           "profit_ratio"),
]

DATA_DICTIONARY = {
    "order_volume_peak_season":      {"type": "float", "range": "[0,inf)",   "description": "Seasonal order volume index (annual cycle, e.g. holiday peaks).",                        "distribution": "Seasonal sinusoid + Gaussian noise"},
    "weather_severity":              {"type": "float", "range": "[0,inf)",   "description": "Weather severity index along the shipping route.",                                         "distribution": "Gamma(shape=1.2, scale=2.0)"},
    "geographic_distance":           {"type": "float", "range": "[0,inf)",   "description": "Shipping distance in km.",                                                                 "distribution": "Gamma(shape=2.0, scale=400.0)"},
    "order_item_quantity":           {"type": "float", "range": "[1,inf)",   "description": "Number of items in the order.",                                                            "distribution": "SCM: f(order_volume_peak_season) + noise"},
    "order_processing_time":         {"type": "float", "range": "[0,inf)",   "description": "Order processing time (hours) from order placed to dispatch-ready.",                      "distribution": "Gamma(shape=2.0, scale=4.0)"},
    "fulfillment_center_congestion": {"type": "float", "range": "[0,100]",   "description": "Index of fulfillment center congestion.",                                                  "distribution": "SCM: f(order_volume_peak_season, order_item_quantity) + noise"},
    "shipping_mode":                 {"type": "float", "range": "[0,4]",     "description": "Ordinal shipping mode (0=standard ground .. 4=same-day).",                               "distribution": "SCM: f(geographic_distance) + noise"},
    "scheduled_transit_time":        {"type": "float", "range": "[0.5,inf)", "description": "Carrier-promised transit time in days.",                                                  "distribution": "SCM: f(shipping_mode, fulfillment_center_congestion) + noise"},
    "traffic_congestion":            {"type": "float", "range": "[0,100]",   "description": "Road traffic congestion index along the route.",                                           "distribution": "SCM: f(weather_severity) + noise"},
    "actual_transit_time":           {"type": "float", "range": "[0.5,inf)", "description": "Realized transit time in days.",                                                          "distribution": "SCM: f(weather_severity, geographic_distance, shipping_mode, traffic_congestion) + noise"},
    "delivery_delay":                {"type": "float", "range": "[-5,inf)",  "description": "Delivery delay in days (actual - scheduled, plus OPT effect).",                           "distribution": "SCM: linear transit gap + CONCAVE f(order_processing_time) + noise"},
    "late_delivery_risk":            {"type": "float", "range": "[0,1]",     "description": "Probability/flag-like risk score of late delivery.",                                       "distribution": "SCM: f(delivery_delay) + noise"},
    "order_discount_rate":           {"type": "float", "range": "[0,0.5]",   "description": "Discount rate applied to the order.",                                                     "distribution": "Beta(2,5) scaled to [0,0.5]"},
    "customer_satisfaction":         {"type": "float", "range": "[1,5]",     "description": "Post-purchase customer satisfaction rating.",                                              "distribution": "SCM: f(delivery_delay) + noise"},
    "profit_ratio":                  {"type": "float", "range": "[-0.5,1]",  "description": "Profit ratio for the order.",                                                             "distribution": "SCM: f(order_discount_rate) + noise"},
    "churn_repeat_purchase_risk":    {"type": "float", "range": "[0,1]",     "description": "Risk of customer churn / not repeat-purchasing.",                                         "distribution": "SCM: f(customer_satisfaction) + noise"},
}

def build_data_dictionary(output_dir):
    data_dict = {
        'leg': 'distributor_to_customer',
        'natural': []
    }
    for col, info in DATA_DICTIONARY.items():
        data_dict['natural'].append({ 'name': col } | info)

    if output_dir:
        utils.write_json(data_dict, output_dir +  '/data_dictionary.json')
    return data_dict

def build_downstream_config(output_dir=None):
    cfg = {
        "target": "delivery_delay",
        "causes": ["fulfillment_center_congestion", "weather_severity"],
        "actionables": [
            "order_processing_time",
            "shipping_mode",
            "scheduled_transit_time",
            "fulfillment_center_congestion",
        ]
    }

    if output_dir:
        utils.write_json(cfg, output_dir + "/downstream_config.json")
    return cfg

def build_ref_graph(output_dir):
    G = utils.build_graph(NODES, EDGES)
    png_bytes, G_json, pos = utils.draw_graph(G, 'Distributor-to-Customer Ground Truth DAG')
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
    # order_volume_peak_season: seasonal sinusoid (annual cycle) + noise.
    t = rng.uniform(0, 2 * np.pi, n)
    d["order_volume_peak_season"] = value("order_volume_peak_season", np.clip(50 + 30 * np.sin(t) + rng.normal(0, 8, n), 0, None))

    # weather_severity: Gamma — most days mild, occasional severe weather.
    d["weather_severity"] = value("weather_severity", rng.gamma(1.2, 2.0, n))

    # geographic_distance (km): Gamma — right-skewed shipping distances.
    d["geographic_distance"] = value("geographic_distance", rng.gamma(2.0, 400.0, n))

    # order_discount_rate (fraction 0-0.5): Beta(2,5) scaled.
    d["order_discount_rate"] = value("order_discount_rate", rng.beta(2, 5, n) * 0.5)

    # order_processing_time (hours): root node (no DAG parents in this leg).
    d["order_processing_time"] = value("order_processing_time", rng.gamma(2.0, 4.0, n))

    # Non-root nodes in topological order
    d["order_item_quantity"] = value("order_processing_time", np.clip(
        scm.linear_combo(
            {"order_volume_peak_season": d["order_volume_peak_season"]},
            {"order_volume_peak_season": 0.4}, intercept=2.0, noise_std=1.5, rng=rng,
        ), 1, None,
    ))

    d["shipping_mode"] = value("shipping_mode", np.clip(
        scm.linear_combo(
            {"geographic_distance": d["geographic_distance"]},
            {"geographic_distance": 0.0015}, intercept=1.0, noise_std=0.5, rng=rng,
        ), 0, 4,
    ))

    d["fulfillment_center_congestion"] = value("fulfillment_center_congestion", np.clip(
        scm.linear_combo(
            {
                "order_volume_peak_season": d["order_volume_peak_season"],
                "order_item_quantity":      d["order_item_quantity"],
            },
            {"order_volume_peak_season": 0.5, "order_item_quantity": 2.0},
            intercept=5.0, noise_std=5.0, rng=rng,
        ), 0, 100,
    ))

    d["traffic_congestion"] = value("traffic_congestion", np.clip(
        scm.linear_combo(
            {"weather_severity": d["weather_severity"]},
            {"weather_severity": 8.0}, intercept=20.0, noise_std=8.0, rng=rng,
        ), 0, 100,
    ))

    d["scheduled_transit_time"] = value("scheduled_transit_time", np.clip(
        scm.linear_combo(
            {
                "shipping_mode":                 d["shipping_mode"],
                "fulfillment_center_congestion": d["fulfillment_center_congestion"],
            },
            {"shipping_mode": -1.0, "fulfillment_center_congestion": 0.04},
            intercept=5.0, noise_std=0.8, rng=rng,
        ), 0.5, None,
    ))

    d["actual_transit_time"] = value("actual_transit_time", np.clip(
        scm.linear_combo(
            {
                "weather_severity":    d["weather_severity"],
                "geographic_distance": d["geographic_distance"],
                "shipping_mode":       d["shipping_mode"],
                "traffic_congestion":  d["traffic_congestion"],
            },
            {
                "weather_severity":    0.4,
                "geographic_distance": 0.002,
                "shipping_mode":       -0.8,
                "traffic_congestion":  0.03,
            },
            intercept=2.0, noise_std=1.0, rng=rng,
        ), 0.5, None,
    ))

    opt_contribution = scm.concave_transform(d["order_processing_time"], scale=2.0, shift=0.0)
    transit_gap = d["actual_transit_time"] - d["scheduled_transit_time"]
    d["delivery_delay"] = value("delivery_delay", np.clip(
        transit_gap + opt_contribution + rng.normal(0, 0.7, n), -5, None,
    ))

    d["late_delivery_risk"] = value("late_delivery_risk", np.clip(
        scm.linear_combo(
            {"delivery_delay": d["delivery_delay"]},
            {"delivery_delay": 0.18}, intercept=0.05, noise_std=0.08, rng=rng,
        ), 0, 1,
    ))

    d["customer_satisfaction"] = value("customer_satisfaction", np.clip(
        scm.linear_combo(
            {"delivery_delay": d["delivery_delay"]},
            {"delivery_delay": -0.35}, intercept=4.3, noise_std=0.4, rng=rng,
        ), 1, 5,
    ))

    d["churn_repeat_purchase_risk"] = value("churn_repeat_purchase_risk", np.clip(
        scm.linear_combo(
            {"customer_satisfaction": d["customer_satisfaction"]},
            {"customer_satisfaction": -0.15}, intercept=0.8, noise_std=0.1, rng=rng,
        ), 0, 1,
    ))

    d["profit_ratio"] = value("profit_ratio", np.clip(
        scm.linear_combo(
            {"order_discount_rate": d["order_discount_rate"]},
            {"order_discount_rate": -0.8}, intercept=0.3, noise_std=0.05, rng=rng,
        ), -0.5, 1,
    ))

    return pd.DataFrame(d)


