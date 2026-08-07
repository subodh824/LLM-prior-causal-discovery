import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
import pickle
import os
import json
import re
import io
import base64
from config import Config

def make_rng(seed):
    return np.random.default_rng(seed)

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return 

def read_json(filepath):
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None

def write_json(data, filepath):
    json_str = json.dumps(data, indent=2)
    json_str = re.sub(
        r'\[\n\s+([^\[\]{}]+?)\n\s*\]',
        lambda m: '[' + ', '.join(x.strip().rstrip(',') for x in m.group(1).split('\n')) + ']',
        json_str
    )
    with open(filepath, "w") as f:
        f.write(json_str)
    return

def write_pickle(pos, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(pos, f)

def read_pickle(pos_path):
    with open(pos_path, 'rb') as f:
        pos = pickle.load(f)
    return pos

def write_to_file(data, output_path):
    with open(output_path, 'wb') as f:
        f.write(data)

def build_graph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def draw_graph(G, title, pos=None):
    pos = nx.spring_layout(G, seed=42, k=1.8) if not pos else pos
    plt.figure(figsize=(16, 10))
    nx.draw(
        G, pos, with_labels=True, node_size=2000, node_color="#e8c39e",
        font_size=7, arrowsize=16, edge_color="#555555",
    )
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    png_bytes = buf.read()
    
    G_json = convert_graph_to_json(G)
    return png_bytes, G_json, pos

def convert_graph_to_json(G):
    if  not G:
        return None
    return {
        "nodes": [{"id": str(n)} for n in G.nodes()],
        "links": [{"source": str(u), "target": str(v)} for u, v in G.edges()],
    }

def convert_json_to_graph(json):
    if  not json:
        return None
    G = nx.DiGraph()
    for node in json["nodes"]:
        G.add_node(node["id"] if isinstance(node, dict) else node)
    for link in json.get("links", json.get("edges", [])):
        G.add_edge(link["source"], link["target"])
    return G

def build_graph_pos(data_dict):
    all_nodes = data_dict.keys()
    reference_graph = nx.complete_graph(all_nodes)
    pos = nx.spring_layout(reference_graph, seed=42)
    return pos

def ci95(values):
    values = np.asarray(values)
    if len(values) < 2 or np.all(values == values[0]):
        return None
    return round(stats.sem(values) * stats.t.ppf(0.975, df=len(values) - 1),2)

def safe_round(x, digits=2):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return round(x, digits)

def get_scm(leg):
    from dags import distributor_to_customer 
    from dags import manufacturer_to_distributor
    from dags import supplier_to_manufacturer
    from dags import dataco
    leg_to_scm = {
        Config.DISTRIBUTION_TO_CUSTOMER: distributor_to_customer,
        Config.MANUFACTURER_TO_DISTRIBUTOR: manufacturer_to_distributor,
        Config.SUPPLIER_TO_MANUFACTURER: supplier_to_manufacturer,
        Config.DATACO: dataco
    }
    return leg_to_scm[leg] if leg in leg_to_scm.keys() else None