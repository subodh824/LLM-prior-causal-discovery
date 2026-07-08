import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats
import pickle
import os
import json

def make_rng(seed):
    return np.random.default_rng(seed)

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return 

def read_json(filepath):
    with open(filepath) as f:
        return json.load(f)

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

def build_graph(nodes, edges, output_path=None, title='', save_pos=True):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if output_path:
        _, pos =  save_graph(G, output_path, title)
        write_pickle(G, os.path.join(os.path.dirname(output_path), 'ref_dag.pkl'))
        if save_pos:
            write_pickle(pos,  os.path.join(os.path.dirname(output_path), 'ref_dag_pos.pkl'))
        
    return G, None

def write_pickle(pos, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(pos, f)

def read_pickle(pos_path):
    with open(pos_path, 'rb') as f:
        pos = pickle.load(f)
    return pos

def save_graph(G, path, title, pos=None):
    pos = nx.spring_layout(G, seed=42, k=1.8) if not pos else pos
    plt.figure(figsize=(16, 10))
    nx.draw(
        G, pos, with_labels=True, node_size=2000, node_color="#e8c39e",
        font_size=7, arrowsize=16, edge_color="#555555",
    )
    plt.title(title)
    plt.savefig(path, dpi=150)
    plt.close()
    return G, pos

def build_graph_pos(data_dict):
    all_nodes = data_dict.keys()
    reference_graph = nx.complete_graph(all_nodes)
    pos = nx.spring_layout(reference_graph, seed=42)
    return pos

def ci95(values):
    values = np.asarray(values)
    if len(values) < 2 or np.all(values == values[0]):
        return None
    return stats.sem(values) * stats.t.ppf(0.975, df=len(values) - 1)