import networkx as nx

def _skeleton(G):
    return {frozenset(e) for e in G.edges()}


def calculate_shd(true_G, est_G):
    true_edges = set(true_G.edges())
    est_edges = set(est_G.edges())
    true_skel = _skeleton(true_G)
    est_skel = _skeleton(est_G)

    missing = true_skel - est_skel
    extra = est_skel - true_skel
    common = true_skel & est_skel
    wrong_direction = 0
    for fs in common:
        a, b = tuple(fs)
        true_dir_ab = (a, b) in true_edges
        est_dir_ab = (a, b) in est_edges
        if true_dir_ab != est_dir_ab:
            wrong_direction += 1

    return len(missing) + len(extra) + wrong_direction


def precision_recall_f1(true_G, est_G, directed=True):
    if directed:
        true_set = set(true_G.edges())
        est_set = set(est_G.edges())
    else:
        true_set = _skeleton(true_G)
        est_set = _skeleton(est_G)

    tp = len(true_set & est_set)
    fp = len(est_set - true_set)
    fn = len(true_set - est_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}