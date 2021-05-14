"""Build an alignment matrix for matching a query subgraph in a target graph.
Subgraph matching model needs to have been trained with the node-anchored option
(default)."""

import argparse
from itertools import permutations
import pickle
from queue import PriorityQueue
import os
import random
import time

from deepsnap.batch import Batch
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn

from common import data
from common import models
from common import utils
from config import parse_encoder
from test import validation
from train import build_model
import itertools
from matplotlib.patches import Patch


import faiss


def to_numpy(x):
    return x.detach().cpu().numpy()

def gen_alignment_matrix(model, query, target, method_type="order"):
    """Generate subgraph matching alignment matrix for a given query and
    target graph. Each entry (u, v) of the matrix contains the confidence score
    the model gives for the query graph, anchored at u, being a subgraph of the
    target graph, anchored at v.

    Args:
        model: the subgraph matching model. Must have been trained with
            node anchored setting (--node_anchored, default)
        query: the query graph (networkx Graph)
        target: the target graph (networkx Graph)
        method_type: the method used for the model.
            "order" for order embedding or "mlp" for MLP model
    """

    tt = time.time()
    e_q, e_v = [], []
    for i, u in enumerate(query.nodes):
        batch = utils.batch_nx_graphs([query], anchors=[u])
        e_q.append(to_numpy(model.emb_model(batch)))

    for i, v in enumerate(target.nodes):
        batch = utils.batch_nx_graphs([target], anchors=[v])
        e_v.append(to_numpy(model.emb_model(batch)))
    
    tt_emb = (time.time() - tt)

    tt = time.time()
    for u,v in itertools.product(e_q, e_v):
        u = torch.Tensor(u).to('cuda')
        v = torch.Tensor(v).to('cuda')
        raw_pred = torch.log(model.predict([u, v]))
    tt_pointwise = (time.time() - tt) + tt_emb


    tt = time.time()
    d = np.concatenate(e_v, axis=0)
    index = faiss.IndexFlatL2(d.shape[1])
    index.add(d)
    xq = np.concatenate(e_q, axis=0)
    D, I = index.search(xq, 1)
    tt_faiss = (time.time() - tt) + tt_emb

    tt = time.time()
    mat = np.zeros((len(query), len(target)))
    for i, u in enumerate(query.nodes):
        for j, v in enumerate(target.nodes):
            batch = utils.batch_nx_graphs([query, target], anchors=[u, v])
            embs = model.emb_model(batch)
            pred = model(embs[1].unsqueeze(0), embs[0].unsqueeze(0))
            raw_pred = model.predict(pred)
            if method_type == "order":
                raw_pred = torch.log(raw_pred)
            elif method_type == "mlp":
                raw_pred = raw_pred[0][1]
            mat[i][j] = raw_pred.item()

    tt_old = (time.time() - tt)
    return mat, tt_pointwise, tt_faiss, tt_old

def main():
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    parser = argparse.ArgumentParser(description='Alignment arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument('--query_path', type=str, help='path of query graph',
        default="")
    parser.add_argument('--target_path', type=str, help='path of target graph',
        default="")
    args = parser.parse_args()
    args.test = True
    
    model = build_model(args)


    counts = [16, 64, 128, 256, 512]
    q_count = [8]
    fig, ax = plt.subplots(1,len(q_count), figsize=(8,8))
    for q_idx, q in enumerate(q_count):
        t_pointwise, t_faiss, t_old = [], [], []
        for ct in counts:
            target = nx.gnp_random_graph(ct, 0.25)
            query = nx.gnp_random_graph(q, 0.25)

            mat,  tt_pointwise, tt_faiss, tt_old = gen_alignment_matrix(model, query, target, method_type=args.method_type)
            print("CT", ct, tt_pointwise, tt_faiss, tt_old)
            t_pointwise.append(tt_pointwise)
            t_faiss.append(tt_faiss)
            t_old.append(tt_old)
            
        
        ax.plot(counts, t_pointwise, c='r')
        ax.plot(counts, t_faiss, c='b')
        ax.plot(counts, t_old, c='g')
    legend_elements = [
        Patch(facecolor='r', edgecolor='r', label='pointwise'),
        Patch(facecolor='b', edgecolor='b', label='ANN'),
        Patch(facecolor='g', edgecolor='g', label='orig. implementation')
        ]
    plt.ylabel("time (s)")
    plt.xlabel("size of target graph")
    plt.legend(handles=legend_elements, loc='lower right')
    fig.savefig("plots/alignment.png")


        # np.save("results/alignment.npy", mat)
        # print("Saved alignment matrix in results/alignment.npy")

        # plt.imshow(mat, interpolation="nearest")
        # plt.savefig("plots/alignment.png")
        # print("Saved alignment matrix plot in plots/alignment.png")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()

