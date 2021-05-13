
import argparse
import random
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim


from common import data
from common import models
from common import utils
from test import validation


# --
# Cli

def set_seeds(seed):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 111)
    _ = torch.cuda.manual_seed(seed + 222)
    _ = random.seed(seed + 333)

def to_numpy(x):
    return x.detach().cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--dataset",    type=str, default='syn-balanced')
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--n_batches",    type=int, default=1_000_000)
    parser.add_argument("--eval_interval", type=int, default=10)

    # model
    parser.add_argument("--conv_type", type=str,  default='SAGE')
    parser.add_argument("--method_type", type=str, default='order')
    parser.add_argument("--n_layers",   type=int, default=8)
    parser.add_argument("--hidden_dim",   type=int, default=64)
    parser.add_argument("--skip",        type=str,  default="learnable")
    parser.add_argument("--dropout",     type=float, default=0.0)
    parser.add_argument("--lr",           type=float,default = 1e-4)
    parser.add_argument("--margin",       type=float, default=0.1)
    parser.add_argument("--node_anchored", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default='ckpt/model.pt')
    return parser.parse_args()


# - 
# cli
args    = parse_args()
set_seeds(3)
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

# -
# model

model   = models.OrderEmbedder(input_dim=1, hidden_dim=args.hidden_dim, args=args).to(device)
filter_fn = filter(lambda p : p.requires_grad, model.parameters())
optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=0.0)
clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)
clf_crit  = nn.NLLLoss()

# - 
# data

ds_train = data.DiskDataSource("reddit-binary", node_anchored=args.node_anchored)
#ds_train = data.OTFSynDataSource(node_anchored=args.node_anchored)
loaders_train = ds_train.gen_data_loaders(
    size        = args.eval_interval * args.batch_size, 
    batch_size  = args.batch_size, 
    train       = True
)

ds_valid = data.OTFSynDataSource(node_anchored=args.node_anchored)
loaders_valid = ds_valid.gen_data_loaders(
    size        = 4096, 
    batch_size  = args.batch_size, 
    train       = False,
    use_distributed_sampling = False
)
test_pts = []
for batch_target, batch_neg_target, batch_neg_query in zip(*loaders_valid):
    pos_a, pos_b, neg_a, neg_b = ds_valid.gen_batch(
        batch_target     = batch_target,
        batch_neg_target = batch_neg_target, 
        batch_neg_query  = batch_neg_query, 
        train            = False
    )
    if pos_a:
        pos_a = pos_a.to(torch.device("cpu"))
        pos_b = pos_b.to(torch.device("cpu"))
    neg_a = neg_a.to(torch.device("cpu"))
    neg_b = neg_b.to(torch.device("cpu"))
    test_pts.append((pos_a, pos_b, neg_a, neg_b))

# -
# Train loop
t_total = 0
t_gen_batch = 0
t_embed_samples = 0
t_predict_subgraphs = 0
t_compute_loss = 0
t_compute_loss_cls = 0

for epoch in range(args.n_batches // args.eval_interval):


    start_all = torch.cuda.Event(enable_timing=True)
    end_all = torch.cuda.Event(enable_timing=True)
    start_all.record()

    # train
    for batch_target, batch_neg_target, batch_neg_query in tqdm(zip(*loaders_train)):

        model.train()
        model.zero_grad()
        
        t_gen_batch_s = time.time()
        # get input for model from "batches"
        # pos_a, pos_b, neg_a, neg_b = ds_train.gen_batch(
        #     batch_target     = batch_target,
        #     batch_neg_target = batch_neg_target, 
        #     batch_neg_query  = batch_neg_query, 
        #     train            = True
        # ) ARGS ARE DIFF FOR REAL DS, but SAME COUNT SUPER DUMB.
        pos_a, pos_b, neg_a, neg_b = ds_train.gen_batch(
            batch_target, batch_neg_target, batch_neg_query, True
        )
        t_gen_batch += (time.time() - t_gen_batch_s)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # embed pos / negative samples
        emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
        emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)

        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        end.record()
        torch.cuda.synchronize()
        t_embed_samples += (start.elapsed_time(end) / 100)

        # create labels
        labels = torch.tensor([1]*pos_a.num_graphs + [0]*neg_a.num_graphs).to(utils.get_device())
        intersect_embs = None

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # predict subgraphs
        pred = model(emb_as, emb_bs)
        end.record()
        torch.cuda.synchronize()
        t_predict_subgraphs += (start.elapsed_time(end) / 100)

        # compute loss on embeddings (margin loss) / backprop
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tt = time.time()
        loss = model.criterion(pred, intersect_embs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        t_compute_loss += (start.elapsed_time(end) / 100)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # compute secondary loss for classifier
        with torch.no_grad():
            pred = model.predict(pred)
        model.clf_model.zero_grad()
        pred     = model.clf_model(pred.unsqueeze(1))
        clf_loss = clf_crit(pred, labels)
        clf_loss.backward()
        clf_opt.step()

        end.record()
        torch.cuda.synchronize()
        t_compute_loss_cls += (start.elapsed_time(end) / 100)

        # compute acc / loss  for logging
        acc        = torch.mean((pred.argmax(dim=-1) == labels).type(torch.float))
        train_loss = to_numpy(loss)
        clf_loss   = to_numpy(clf_loss)
        train_acc  = to_numpy(acc)

    
    print("epoch", epoch, "acc", train_acc, "train_loss", train_loss, "clf_loss", clf_loss, "time")#, time.time() - t_start)
    
    end_all.record()
    torch.cuda.synchronize()
    t_total += (start_all.elapsed_time(end_all) / 100)    
    t_overhead = t_total - (t_gen_batch + t_embed_samples + t_predict_subgraphs + t_compute_loss + t_compute_loss_cls)
    print("gen_batch", t_gen_batch / t_total)
    print("embed", t_embed_samples / t_total) 
    print("pred_subg", t_predict_subgraphs / t_total)
    print("t_compute_losss", t_compute_loss / t_total)
    print("t_compute_clsloss", t_compute_loss_cls / t_total)
    print("t_overhead (get data)", t_overhead / t_total)

    
    # val
    #validation(args, model, test_pts, logger, batch_n, epoch)

