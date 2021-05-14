
from common import data
from common import models

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
# Data

def load_dataset(name):
    """ Load real-world datasets, available in PyTorch Geometric.

    Used as a helper for DiskDataSource.
    """
    if name == "enzymes":
        dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
    elif name == "proteins":
        dataset = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
    elif name == "cox2":
        dataset = TUDataset(root="/tmp/cox2", name="COX2")
    elif name == "aids":
        dataset = TUDataset(root="/tmp/AIDS", name="AIDS")
    elif name == "reddit-binary":
        dataset = TUDataset(root="/tmp/REDDIT-BINARY", name="REDDIT-BINARY")
    elif name == "imdb-binary":
        dataset = TUDataset(root="/tmp/IMDB-BINARY", name="IMDB-BINARY")
    elif name == "firstmm_db":
        dataset = TUDataset(root="/tmp/FIRSTMM_DB", name="FIRSTMM_DB")
    elif name == "dblp":
        dataset = TUDataset(root="/tmp/DBLP_v1", name="DBLP_v1")
    elif name == "ppi":
        dataset = PPI(root="/tmp/PPI")
    elif name == "qm9":
        dataset = QM9(root="/tmp/QM9")
    elif name == "atlas":
        dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]
    
    train_len = int(0.8 * len(dataset))
    train, test = [], []
    dataset = list(dataset)
    random.shuffle(dataset)
    has_name = hasattr(dataset[0], "name")
    for i, graph in tqdm(enumerate(dataset)):
        if not type(graph) == nx.Graph:
            if has_name: del graph.name
            graph = pyg_utils.to_networkx(graph).to_undirected()
        if i < train_len:
            train.append(graph)
        else:
            test.append(graph)
    return train, test, "graph"

class GraphDS(Dataset):
    def __init__(self, dataset, min_size, max_size):


class DiskDataSource:
    def __init__(self, dataset_name, node_anchored, batch_size, min_size, max_size):
        self.node_anchored = node_anchored
        self.dataset  = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size

    def gen_batch(self):
        train_set, test_set, task = self.dataset
        graphs = train_set
        
        pos_a, pos_b = [], []
        pos_a_anchors, pos_b_anchors = [], []
        for i in range(batch_size//2):
            size = random.randint(self.min_size+1, self.max_size)
            graph, a = utils.sample_neigh(graphs, size)
            b        = a[:random.randint(self.min_size, len(a)-1)]

            if self.node_anchored:
                anchor = list(graph.nodes)[0]
                pos_a_anchors.append(anchor)
                pos_b_anchors.append(anchor)
            
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)
            pos_a.append(neigh_a)
            pos_b.append(neigh_b)
        
        neg_a, neg_b = [], []
        neg_a_anchors, neg_b_anchors = [], []
        while len(neg_a) < batch_size // 2:
            size = random.randint(self.min_size+1, self.max_size)
            graph_a, a = utils.sample_neigh(graphs, size)
            graph_b, b = utils.sample_neigh(graphs, random.randint(self.min_size, size - 1))

        if self.node_anchored:
            neg_a_anchors.append(list(graph_a.nodes)[0])
            neg_b_anchors.append(list(graph_b.nodes)[0])
        neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)
        neg_a.append(neigh_a)
        neg_b.append(neigh_b)







# - 
# model
model   = models.OrderEmbedder(input_dim=1, hidden_dim=args.hidden_dim, args=args).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))

ds_valid =  data.DiskDataSource((node_anchored=args.node_anchored)
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