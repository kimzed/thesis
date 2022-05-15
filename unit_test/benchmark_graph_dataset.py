
#from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import NodePropPredDataset



dataset = NodePropPredDataset(name = "ogbn-proteins", root="benchmark_graph_dataset")

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0] # pyg graph object

i_graph_data = 0
graph_data = graph[i_graph_data]

a = graph[1]
5+5
# graph[0]
"dict_keys(['edge_index', 'edge_feat', 'node_feat', 'node_species', 'num_nodes'])"