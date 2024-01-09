import os.path as osp
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData, Data
from torch_geometric.data import Dataset, download_url
import csv

class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()
    
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge_csv(path, encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    edge_index = torch.tensor([df['source'], df['des']])
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


x, m = load_node_csv(path = '/DATA1/bzhu/CXR-Report-Generation/PyG_dataset/entity.csv', index_col = 'ID', encoders={'entity': SequenceEncoder()})
edge_index, edge_label = load_edge_csv(path = '/DATA1/bzhu/CXR-Report-Generation/PyG_dataset/triplet.csv',encoders={'relation': SequenceEncoder()})
data = Data(x=x, edge_index=edge_index, edge_attr=edge_label)
# data['entity'].x = x
# data['entity', 'relation', 'entity'].edge_index = edge_index
# data['entity', 'relation', 'entity'].edge_label = edge_label
print(data)

# import torch
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.data import Data

# # 将数据对象转换为networkx格式的图
# G = nx.Graph()
# G.add_nodes_from(range(data.num_nodes))
# G.add_edges_from(data.edge_index.t().tolist())
# import random
# random_nodes = random.sample(G.nodes, 200)
# subgraph = G.subgraph(random_nodes)
# # 可视化图
# pos = nx.spring_layout(G)
# nx.draw(G, pos, node_size=5, width=0.1)
# plt.savefig('result.png')
