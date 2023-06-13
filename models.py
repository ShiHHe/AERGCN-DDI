
import dgl.function as fn
import dgl
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
import numpy as np

import math

class RGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=4, dropout=0.3):
        super(RGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(600, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)
        self.rgcn2 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, 4)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn1(x, edge_index, edge_type)

        x = self.linear_output2(x)

        return x


class RGCN64(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=64, relation_num=65, dropout=0.3):
        super(RGCN64, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(600, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)
        self.rgcn2 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, 65)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn1(x, edge_index, edge_type)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


bert_n_heads=4
len_after_AE=128
drop_out_rating=0.3
class AE1(torch.nn.Module):  # Joining together
    def __init__(self, vector_size):
        super(AE1, self).__init__()

        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        # X = self.att2(X)
        # X = self.l2(X)
        # X_AE = self.dr(self.bn3(self.ac(self.l3(X))))
        # X_AE = self.l4(X_AE)
        X_AE = 0
        return X, X_AE




class AERGCNDDI(nn.Module):
    def __init__(self,input_dim, embedding_dimension=16, hidden_dimension=128, out_dim=64, relation_num=65, dropout=0.3):
        super(AERGCNDDI, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            # nn.Linear(600, hidden_dimension),
            nn.Linear(600, 500),
            nn.LeakyReLU()
        )

        self.rgcn1 = RGCNConv(500, 400, num_relations=relation_num,bias=False)
        self.rgcn2 = RelGraphConv(300, 300, 65, regularizer='basis',
                                  num_bases=65, self_loop=False)
        self.rgcn3 = RelGraphConv(300, 300, 4, regularizer='basis',
                                  num_bases=4, self_loop=False)

        self.linear_output2 = nn.Linear(300,65)
        self.ae1 = AE1(input_dim)  # Joining together
        n_heads =8
        self.attn = MultiHeadAttention(300, n_heads)
        self.AN1 = torch.nn.LayerNorm(300)

    def forward(self,x_feature, feature, edge_index, edge_type):

        g = dgl.graph((edge_index[0], edge_index[1]))
        # x_feature = x_feature[1:g.num_nodes()+1].cuda()
        x_feature = x_feature[0:g.num_nodes()].cuda()
        edge_type = edge_type.cuda()
        x = self.rgcn2(g, x_feature, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        g.ndata['h'] = x
        g.edata['x'] = edge_type
        g.apply_edges(fn.u_add_v('h', 'h', 'x'))
        x = g.edata['x']

        output = self.attn(x)
        x = self.AN1(output + x)
        x = self.linear_output2(x)
        return x



class RGCN64more(nn.Module):
    def __init__(self,input_dim, embedding_dimension=16, hidden_dimension=128, out_dim=64, relation_num=65, dropout=0.3):
        super(RGCN64more, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            # nn.Linear(600, hidden_dimension),
            nn.Linear(600, 500),
            nn.LeakyReLU()
        )

        self.rgcn1 = RGCNConv(500, 400, num_relations=relation_num,bias=False)
        self.rgcn2 = RelGraphConv(300, 300, 4, regularizer='basis',
                                  num_bases=4, self_loop=False)

        self.linear_output2 = nn.Linear(600, 4)
        self.ae1 = AE1(input_dim)  # Joining together
        n_heads =8
        # self.attn = MultiHeadAttention(input_dim, n_heads)
        self.attn = MultiHeadAttention(600, n_heads)
        # self.AN1 = torch.nn.LayerNorm(input_dim)
        self.AN1 = torch.nn.LayerNorm(600)

    def forward(self,x_feature, feature, edge_index, edge_type):

        g = dgl.graph((edge_index[0], edge_index[1]))
        # x_feature = x_feature[1:g.num_nodes()+1].cuda()
        x_feature = x_feature[1:g.num_nodes() + 1].cuda()
        edge_type = edge_type.cuda()

        x = self.rgcn2(g, x_feature, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        g.ndata['h'] = x
        g.edata['x'] = edge_type
        g.apply_edges(fn.u_add_v('h', 'h', 'x'))
        x = g.edata['x']
        output = self.attn(x)
        x = self.AN1(output + x)
        x = self.linear_output2(x)
        return x




class GAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=4, relation_num=2, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension*2, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gat1 = GATConv(hidden_dimension, int(hidden_dimension / 8), heads=8)
        self.gat2 = GATConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, 4)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class GCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension*2, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gcn1 = GCNConv(hidden_dimension, hidden_dimension)
        self.gcn2 = GCNConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, 4)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.linear_output2(x)

        return x



class SAGE(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=4, relation_num=2, dropout=0.3):
        super(SAGE, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension*2, hidden_dimension),
            nn.LeakyReLU()
        )

        self.sage1 = SAGEConv(hidden_dimension, hidden_dimension)
        self.sage2 = SAGEConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, 4)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.sage1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        x = self.linear_output2(x)

        return x

