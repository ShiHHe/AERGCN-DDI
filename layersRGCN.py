# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        '''
        GCN - ChebNet's first-order approximation
        :param input: (B, N, in_F)
        :param adj: (N， N)
        :return: (B, N, out_F)
        '''

        support = torch.matmul(input, self.weight)

        output = torch.matmul(adj, support)

        return output + self.bias if self.bias is not None else output


class GCN_layer(nn.Module):
    def __init__(self, gcn1_in_feature, gcn1_out_feature, gcn2_out_feature):
        super(GCN_layer, self).__init__()
        self.gc1 = GraphConvolution(gcn1_in_feature, gcn1_out_feature)
        self.gc2 = GraphConvolution(gcn1_out_feature, gcn2_out_feature)

    def forward(self, x, adj):
        '''
        GCN for each timestep.
        :param x: (B, N, in_F, T)
        :param adj: (N， N)
        :return: (B, N, gcn_F, T)
        '''

        batch_size, node, in_channels, timesteps = x.shape

        gcn_outputs = []

        gcn_1 = self.gc1(x , adj)  #


        for time_step in range(timesteps):
            gcn_1 = self.gc1(x[:, :, :, time_step], adj)  # (B, N, in_F) - (B, N, gcn1_F)

            gcn_2 = self.gc2(gcn_1, adj)  # (B, N, gcn1_F) - (B, N, gcn_F)

            gcn_outputs.append(gcn_2.unsqueeze(-1))  # (B, N, gcn_F) - (B, N, gcn_F, 1)

        return F.relu(torch.cat(gcn_outputs, dim=-1))  # (B, N, gcn_F, T)


class Temporal_Attention(nn.Module):
    def __init__(self, DEVICE, in_channels, nodes, timesteps):
        super(Temporal_Attention, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(nodes).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, nodes).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, timesteps, timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(timesteps, timesteps).to(DEVICE))

    def forward(self, x):
        '''
        ASTGCN - temporal attention
        :param x: (B, N, in_F, T)
        :return: (B, T, T)
        '''

        # x:(B, N, in_F, T) -> (B, T, in_F, N)
        # (B, T, in_F, N)(N) -> (B, T, in_F)
        # (B, T, in_F)(in_F,N) -> (B, T, N)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)

        rhs = torch.matmul(self.U3, x)  # (F)(B, N, F, T) -> (B, N, T)

        product = torch.matmul(lhs, rhs)  # (B, T, N)(B, N, T) -> (B, T, T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class Spatial_Attention(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, in_channels, nodes, timesteps):
        super(Spatial_Attention, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, nodes, nodes).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(nodes, nodes).to(DEVICE))

    def forward(self, x):
        '''
        ASTGCN - spatial attention
        :param x: (B, N, in_F, T)
        :return: (B, N, N)
        '''

        # (B, N, F, T)(T) -> (B, N, F)(F, T) -> (B, N, T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(B, N, F, T) -> (B, N, T) -> (B, T, N)

        product = torch.matmul(lhs, rhs)  # (B, N, T)(B, T, N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N, N)(B, N, N) -> (B, N, N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


