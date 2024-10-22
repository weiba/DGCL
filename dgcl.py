import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import ChebConv
from torch_geometric.utils import negative_sampling
class MLP(nn.Module):
    def __init__(self, inp_size,  hidden_size, outp_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, outp_size)
        )
        for model in self.net:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def forward(self, x):
        return self.net(x)

class GraphEncoder(nn.Module):
    def __init__(self, gnn,):
        super().__init__()
        self.gnn = gnn

    def forward(self, adj, in_feats):
        representations = self.gnn(in_feats, adj)
        representations = representations.view(-1, representations.size(-1))
        return representations

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

def contrastive_loss(h1, h2,pos,tau):
        sim_matrix = sim(h1, h2)
        f = lambda x: torch.exp(x/tau)
        matrix_t = f(sim_matrix)
        numerator = matrix_t.mul(pos).sum(dim=-1)
        denominator= torch.sum(matrix_t, dim=-1)
        loss  = -torch.log(numerator/denominator).mean()
        return loss

class DGCL(nn.Module):
    def __init__(self, 
                 gnn,
                 pos1,
                 pos2,
                 tau,
                 edge_index1,
                 edge_index2,
                 gnn_outsize,
                 projection_size,
                 projection_hidden_size,
                ):
        super().__init__()
        self.encoder = GraphEncoder(gnn)
        self.projector = MLP(gnn_outsize, projection_hidden_size, projection_size)
        self.pos1 = pos1
        self.pos2 = pos2
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.tau = tau
        self.conv1 = ChebConv(gnn_outsize, 1, K=2, normalization="sym")
        self.conv2 = ChebConv(gnn_outsize, 1, K=2, normalization="sym")
        
    def forward(self, aug_adj_1, aug_adj_2, aug_feat_1, aug_feat_2):

        encoder_one = self.encoder(aug_adj_1,aug_feat_1)
        encoder_two = self.encoder(aug_adj_2,aug_feat_2)

        proj_one = self.projector(encoder_one)
        proj_two = self.projector(encoder_two)

        l1 = contrastive_loss(proj_one, proj_two, self.pos1, self.tau)
        l2 = contrastive_loss(proj_two, proj_one, self.pos1, self.tau)
        Conloss = l1+l2
        #Learning network-specific gene feature
        emb1 = self.conv1(encoder_one,aug_adj_1)
        emb2 = self.conv2(encoder_two,aug_adj_2)

        # Logistic Regression Module input feature
        emb = torch.cat((emb1,emb2),dim=1)

        pos_loss = -torch.log(torch.sigmoid((emb1[self.edge_index1[0]] * emb1[self.edge_index1[1]]).sum(dim=1)) + 1e-15).mean()
        neg_edge_index = negative_sampling(self.edge_index2, emb1.shape[0], 504378)
        neg_loss = -torch.log(
            1 - torch.sigmoid((emb1[neg_edge_index[0]] * emb1[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        r_loss = (pos_loss + neg_loss)/2
        return emb1,emb2,emb,Conloss,r_loss