import numpy as np
import torch
from torch import nn
from torchsummary import summary

class FCLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.1):
        super(FCLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = self.dropout(self.activation(self.bn(self.fc1(x))))
        return x


class FCLayers(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=256, dropout_rate=0.1):
        super(FCLayers, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.dropout(self.activation(self.bn(self.fc1(x))))
        x = self.fc2(x)
        return x

class SpecificNetLayer(nn.Module):
    def __init__(self, special_dim=27, other_dim1=80, other_dim2=80, output_size=64, hidden_dim=256, dropout_rate=0.1):
        super(SpecificNetLayer, self).__init__()
        self.mlp1 = FCLayers(input_size=special_dim, output_size=output_size, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.mlp2 = FCLayers(input_size=output_size+other_dim1+other_dim2, output_size=output_size, hidden_dim=hidden_dim,
                        dropout_rate=dropout_rate)
        self.proj1 = FCLayer(input_size=other_dim1, output_size=other_dim1, dropout_rate=dropout_rate)
        self.proj2 = FCLayer(input_size=other_dim2, output_size=other_dim2, dropout_rate=dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sx, ox1, ox2):
        sx = self.mlp1(sx)
        ox1 = self.proj1(ox1)
        ox2 = self.proj2(ox2)
        weight = self.sigmoid(self.mlp2(torch.cat((sx, ox1, ox2), dim=1)))*2
        return sx*weight

class SpecificNet(nn.Module):
    def __init__(self, clinical_dim=27, gene_dim=80, wsi_dim=256, output_size=64, hidden_dim=256, dropout_rate=0.1):
        super(SpecificNet, self).__init__()
        self.specialNet1 = SpecificNetLayer(special_dim=clinical_dim, other_dim1=gene_dim, other_dim2=wsi_dim,
                                  output_size=output_size, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.specialNet2 = SpecificNetLayer(special_dim=gene_dim, other_dim1=clinical_dim, other_dim2=wsi_dim,
                                  output_size=output_size, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.specialNet3 = SpecificNetLayer(special_dim=wsi_dim, other_dim1=clinical_dim, other_dim2=gene_dim,
                                  output_size=output_size, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

    def forward(self, x1, x2, x3):
        sx1= self.specialNet1(x1, x2, x3)
        sx2 = self.specialNet2(x2, x1, x3)
        sx3 = self.specialNet3(x3, x1, x2)
        return sx1, sx2, sx3


class CrossAttention(nn.Module):
    def __init__(self, dim1=27, dim2=80, output_size=64, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        self.dk = dim1
        self.fc1 = FCLayer(input_size=dim1, output_size=output_size, dropout_rate=dropout_rate)
        self.fc2 = FCLayer(input_size=dim2, output_size=output_size, dropout_rate=dropout_rate)
        self.fc3 = FCLayer(input_size=dim2, output_size=output_size, dropout_rate=dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        Q = self.fc1(x1)
        K = self.fc2(x2)
        V = self.fc3(x2)
        attn = self.softmax(torch.mm(Q.T, K) / np.sqrt(self.dk))
        return torch.mm(V, attn.T)

class ShareNet(nn.Module):
    def __init__(self, dim1=128, dim2=128, dim3=128,output_size=64, dropout_rate=0.1):
        super(ShareNet, self).__init__()
        self.CA1 = CrossAttention(dim1=dim1, dim2=dim2, output_size=output_size, dropout_rate=dropout_rate)
        self.CA2 = CrossAttention(dim1=dim2, dim2=dim3, output_size=output_size, dropout_rate=dropout_rate)
        self.CA3 = CrossAttention(dim1=dim3, dim2=dim1, output_size=output_size, dropout_rate=dropout_rate)

    def forward(self, cx, gx, wx):
        share_cg = self.CA1(cx, gx)
        share_gw = self.CA2(gx, wx)
        share_wc = self.CA3(wx, cx)
        return share_cg, share_gw, share_wc


class GateLayer(nn.Module):
    def __init__(self, dim=27, output_size=1):
        super(GateLayer, self).__init__()
        self.fc = nn.Linear(dim, output_size)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(output_size)
    def forward(self, x):
        weight = self.sigmoid(self.bn(self.fc(x)))*2
        return x + x*weight


class GNN(nn.Module):
    def __init__(self,dim=64):
        super(GNN, self).__init__()
        self.K_gate1 = GateLayer(dim=dim, output_size=1)
        self.K_gate2 = GateLayer(dim=dim, output_size=1)
        self.K_gate3 = GateLayer(dim=dim, output_size=1)
        self.K_gate4 = GateLayer(dim=dim, output_size=1)
        self.K_gate5 = GateLayer(dim=dim, output_size=1)
        self.K_gate6 = GateLayer(dim=dim, output_size=1)
        self.K_gate7 = GateLayer(dim=dim, output_size=1)


    def forward(self, x1, x2, x3, x4,x5,x6,x7):
        x1 = self.K_gate1(x1)
        x2 = self.K_gate2(x2)
        x3 = self.K_gate3(x3)
        x4 = self.K_gate4(x4)
        x5 = self.K_gate5(x5)
        x6 = self.K_gate6(x6)
        x7 = self.K_gate7(x7)
        emb = torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=-1)
        return emb


class HKD(nn.Module):
    def __init__(self, clinical_dim=27, gene_dim=80, wsi_dim=64, output_size=64, hidden_dim=256, dropout_rate=0.1):
        super(HKD, self).__init__()
        self.specificNet1 = SpecificNet(clinical_dim=clinical_dim, gene_dim=gene_dim, wsi_dim=wsi_dim,
                                       hidden_dim=hidden_dim,
                                       output_size=output_size, dropout_rate=dropout_rate)
        self.specificNet2 = SpecificNet(clinical_dim=output_size, gene_dim=output_size, wsi_dim=output_size,
                                        hidden_dim=hidden_dim,
                                        output_size=output_size, dropout_rate=dropout_rate)
        self.shareNet1 = ShareNet(dim1=clinical_dim, dim2=gene_dim, dim3=wsi_dim,output_size=output_size, dropout_rate=dropout_rate)
        self.shareNet2 = ShareNet(dim1=output_size, dim2=output_size, dim3=output_size,output_size=output_size, dropout_rate=dropout_rate)
        self.fc1 = FCLayer(output_size*3, output_size, dropout_rate)
        self.Gate = GNN(dim=output_size)
        self.proj = FCLayers(output_size*7, 1, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.proj2 = FCLayers(output_size*7, 1, hidden_dim=hidden_dim, dropout_rate=dropout_rate)


    def forward(self, cx, gx, wx):
        cgshx,gwshx,wcshx = self.shareNet1(cx, gx, wx)
        cspx, gspx, wspx = self.specificNet1(cx, gx, wx)
        # Divide and conquer, recalculate share and specific
        triple_share1,triple_share2,triple_share3 = self.shareNet2(cgshx, gwshx, wcshx)
        triple_share = self.fc1(torch.cat([triple_share1,triple_share2,triple_share3], dim=-1))
        dualx1, dualx2, dualx3 = self.specificNet2(cgshx, gwshx, wcshx)
        emb = self.Gate(cspx, gspx, wspx, dualx1, dualx2, dualx3, triple_share)
        log_h = self.proj(emb)
        log_st = self.proj2(emb)
        return cspx,gspx,wspx,cgshx,gwshx,wcshx,dualx1,dualx2,dualx3,triple_share1,triple_share2,triple_share3,triple_share,log_h,log_st


if __name__ == '__main__':
    test_model = HKD(27,80,64,256, 256)
    cspx,gspx,wspx,cgshx,gwshx,wcshx,dualx1,dualx2,dualx3,triple_share1,triple_share2,triple_share3,triple_share,log_h,log_st = test_model(torch.randn(4, 27), torch.randn(4, 80),torch.randn(4, 64))
    print(cspx.shape)
    print(gspx.shape)
    print(wspx.shape)
    print(cgshx.shape)
    print(gwshx.shape)
    print(wcshx.shape)
    print(dualx1.shape)
    print(dualx2.shape)
    print(dualx3.shape)
    print(triple_share1.shape)
    print(triple_share2.shape)
    print(triple_share3.shape)
    print(triple_share.shape)
    print(log_h.shape)
    print(log_st.shape)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    summary(test_model.to(device), input_size=[(27,),(80,),(64,)], batch_size=4)
