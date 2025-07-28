import torch
from torch import nn
from torchsummary import summary
from MRLHKD.model.HierarchicalKnowledgeDecomposition import FCLayers
# patch-level attention
class PatchAttentionNet(nn.Module):
    def __init__(self, clinical_dim, gene_dim, patch_num, wsi_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.fc_c = nn.Linear(clinical_dim, clinical_dim)
        self.fc_g = nn.Linear(gene_dim, gene_dim)
        self.fc_w = nn.Linear(wsi_dim, wsi_dim//4)
        self.patch_num = patch_num
        self.fcs = FCLayers(clinical_dim+gene_dim+wsi_dim//4, 1, hidden_dim, dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cx, gx, wx):
        cx = self.fc_c(cx)
        gx = self.fc_g(gx)
        wx = self.fc_w(wx)
        ans = []
        for i in range(self.patch_num):
            wx_i = wx[:,i,:]
            ans.append(self.fcs(torch.cat((cx, gx, wx_i), dim=1)))
        attn = self.softmax(torch.stack(ans, dim=1))
        # sum polling
        return torch.sum(attn*wx, dim=1)


if __name__ == '__main__':
    test_model = PatchAttentionNet(27,80,40,256, 256, 0.1)
    a = test_model(torch.randn(128, 27), torch.randn(128, 80),torch.randn(128, 40, 256))
    print(a.shape)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    summary(test_model.to(device), input_size=[(27,), (80,), (40, 256)], batch_size=128)
