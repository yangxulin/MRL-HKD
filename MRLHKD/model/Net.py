import torch
from torch import nn
from torchsummary import summary

from MRLHKD.model.MultimodalPatchAttentionNetwork import PatchAttentionNet
from MRLHKD.model.HierarchicalKnowledgeDecomposition import HKD


class Net(nn.Module):
    def __init__(self, clinical_dim, gene_dim, patch_num, wsi_dim, output_size=64, hidden_dim=256, dropout_rate=0.1):
        super(Net, self).__init__()
        self.patchEmb = PatchAttentionNet(clinical_dim,gene_dim,patch_num,wsi_dim,hidden_dim, dropout_rate=dropout_rate)
        self.MMK = HKD(clinical_dim,gene_dim,wsi_dim//4,output_size=output_size, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    def forward(self, cx,gx,wx):
        wx = self.patchEmb(cx, gx, wx)
        cspx,gspx,wspx,cgshx,gwshx,wcshx,dualx1,dualx2,dualx3,triple_share1,triple_share2,triple_share3,triple_share,log_h,log_st = self.MMK(cx, gx, wx)
        return cspx,gspx,wspx,cgshx,gwshx,wcshx,dualx1,dualx2,dualx3,triple_share1,triple_share2,triple_share3,triple_share,log_h,log_st


if __name__ == '__main__':
    test_model = Net(27,80,40,256,64,128)
    cspx,gspx,wspx,cgshx,gwshx,wcshx,dualx1,dualx2,dualx3,triple_share1,triple_share2,triple_share3,triple_share,log_h,log_st = test_model(torch.randn(128, 27), torch.randn(128, 80),torch.randn(128, 40, 256))
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
    summary(test_model.to(device), input_size=[(27,), (80,), (40, 256)], batch_size=128)
