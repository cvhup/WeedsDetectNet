class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])
        self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
    
    def forward(self, x):
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)


