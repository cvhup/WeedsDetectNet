import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class GA(nn.Module):
    default_act = nn.SiLU()  # Default activation function

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0392, factor=1.2):
        super(GA, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # Convert threshold and factor to nn.Parameter for optimization during training
        self.threshold = nn.Parameter(torch.Tensor([threshold]))
        self.factor = nn.Parameter(torch.Tensor([factor]))

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))


class G1012(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0392, factor=1.2):
        super(G1012, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))
    



class G1013(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0392, factor=1.3):
        super(G1013, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1315(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0510, factor=1.5):
        super(G1315, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1415(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0549, factor=1.5):
        super(G1415, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1416(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0549, factor=1.6):
        super(G1416, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1513(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0588, factor=1.3):
        super(G1513, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1515(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0588, factor=1.5):
        super(G1515, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1615(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0627, factor=1.5):
        super(G1615, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1715(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0667, factor=1.5):
        super(G1715, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1815(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0706, factor=1.5):
        super(G1815, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G1915(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0745, factor=1.5):
        super(G1915, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G2015(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0784, factor=1.5):
        super(G2015, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G2115(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0824, factor=1.5):
        super(G2115, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G17515(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.06863, factor=1.5):
        super(G17515, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G18515(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.07255, factor=1.5):
        super(G18515, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G19515(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.07647, factor=1.5):
        super(G19515, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G2210(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0863, factor=1):
        super(G2210, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G2211(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0863, factor=1.1):
        super(G2211, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G2212(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0863, factor=1.2):
        super(G2212, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G2213(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0863, factor=1.3):
        super(G2213, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G2214(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0863, factor=1.4):
        super(G2214, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))

class G2215(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0863, factor=1.5):
        super(G2215, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.threshold = threshold
        self.factor = factor

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))


class GA(nn.Module):
    default_act = nn.SiLU()  # Default activation function

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0392, factor=1.2):
        super(GA, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # Convert threshold and factor to nn.Parameter for optimization during training
        self.threshold = nn.Parameter(torch.Tensor([threshold]))
        self.factor = nn.Parameter(torch.Tensor([factor]))

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))


class GA2116(nn.Module):
    default_act = nn.SiLU()  # Default activation function

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, threshold=0.0824, factor=1.6):
        super(GA2116, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # Convert threshold and factor to nn.Parameter for optimization during training
        self.threshold = nn.Parameter(torch.Tensor([threshold]))
        self.factor = nn.Parameter(torch.Tensor([factor]))

    def forward(self, x):
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        diff = g - r
        mask = (diff > self.threshold).half()
        adjusted_mask = mask * self.factor
        adjusted_mask[adjusted_mask == 0] = 1
        adjusted_r = r * adjusted_mask
        adjusted_g = g * adjusted_mask
        adjusted_b = b * adjusted_mask
        y = torch.cat([adjusted_r.unsqueeze(1), adjusted_g.unsqueeze(1), adjusted_b.unsqueeze(1)], dim=1)
        return self.act(self.bn(self.conv(y)))





























