
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

class CoherenceSpatialAdapter(nn.Module):
    def __init__(self, block_channels, scales=(1,3,5,7)):
        super().__init__()
        self.ch, self.scales = block_channels, scales
        self.coh_est = nn.ModuleList([nn.Conv2d(1,16,k,padding=k//2) for k in self.scales])
        self.corr_pred = nn.Sequential(nn.Conv2d(16*len(self.scales),64,3,1,1), nn.GELU(), nn.Conv2d(64,64,3,1,1), nn.GELU())
        self.heads=nn.ModuleDict({n:nn.Conv2d(64,2*c,1) for n,c in self.ch.items()})
        self.decomp=nn.Sequential(nn.Conv2d(64,32,1),nn.GELU(),nn.Conv2d(32,2,1),nn.Softmax(dim=1))
    def forward(self,x):
        feats = torch.cat([est(x) for est in self.coh_est],1)
        code=self.corr_pred(feats); decomp=self.decomp(code); wc,wi=decomp[:,0:1],decomp[:,1:2]
        H,W=x.shape[-2:]; h2,w2=(H+1)//2,(W+1)//2
        code_h=F.interpolate(code, size=(h2,w2), mode="bilinear",align_corners=False); wc_h=F.interpolate(wc, size=(h2,w2), mode="bilinear",align_corners=False); wi_h=F.interpolate(wi, size=(h2,w2), mode="bilinear",align_corners=False)
        params={}
        for n,h in self.heads.items():
            c,w_c,w_i = (code_h,wc_h,wi_h) if n in ("enc2","dec2") else (code,wc,wi)
            g,b=torch.split(h(c),h.out_channels//2,1)
            params[n]={"gamma":1.+0.3*torch.tanh(g)*w_c, "beta":0.2*torch.tanh(b)*w_i}
        return params, {"coherent_map":wc, "incoherent_map":wi}

if __name__ == '__main__':
    block_channels = {"enc1": 64, "enc2": 128, "dec1": 64, "dec2": 128}
    adapter = CoherenceSpatialAdapter(block_channels=block_channels)
    dummy_tensor = torch.randn(1, 1, 64, 64)
    params, aux = adapter(dummy_tensor)
    print("Test successful!")
    print("Params keys:", params.keys())
    print("Aux keys:", aux.keys())
