"""
This script is a consolidated and corrected version of adaptive_oct_denoise.py
to run the experiments as requested by the user.
"""

from __future__ import annotations

import os
import sys
import signal
import glob
import math
import random
import copy
import argparse
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image as _tv_save_image
import torchvision.transforms.functional as TF

# ------------------------------- 
# Device setup
# ------------------------------- 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ------------------------------- 
# Dataset definitions
# ------------------------------- 
class PairedOCTDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]] | str, transform: Optional[Callable] = None):
        if isinstance(pairs, str):
            if not os.path.isfile(pairs): raise FileNotFoundError(f"Pairs file not found: {pairs}")
            loaded = []
            with open(pairs, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    parts = line.strip().split(',')
                    if len(parts) == 2: loaded.append((parts[0], parts[1]))
            self.pairs = loaded
        else: self.pairs = pairs
        self.transform = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        np, cp = self.pairs[idx]
        x = read_image(np).float() / 255.
        y = read_image(cp).float() / 255.
        if self.transform: x,y = self.transform(x), self.transform(y)
        return x, y

def resize_to(shape_hw: Tuple[int, int]) -> Callable:
    h, w = shape_hw
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x.unsqueeze(0), size=shape_hw, mode="bilinear", align_corners=False).squeeze(0)
    return _fn

# ------------------------------- 
# Backbone, Adapter, and Model definitions
# ------------------------------- 
class FiLM(nn.Module):
    def forward(self, x, gamma, beta): return x * gamma + beta if gamma is not None else x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.GELU(), nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.GELU())
        self.film = FiLM()
    def forward(self, x, gamma=None, beta=None): return self.film(self.net(x), gamma, beta)

class DenoisingBackbone(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        c1, c2 = base_channels, base_channels*2
        self.enc1, self.d1 = ConvBlock(in_channels, c1), nn.Conv2d(c1, c1, 3, 2, 1)
        self.enc2, self.d2 = ConvBlock(c1, c2), nn.Conv2d(c2, c2, 3, 2, 1)
        self.bottle = ConvBlock(c2, c2)
        self.u2, self.dec2 = nn.ConvTranspose2d(c2,c2,2,2), ConvBlock(c2+c2, c2)
        self.u1, self.dec1 = nn.ConvTranspose2d(c2,c1,2,2), ConvBlock(c1+c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 1)
    @property
    def modulated_channels(self): return { "enc1": self.enc1.net[0].out_channels, "enc2": self.enc2.net[0].out_channels, "dec2": self.dec2.net[0].out_channels, "dec1": self.dec1.net[0].out_channels }
    def forward(self, x, adapter_features=None):
        def align(ref, src):
            _,_,Hr,Wr=ref.shape; _,_,Hs,Ws=src.shape
            p_h,p_w = max(0,Hr-Hs),max(0,Wr-Ws)
            if p_h>0 or p_w>0: src = F.pad(src,(0,p_w,0,p_h))
            if Hs>Hr or Ws>Wr: src=src[:,:,:Hr,:Wr]
            return src
        def gb(n, sh):
            if not adapter_features: return None,None
            a=adapter_features.get(n,{}); g,b=a.get("gamma"),a.get("beta")
            if g is not None and g.ndim==4 and g.shape[-2:]!=sh: g=F.interpolate(g,sh,"bilinear",False)
            if b is not None and b.ndim==4 and b.shape[-2:]!=sh: b=F.interpolate(b,sh,"bilinear",False)
            return g,b
        g,b=gb("enc1",x.shape[-2:]); x1=self.enc1(x,g,b); d1=self.d1(x1)
        g,b=gb("enc2",d1.shape[-2:]); x2=self.enc2(d1,g,b); d2=self.d2(x2)
        xb=self.bottle(d2); u2=self.u2(xb); u2=align(x2,u2)
        g,b=gb("dec2",u2.shape[-2:]); x3=self.dec2(torch.cat([u2,x2],1),g,b)
        u1=self.up1(x3); u1=align(x1,u1)
        g,b=gb("dec1",u1.shape[-2:]); x4=self.dec1(torch.cat([u1,x1],1),g,b)
        return self.out_conv(x4)

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
        code_h=F.interpolate(code,(h2,w2),"bilinear",False); wc_h=F.interpolate(wc,(h2,w2),"bilinear",False); wi_h=F.interpolate(wi,(h2,w2),"bilinear",False)
        params={}
        for n,h in self.heads.items():
            c,w_c,w_i = (code_h,wc_h,wi_h) if n in ("enc2","dec2") else (code,wc,wi)
            g,b=torch.split(h(c),h.out_channels//2,1)
            params[n]={"gamma":1.+0.3*torch.tanh(g)*w_c, "beta":0.2*torch.tanh(b)*w_i}
        return params, {"coherent_map":wc, "incoherent_map":wi}

class AdaptiveDenoiser(nn.Module):
    def __init__(self, backbone, adapter, residual_mode=False):
        super().__init__(); self.backbone,self.adapter,self.residual_mode = backbone,adapter,residual_mode
    def forward(self, x, return_aux=False):
        features, aux = self.adapter(x) if isinstance(self.adapter(x), tuple) else (self.adapter(x), {})
        raw = self.backbone(x, adapter_features=features)
        if raw.shape[-2:] != x.shape[-2:]: raw=F.interpolate(raw,x.shape[-2:],"bilinear",False)
        out = (x - torch.tanh(raw)).clamp(0.,1.) if self.residual_mode else torch.sigmoid(raw)
        return (out,aux) if return_aux and aux else out

def build_model(base_channels=64, residual_mode=False, adapter_type="casa", backbone_type="unet"):
    backbone = DenoisingBackbone(base_channels=base_channels)
    adapter = CoherenceSpatialAdapter(block_channels=backbone.modulated_channels)
    return AdaptiveDenoiser(backbone, adapter, residual_mode)

# ------------------------------- 
# Losses
# ------------------------------- 
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3): super().__init__(); self.eps=eps
    def forward(self,x,y): return torch.mean(torch.sqrt((x-y)**2+self.eps*self.eps))

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
        sy=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx',sx); self.register_buffer('sy',sy)
    def forward(self,x,y):
        gxp,gyp=F.conv2d(x,self.sx,padding=1),F.conv2d(x,self.sy,padding=1)
        gxt,gyt=F.conv2d(y,self.sx,padding=1),F.conv2d(y,self.sy,padding=1)
        return F.l1_loss(gxp,gxt)+F.l1_loss(gyp,gyt)

def total_variation_loss(img): return F.l1_loss(img[:,:,1:,:],img[:,:,:-1,:])+F.l1_loss(img[:,:,:,1:],img[:,:,:,:-1])

# ------------------------------- 
# Supervised Fine-tuning
# ------------------------------- 
def supervised_finetune(model, paired_loader, val_loader, num_epochs, lr_adapter, lr_backbone, weight_decay, loss_type, lambda_grad, lambda_tv, log_domain, amp, grad_clip_norm, scheduler_type, warmup_epochs, early_stopping_patience, ema_enable, ema_decay, casa_lambda_tv, freeze_backbone=False):
    model.to(device); model.train()
    optim = torch.optim.AdamW([{"params":model.adapter.parameters(),"lr":lr_adapter},{"params":model.backbone.parameters(),"lr":lr_backbone}],weight_decay=weight_decay)
    if freeze_backbone: optim=torch.optim.AdamW(model.adapter.parameters(),lr=lr_adapter,weight_decay=weight_decay)
    scaler=torch.amp.GradScaler('cuda', enabled=amp)
    loss_pix=CharbonnierLoss() if loss_type=='charbonnier' else nn.L1Loss(); loss_grad=GradientLoss()
    ema= {k:v.clone() for k,v in model.state_dict().items()} if ema_enable else None
    
    steps=len(paired_loader)*num_epochs
    sched_main=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=steps-warmup_epochs*len(paired_loader)) if scheduler_type=='cosine' else torch.optim.lr_scheduler.ReduceLROnPlateau(optim,'min',patience=5)
    scheduler=torch.optim.lr_scheduler.SequentialLR(optim,[torch.optim.lr_scheduler.LinearLR(optim,0.01,total_iters=warmup_epochs*len(paired_loader)),sched_main],milestones=[warmup_epochs*len(paired_loader)]) if warmup_epochs>0 else sched_main
    
    best_val, no_improve = float('inf'), 0
    for epoch in range(num_epochs):
        model.train(); losses=[]
        for x,y in paired_loader:
            x,y=x.to(device),y.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                out=model(x,return_aux=True); pred,aux=out if isinstance(out,tuple) else(out,{})
                p,t=(torch.log(pred.clamp(1e-6)),torch.log(y.clamp(1e-6))) if log_domain else (pred,y)
                loss=loss_pix(p,t)
                if lambda_grad>0: loss+=lambda_grad*loss_grad(p,t)
                if lambda_tv>0: loss+=lambda_tv*total_variation_loss(pred)
                if aux and casa_lambda_tv>0: loss+=casa_lambda_tv*sum(total_variation_loss(aux[k]) for k in ['coherent_map','incoherent_map'] if k in aux)
            optim.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            if grad_clip_norm>0: scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip_norm)
            scaler.step(optim); scaler.update()
            if scheduler_type=='cosine' or warmup_epochs>0: scheduler.step()
            losses.append(loss.item())
            if ema:
                with torch.no_grad():
                    for k,v in model.state_dict().items(): ema[k].mul_(ema_decay).add_((1-ema_decay)*v)
        
        val_loss=float('inf')
        if val_loader:
            model.eval(); vlosses=[]
            with torch.no_grad():
                for xv,yv in val_loader:
                    xv,yv=xv.to(device),yv.to(device)
                    with torch.cuda.amp.autocast(enabled=amp):
                        vp=model(xv); p,t=(torch.log(vp.clamp(min=1e-6)),torch.log(yv.clamp(min=1e-6))) if log_domain else(vp,yv)
                        v_loss=loss_pix(p,t)
                    vlosses.append(v_loss.item())
            val_loss=np.mean(vlosses)

        if scheduler_type=='plateau': scheduler.step(val_loss)
        print(f"[Finetune] Epoch {epoch+1}/{num_epochs} Train Loss={np.mean(losses):.4f} Val Loss={val_loss:.4f}")
        if val_loss<best_val: best_val,no_improve=val_loss,0
        else: no_improve+=1
        if no_improve>=early_stopping_patience: print("Early stopping."); break
        
    return ema

def main():
    parser = argparse.ArgumentParser(description="Domain-Adaptive OCT Denoising")
    parser.add_argument("--paired_list", type=str, required=True)
    parser.add_argument("--val_paired_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/finetune_run")
    parser.add_argument("--adapter", type=str, default="casa", choices=["global","spatial","casa"])
    parser.add_argument("--backbone", type=str, default="unet", choices=["unet","nafnet"])
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resize_h", type=int, default=64)
    parser.add_argument("--resize_w", type=int, default=64)
    parser.add_argument("--finetune_lr_adapter", type=float, default=1e-3)
    parser.add_argument("--finetune_lr_backbone", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default='charbonnier', choices=['charbonnier','l1'])
    parser.add_argument("--lambda_grad", type=float, default=0.05)
    parser.add_argument("--lambda_tv", type=float, default=1e-5)
    parser.add_argument("--log_domain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scheduler_type", type=str, default='cosine', choices=['cosine','plateau'])
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--residual_mode", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("=" * 80)
    print("MODEL CONFIGURATION")
    print(f"  Backbone: {args.backbone}, Adapter: {args.adapter}, Base Channels: {args.base_channels}")
    print(f"  Fine-tuning for {args.finetune_epochs} epochs with batch size {args.batch_size}")
    print(f"  Training on {args.resize_h}x{args.resize_w} patches")
    print("=" * 80 + "\n")

    model = build_model(base_channels=args.base_channels, residual_mode=args.residual_mode, adapter_type=args.adapter, backbone_type=args.backbone)
    
    tfm = resize_to((args.resize_h, args.resize_w))
    paired_ds = PairedOCTDataset(args.paired_list, transform=tfm)
    paired_loader = DataLoader(paired_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = None
    if args.val_paired_list:
        val_ds = PairedOCTDataset(args.val_paired_list, transform=tfm)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ema_state = supervised_finetune(
        model, paired_loader, val_loader, args.finetune_epochs, args.finetune_lr_adapter,
        args.finetune_lr_backbone, args.weight_decay, args.loss_type, args.lambda_grad, args.lambda_tv,
        args.log_domain, args.amp, args.grad_clip_norm, args.scheduler_type, args.warmup_epochs,
        args.early_stopping_patience, args.ema, 0.999, 0.0
    )
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, "finetuned.pth"))
    if ema_state: torch.save(ema_state, os.path.join(args.output_dir, "finetuned_ema.pth"))
    print("Done.")

if __name__ == "__main__":
    main()