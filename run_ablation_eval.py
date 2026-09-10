import torch, json
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative, PKU37Dataset, compute_psnr
from validate_crossdataset import otsu_tissue_mask
import torch.nn.functional as F

def compute_cnr(img):
    mask = otsu_tissue_mask(img)
    bg = 1.0 - mask
    sig_mean = (img * mask).sum() / mask.sum().clamp(min=1)
    bg_mean = (img * bg).sum() / bg.sum().clamp(min=1)
    bg_std = torch.sqrt(((img - bg_mean)**2 * bg).sum() / bg.sum().clamp(min=1) + 1e-8)
    return ((sig_mean - bg_mean) / bg_std.clamp(min=1e-4)).item()

def compute_tci(img, clean):
    dy = F.conv2d(img, torch.tensor([[-1],[1]],dtype=img.dtype,device=img.device).view(1,1,2,1), padding=(1,0))
    dy_c = F.conv2d(clean, torch.tensor([[-1],[1]],dtype=clean.dtype,device=clean.device).view(1,1,2,1), padding=(1,0))
    return (dy.abs().mean() / dy_c.abs().mean().clamp(min=1e-8)).item()

def compute_epi(img, clean):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=img.dtype,device=img.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=img.dtype,device=img.device).view(1,1,3,3)
    ex = F.conv2d(F.pad(img,(1,1,1,1),'reflect'), kx)
    ey = F.conv2d(F.pad(img,(1,1,1,1),'reflect'), ky)
    cx = F.conv2d(F.pad(clean,(1,1,1,1),'reflect'), kx)
    cy = F.conv2d(F.pad(clean,(1,1,1,1),'reflect'), ky)
    e_mag = torch.sqrt(ex**2 + ey**2 + 1e-8)
    c_mag = torch.sqrt(cx**2 + cy**2 + 1e-8)
    return (e_mag.mean() / c_mag.mean().clamp(min=1e-8)).item()

def compute_bs(img, clean):
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=img.dtype,device=img.device).view(1,1,3,3)
    ey = F.conv2d(F.pad(img,(1,1,1,1),'reflect'), ky)
    cy = F.conv2d(F.pad(clean,(1,1,1,1),'reflect'), ky)
    return (ey.abs().mean() / cy.abs().mean().clamp(min=1e-8)).item()

def compute_enl(img):
    mask = otsu_tissue_mask(img)
    bg = 1.0 - mask
    bg_pixels = img[bg > 0.5]
    if bg_pixels.numel() < 10: return 0.0
    return (bg_pixels.mean()**2 / bg_pixels.var().clamp(min=1e-8)).item()

ckpt_path = 'outputs/ablation_no_uncertainty/best_model_cooperative.pth'
model = NeuroSymbolicDenoiserV8Cooperative(backbone_name='nafnet', pretrained_backbone='outputs/nafnet_pku37_w40/best_model.pth', hidden_channels=64)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt.get('model_state_dict', ckpt)
cleaned = {k.replace('._orig_mod.', '.').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model_state = model.state_dict()
compatible = {k: v for k, v in cleaned.items() if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict(compatible, strict=False)
model.corrector.set_ablation('no_uncertainty')
model.eval()

dataset = PKU37Dataset('pku37_oct_dataset/pku37_real_test.jsonl', patch_size=0, is_train=False)
print(f'Evaluating {len(dataset)} images...')

psnr_deltas, cnr_deltas, tci_deltas, epi_deltas, bs_deltas, enl_deltas = [], [], [], [], [], []
for i in range(len(dataset)):
    s = dataset[i]
    clean = s['clean'].unsqueeze(0)
    noisy = s['noisy'].unsqueeze(0)
    with torch.no_grad():
        bb_out, unc = model.backbone(noisy)
        corr, _ = model.corrector(bb_out, noisy, None, nafnet_uncertainty=unc, return_details=False)
    psnr_bb = compute_psnr(bb_out, clean)
    psnr_co = compute_psnr(corr, clean)
    cnr_bb = compute_cnr(bb_out)
    cnr_co = compute_cnr(corr)
    tci_bb = compute_tci(bb_out, clean)
    tci_co = compute_tci(corr, clean)
    epi_bb = compute_epi(bb_out, clean)
    epi_co = compute_epi(corr, clean)
    bs_bb = compute_bs(bb_out, clean)
    bs_co = compute_bs(corr, clean)
    enl_bb = compute_enl(bb_out)
    enl_co = compute_enl(corr)
    psnr_deltas.append(psnr_co - psnr_bb)
    cnr_deltas.append((cnr_co - cnr_bb) / max(abs(cnr_bb), 1e-8) * 100)
    tci_deltas.append((tci_co - tci_bb) / max(abs(tci_bb), 1e-8) * 100)
    epi_deltas.append((epi_co - epi_bb) / max(abs(epi_bb), 1e-8) * 100)
    bs_deltas.append((bs_co - bs_bb) / max(abs(bs_bb), 1e-8) * 100)
    enl_deltas.append((enl_co - enl_bb) / max(abs(enl_bb), 1e-8) * 100)
    if (i+1) % 20 == 0:
        print(f'  {i+1}/{len(dataset)}')

import numpy as np
results = {
    'ablation': 'no_uncertainty',
    'psnr_delta': float(np.mean(psnr_deltas)),
    'cnr_delta_pct': float(np.mean(cnr_deltas)),
    'tci_delta_pct': float(np.mean(tci_deltas)),
    'epi_delta_pct': float(np.mean(epi_deltas)),
    'bs_delta_pct': float(np.mean(bs_deltas)),
    'enl_delta_pct': float(np.mean(enl_deltas)),
}
print(json.dumps(results, indent=2))
with open('outputs/ablation_no_uncertainty/ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
