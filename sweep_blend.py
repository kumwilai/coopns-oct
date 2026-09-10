import torch
from torch.utils.data import DataLoader
from adaptive_oct_denoise import build_model, PairedOCTDataset, resize_to, compute_psnr, compute_ssim

def eval_alpha(model, dl, alpha, device):
    psnrs, ssims = [], []
    with torch.no_grad():
        for i, (x_noisy, x_clean) in enumerate(dl, 1):
            x_noisy, x_clean = x_noisy.to(device), x_clean.to(device)
            base = model(x_noisy)
            blend = (1 - alpha) * base + alpha * x_noisy
            psnrs.append(compute_psnr(blend, x_clean))
            ssims.append(compute_ssim(blend, x_clean))
            if i % 10 == 0 or i == len(dl):
                print(f"    [{i}/{len(dl)}] alpha={alpha:.2f} running PSNR={sum(psnrs)/len(psnrs):.2f} dB", flush=True)
    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = "checkpoints/universal_casa/finetuned.pth"
    model = build_model(adapter_type="casa").to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    val_files = {
        'gaussian': 'val_pairs_gaussian.txt',
        'moderate_gamma': 'val_pairs_moderate_gamma.txt',
        'heavy_gamma': 'val_pairs_heavy_gamma.txt',
        'poisson': 'val_pairs_poisson.txt',
        'rayleigh': 'val_pairs_rayleigh.txt',
    }
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    resize = resize_to((64, 64))
    results = {}
    for noise, vf in val_files.items():
        ds = PairedOCTDataset(vf, transform=resize)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        best = None
        for a in alphas:
            ps, ss = eval_alpha(model, dl, a, device)
            if best is None or ps > best[1]:
                best = (a, ps, ss)
            print(f"{noise}: alpha={a:.2f} -> PSNR={ps:.2f} dB, SSIM={ss:.4f}", flush=True)
        results[noise] = best
        print(f"{noise}: best alpha={best[0]:.2f}, PSNR={best[1]:.2f} dB, SSIM={best[2]:.4f}")

    avg_psnr = sum(v[1] for v in results.values())/len(results)
    avg_ssim = sum(v[2] for v in results.values())/len(results)
    print(f"Average PSNR over noises: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
