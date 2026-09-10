import argparse
import torch
from torch.utils.data import DataLoader
from adaptive_oct_denoise import (
    build_model, test_time_adaptation, SpectralNoiseCharacterizer,
    PairedOCTDataset, resize_to, compute_psnr, compute_ssim
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--val_pairs', type=str, required=True)
    parser.add_argument('--adapter', type=str, default='casa', choices=['global','spatial','casa'])
    parser.add_argument('--tta_steps', type=int, default=30)
    parser.add_argument('--tta_lr', type=float, default=5e-4)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--tv_weight', type=float, default=1e-4)
    parser.add_argument('--self_consistency', type=float, default=0.2)
    parser.add_argument('--anchor_weight', type=float, default=0.08)
    parser.add_argument('--spectral_weight', type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(adapter_type=args.adapter).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Adapt only the adapter
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.adapter.parameters():
        p.requires_grad = True

    ds = PairedOCTDataset(args.val_pairs, transform=resize_to((args.image_size, args.image_size)))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    spec = SpectralNoiseCharacterizer().to(device)

    print(f"Evaluating on {len(ds)} validation pairs...")
    print(f"TTA settings: steps={args.tta_steps}, lr={args.tta_lr}")
    print("Adapt adapter only: True")
    print("----------------------------------------------------------------------")
    print("Progress will be shown every image (PSNR/SSIM before vs after TTA)...")

    gains = []
    gains_ssim = []
    for i, (x_noisy, x_clean) in enumerate(dl, 1):
        x_noisy, x_clean = x_noisy.to(device), x_clean.to(device)
        with torch.no_grad():
            base = model(x_noisy)
        tta = test_time_adaptation(
            model, x_noisy,
            num_steps=args.tta_steps, lr=args.tta_lr,
            tv_weight=args.tv_weight,
            self_consistency=args.self_consistency,
            anchor_weight=args.anchor_weight,
            spectral_weight=args.spectral_weight,
            noise_characterizer=spec,
        ).to(device)
        ps_base = compute_psnr(base, x_clean)
        ps_tta = compute_psnr(tta, x_clean)
        ss_base = compute_ssim(base, x_clean)
        ss_tta = compute_ssim(tta, x_clean)
        gains.append(ps_tta - ps_base)
        gains_ssim.append(ss_tta - ss_base)
        print(f"[{i:4d}/{len(dl)}] ({100.0*i/len(dl):5.1f}%) | "
              f"PSNR base {ps_base:5.2f} dB vs TTA {ps_tta:5.2f} dB (gain {ps_tta-ps_base:+.2f}), "
              f"SSIM base {ss_base:0.4f} vs TTA {ss_tta:0.4f} (gain {ss_tta-ss_base:+.4f})",
              flush=True)

    mean_gain = sum(gains)/len(gains)
    mean_gain_ssim = sum(gains_ssim)/len(gains_ssim)
    print(f"Mean PSNR gain over {len(gains)} images: {mean_gain:+.2f} dB")
    print(f"Mean SSIM gain over {len(gains_ssim)} images: {mean_gain_ssim:+.4f}")

if __name__ == "__main__":
    main()
