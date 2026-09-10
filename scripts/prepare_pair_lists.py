import sys
from pathlib import Path

# Add project root to the Python path to allow importing 'sota'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sota.data import gather_pairs

def create_pair_list(split, output_file, noisy_folder="noisy_gaussian"):
    """
    Generates a text file containing noisy-clean image pairs for a given data split and noise type.
    It relies on the 'oct_splits.json' file being present in the project root.
    """
    splits_file = project_root / 'oct_splits.json'
    if not splits_file.is_file():
        print(f"Error: {splits_file} not found. Please run sota/data_setup.py first.")
        return

    pairs = gather_pairs(str(splits_file), split=split, noisy_folder=noisy_folder)
    
    with open(project_root / output_file, 'w') as f:
        for noisy, clean in pairs:
            f.write(f"{noisy},{clean}\n")
            
    print(f"Created {output_file} with {len(pairs)} pairs (noise={noisy_folder}).")

if __name__ == '__main__':
    noises = [
        "noisy_gaussian",
        "noisy_moderate_gamma",
        "noisy_heavy_gamma",
    ]
    print("Generating training and validation pair lists for multiple noise types...")
    for noise in noises:
        suffix = noise.replace("noisy_", "")
        create_pair_list('train', f'train_pairs_{suffix}.txt', noisy_folder=noise)
        create_pair_list('val', f'val_pairs_{suffix}.txt', noisy_folder=noise)
    print("Done.")
