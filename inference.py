import argparse
import torch
from tqdm.auto import tqdm
from pathlib import Path

from utils.misc import *
from utils.denoise import *
from models.denoise import *
from datasets.pcl import load_pcd
from utils.logger import Logger


def input_iter(path):
    fn = Path(path).name
    pcl_noisy = torch.FloatTensor(load_pcd(path))
    pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
    return {
        'pcl_noisy': pcl_noisy,
        'name': fn[:-4],
        'center': center,
        'scale': scale
    }


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description="Inference script to denoise pointclouds")
    parser.add_argument('--input_file', type=str, required=True, help="Noisy input pointcloud (.ply/.npy/.txt/.xyz)")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory. Pointclouds are saved on xyz format.")
    parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2020)
    # Denoiser parameters
    parser.add_argument('--cluster_size', type=int, default=30000)
    args = parser.parse_args()
    print("Args: ", args)
    seed_all(args.seed)

    # Logging
    logger = Logger("score-denoise")

    # Input/Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Model
    logger.info('Loading DenoiseNet model and weights...')
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = DenoiseNet(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Denoise
    logger.info(f'Performing denoising on {args.input_file}...')
    data = input_iter(args.input_file)
    pcl_noisy = data['pcl_noisy'].to(args.device)
    pcl_is_large = False
    if len(pcl_noisy)>50000:
        logger.info(f'PCD is considered large (>50k points)!')
        pcl_is_large = True
    try:
        with torch.no_grad():
            if pcl_is_large:
                pcl_denoised = denoise_large_pointcloud(
                    model=model,
                    pcl=pcl_noisy,
                    cluster_size=args.cluster_size,
                    seed=args.seed
                )
                pcl_denoised = pcl_denoised.cpu().numpy()
            else:
                scale, center = data['scale'], data['center']
                pcl_denoised = patch_based_denoise(model, pcl_noisy).cpu()
                pcl_denoised = pcl_denoised * scale + center

        # Save result
        save_path = output_dir / Path(args.input_file).with_suffix(".xyz").name
        logger.info(f"Saving pointcloud at {save_path}")
        np.savetxt(save_path, pcl_denoised, fmt='%.8f')

    except Exception as e:
        logger.error(e)

    finally:
        logger.warning(f"Finished!")

