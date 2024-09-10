import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from vggsfm.runners.relocalization_runner import RelocalizationRunner
from vggsfm.datasets.relocalization_demo_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines
from pdb import set_trace

import os
import loguru
@hydra.main(config_path="cfgs/", config_name="relocalization_demo")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VGGSfMRunner is the main controller.
    """

    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    # Configure CUDA settings
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)

    # Initialize VGGSfM Runner
    vggsfm_runner = RelocalizationRunner(cfg)
    
    recursive = cfg.recursive
    scenes = []
    # if recursicve working mode, run the demo for all scenes
    if recursive: 
        root = cfg.SCENE_DIR
        # dataset format:
        '''
            root:
                scene1:
                    images
                    masks
                    poses
                scene2:
                    scene2_1:
                        images
                        masks
                        poses
                    scene2_2:
                        ...
        '''
        # recursively get all scenes (DFS)
        def dfs(path):
            # if include images (at least) dir , that is a scene
            if 'images' in os.listdir(path):
                scenes.append(path)
                return
            for sub in os.listdir(path):
                dfs(os.path.join(path, sub))
        dfs(root)
        
        loguru.logger.info(f"Found {len(scenes)} scenes")
    else:
        scenes.append(cfg.SCENE_DIR)
        loguru.logger.info(f"Found 1 scene")
        
    
    for scene in scenes:
        # Load Data
        test_dataset = DemoLoader(
            SCENE_DIR=scene,
            img_size=cfg.img_size,
            normalize_cameras=False,
            relocalization_method=cfg.relocalization_method,
            use_mask=cfg.use_mask,
        )

        sequence_list = test_dataset.sequence_list

        seq_name = sequence_list[0]

        # Load the data for the selected sequence
        batch, image_paths = test_dataset.get_data(
            sequence_name=seq_name, return_path=True
        )

        output_dir = batch[
            "scene_dir"
        ]  # which is also cfg.SCENE_DIR for DemoLoader

        images = batch["image"]
        gt_poses = batch["gt_poses"]
        masks = batch["masks"] if batch["masks"] is not None else None
        crop_params = (
            batch["crop_params"] if batch["crop_params"] is not None else None
        )

        # Cache the original images for visualization, so that we don't need to re-load many times
        original_images = batch["original_images"]

        # Run VGGSfM
        # Both visualization and output writing are performed inside VGGSfMRunner
        predictions = vggsfm_runner.run(
            images,
            masks=masks,
            original_images=original_images,
            image_paths=image_paths,
            crop_params=crop_params,
            seq_name=seq_name,
            output_dir=output_dir,
            relocalization_method=cfg.relocalization_method,
            gt_poses=gt_poses,
        )

        loguru.logger.info(f"Scene {scene} done")

    return True


if __name__ == "__main__":
    with torch.no_grad():
        demo_fn()
