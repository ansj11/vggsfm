import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from vggsfm.runners.relocalization_runner import RelocalizationRunner
from vggsfm.datasets.onepose_lm_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines
from vggsfm.utils.metric import calculate_auc_np, calculate_auc_single_np

import os
import loguru
import sys
@hydra.main(config_path="cfgs/", config_name="pose_demo")
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
                    train:
                        images
                        masks
                        poses
                    test:
                        ...
                scene2:
                    ...
        '''
        # recursively get all scenes (DFS)
        def dfs(path):
            # if include images (at least) dir , that is a scene
            
            # check if the path is dir 
            if not os.path.isdir(path):
                return
                
            if 'train' in os.listdir(path) and 'test' in os.listdir(path):
                scenes.append(path)
                return
            for sub in os.listdir(path):
                dfs(os.path.join(path, sub))
                
        dfs(root)
        
        loguru.logger.info(f"Found {len(scenes)} scenes")
    else:
        scenes.append(cfg.SCENE_DIR)
        loguru.logger.info(f"Found 1 scene")
        
    scenes.sort()
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
        
        cat = seq_name.split('_')[-1]
        if cat in ['ape']:
            loguru.logger.info(f"Skip {scene}")
            continue
        
        scene_metrics = {}
        
        output_dir = None
        
        import tqdm
        for query_id in tqdm.tqdm(range(0, len(test_dataset)), desc=f"Processing {scene}"):
            try:
                # Load the data for the selected sequence
                batch, image_paths = test_dataset.get_data(
                    sequence_name=seq_name, return_path=True, ref_size=cfg.ref_num, query_index=query_id
                )

                output_dir = batch[
                    "scene_dir"
                ]  # which is also cfg.SCENE_DIR for DemoLoader

                images = batch["image"]
                # log query image path for debug:
                # loguru.logger.info(f"Query image path: {image_paths[-1]}")
                gt_poses = batch["gt_poses"]
                # log gt poses for debug:
                # loguru.logger.info(f"GT poses: {gt_poses}")
                masks = batch["masks"] if batch["masks"] is not None else None
                # log masks shape for debug:
                # loguru.logger.info(f"Masks shape: {masks.shape}")
                crop_params = (
                    batch["crop_params"] if batch["crop_params"] is not None else None
                )

                # Cache the original images for visualization, so that we don't need to re-load many times
                original_images = batch["original_images"]

                # Run VGGSfM
                # Both visualization and output writing are performed inside VGGSfMRunner
                
                # supress the output of the runner
                # with SuppressOutput():                    
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
                    model_path = os.path.join('data/linemod_onepose/lm_full/models', cat, f'{cat}.ply')
                )
                    
                    
                
                for key in predictions['metrics']:
                    if key not in scene_metrics:
                        scene_metrics[key] = []
                    scene_metrics[key].append(predictions['metrics'][key])
                
            except Exception as e:
                import traceback
                loguru.logger.error(f"Error in processing {scene}")
                loguru.logger.error(e)
                
                loguru.logger.error(traceback.format_exc())
                continue
                    
        # dump metrics to file
        # for add metrics: calculate mean value
        # for pose metrics & reproj2d metrics: calculate auc
        if scene_metrics.__len__() == 0:
            continue
        mean_add = sum(scene_metrics['add']) / len(scene_metrics['add'])
        pose_5deg_5cm = auc_pair(scene_metrics['rotation_error'], scene_metrics['translation_error_meter']*100, 5)
        pose_10deg_10cm = auc_pair(scene_metrics['rotation_error'], scene_metrics['translation_error_meter']*100, 10)
        pose_20deg_20cm = auc_pair(scene_metrics['rotation_error'], scene_metrics['translation_error_meter']*100, 20)
        pose_30deg_30cm = auc_pair(scene_metrics['rotation_error'], scene_metrics['translation_error_meter']*100, 30)
        
        reprojection_2d_5p = auc(scene_metrics['proj2d'], 5)
        reprojection_2d_10p = auc(scene_metrics['proj2d'], 10)
        reprojection_2d_20p = auc(scene_metrics['proj2d'], 20)
        
        
        # save above metrics to output dir (json)
        import json
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump({
                'mean_add': mean_add,
                'pose': {
                    '5deg_5cm': pose_5deg_5cm,
                    '10deg_10cm': pose_10deg_10cm,
                    '20deg_20cm': pose_20deg_20cm,
                    '30deg_30cm': pose_30deg_30cm,
                },
                'reproj2d':{
                    '5p': reprojection_2d_5p,
                    '10p': reprojection_2d_10p,
                    '20p': reprojection_2d_20p,
                },
                'len': len(scene_metrics['add']),
            }, f)
        

        loguru.logger.info(f"Scene {scene} done")

    return True

def auc(data, threshold):
    data = sorted(data)
    n = len(data)
    for i in range(n):
        if data[i] > threshold:
            break
    return i / n

def auc_pair(data1, data2, thres):
    n = len(data1)
    cnt = 0
    for i in range(n):
        if data1[i] < thres and data2[i] < thres:
            cnt += 1
    return cnt / n

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

if __name__ == "__main__":
    with torch.no_grad():
        demo_fn()
