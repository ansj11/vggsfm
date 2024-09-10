import os
import sys

scene_root = "data"

# list all scenes with co3d_ or lm_ prefix
scenes = [f for f in os.listdir(scene_root) if f.startswith("co3d_") or f.startswith("lm_")]

# cmd python relocalization_demo.py SCENE_DIR=data/*

for scene in scenes:
    os.system(f"python relocalization_demo.py SCENE_DIR=data/{scene}")
    # wait for the process to finish
    print(f"Finished processing scene {scene}")