# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import time
import glob
import json
import random
import numpy as np
import torch
from .loader_util import BaseDataset
import re
from PIL import Image

class TextureDataset(BaseDataset):

    def __init__(
        self, json_path, num_view=6, image_size=512, lighting_suffix_pool=["0000", "0001", "0002","0003", "0004", "0005","0006"]
    ):
        self.data = list()
        self.num_view = num_view
        self.image_size = image_size
        self.lighting_suffix_pool = lighting_suffix_pool
        if isinstance(json_path, str):
            json_path = [json_path]
        for jp in json_path:
            with open(jp) as f:
                self.data.extend(json.load(f))
        
        
        print("============= length of dataset %d =============" % len(self.data))

    def __getitem__(self, index):
        try:
            dirx = self.data[index]
            basename = os.path.basename(dirx)
            
            # 1. Identify dataset type based on naming convention
            is_bedlam = "rp" in basename 
            
            # 2. Preset standard background colors (RGB)
            bg_white, bg_black, bg_gray = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [127/255.0, 127/255.0, 127/255.0]

            # 3. Define condition strategy and initial states
            # Strategies: "general" (Objeverse/Texverse), "Bedlam" (3D Garment Videos), "Bedlam_edited" (edited by Nano Banana)
            cond_strategy = "Bedlam" if is_bedlam else "general"
            dirx_cond = dirx
            ref_image_path = None

            # ---------------------------------------------------------
            # Determine Condition Strategy
            # ---------------------------------------------------------
            if is_bedlam:
                # Strategy A: "Bedlam" (3D Garment Videos)
                inf = basename.rsplit('_', 2)
                if len(inf) == 3:
                    main_id, texture_id = inf[0], inf[-1]
                    candidates_same_char = [
                        p for p in self.data 
                        if p != dirx and os.path.basename(p).startswith(main_id) and os.path.basename(p).endswith(texture_id)
                    ]
                    if candidates_same_char:
                        dirx_cond = random.choice(candidates_same_char)

                # Strategy B: "Bedlam_edited" (edited by Nano Banana) (50% probability)
                dst_dir = re.sub(
                    r"/06\.Bedlam_Dataset/outputs_\d+/",
                    "DATASET/Bedlam_edited_by_NanoBanana/",
                    dirx
                )
                filename = re.sub(r"\+\d+_\d+", "", os.path.basename(dst_dir))
                dst = os.path.join(os.path.dirname(dst_dir), filename)

                if os.path.exists(dst):
                    candidates_prompt = [
                        os.path.join(dst, f) for f in os.listdir(dst)
                        if f.lower().startswith("image_prompt") and f.lower().endswith(".png")
                    ]
                    if candidates_prompt and random.random() >= 0.5:
                        cond_strategy = "Bedlam_edited"
                        ref_image_path = random.choice(candidates_prompt)
                        dirx_cond = ref_image_path 

            # ---------------------------------------------------------
            # Extract Reference Image & MR Recoloring Value
            # ---------------------------------------------------------
            mr_recolor_value = None 
            
            if cond_strategy != "Bedlam_edited":
                # Sample a random condition image from the rendering folder
                cond_images = []
                rend_dir = os.path.join(dirx_cond, "rendering")
                for i in range(8):
                    for ext in ["png", "jpg", "jpeg"]:
                        cond_images.extend(glob.glob(os.path.join(rend_dir, f"{i:03d}_*.{ext}")))
                
                if not cond_images:
                    raise FileNotFoundError(f"No rendering images found in {rend_dir}")
                ref_image_path = random.choice(cond_images)

                # Pre-extract unique color from MR map for the "Bedlam" strategy
                if cond_strategy == "Bedlam":
                    mr_exts = ["*_0001.png", "*_0001.jpg", "*_0001.jpeg"]
                    mr_pixels = []
                    for ext in mr_exts:
                        mr_pixels.extend(glob.glob(os.path.join(dirx_cond, "roughness_metallic", ext)))
                    
                    img_mr = Image.open(mr_pixels[0]).convert("RGBA")
                    non_transparent = [p[:3] for p in img_mr.getdata() if p[3] > 0]
                    unique_colors = set(non_transparent) # Appendix A.1
                    
                    if len(unique_colors) != 1:
                        raise ValueError(f"MR map in {dirx_cond} must contain exactly one solid color.")
                    mr_recolor_value = unique_colors.pop()

            images_ref_paths = [ref_image_path]

            # ---------------------------------------------------------
            # Collect and Sort Multi-view Images
            # ---------------------------------------------------------
            available_views = []
            for ext in ["*_0001.png", "*_0001.jpg", "*_0001.jpeg"]:
                available_views.extend(glob.glob(os.path.join(dirx, "albedo", ext)))

            # Fallback to available views if the requested number is not met
            images_gen = available_views if len(available_views) < self.num_view else random.sample(available_views, self.num_view)
            images_gen.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

            # ---------------------------------------------------------
            # Image Loading & Augmentation
            # ---------------------------------------------------------
            # Determine randomized background color for reference image
            rand_bg = random.random()
            bg_c_record = bg_gray if rand_bg < 0.6 else (bg_black if rand_bg < 0.8 else bg_white)

            # 1. Process reference image
            images_ref = []
            for img_path in images_ref_paths:
                image, _ = self.load_image(img_path, bg_c_record)
                if cond_strategy != "Bedlam_edited":
                    image = self.augment_image(image, bg_c_record, identity_prob=0).float()
                images_ref.append(image)

            # 2. Process all PBR maps (Albedo, MR, Normal, Position)
            images_albedo, images_mr, images_normal, images_position = [], [], [], []
            
            for img_gen in images_gen:
                # Albedo Map
                albedo_img, _ = self.load_image(img_gen, bg_gray)
                images_albedo.append(self.augment_image(albedo_img, bg_gray))
                
                # Metallic-Roughness Map
                mr_path = img_gen.replace("albedo", "roughness_metallic")
                if cond_strategy == "general":
                    mr_img, _ = self.load_image(mr_path, bg_gray)
                elif cond_strategy == "Bedlam":
                    mr_img, _ = self.recolor_image_with_background(mr_path, mr_recolor_value, bg_gray) # Appendix A.1
                elif cond_strategy == "Bedlam_edited":
                    mr_img, _ = self.recolor_image_with_background(mr_path, (0, 0, 0), bg_gray)
                images_mr.append(self.augment_image(mr_img, bg_gray))

                # Normal & Position Maps
                norm_img, _ = self.load_image(img_gen.replace("albedo", "normal"), bg_gray)
                images_normal.append(self.augment_image(norm_img, bg_gray))
                
                pos_img, _ = self.load_image(img_gen.replace("albedo", "position"), bg_gray)
                images_position.append(self.augment_image(pos_img, bg_gray))

            # ---------------------------------------------------------
            # Final Output Tensor Dictionary
            # ---------------------------------------------------------
            return {
                "images_cond": torch.stack(images_ref, dim=0).float(),
                "images_albedo": torch.stack(images_albedo, dim=0).float(),
                "images_mr": torch.stack(images_mr, dim=0).float(),
                "images_normal": torch.stack(images_normal, dim=0).float(),
                "images_position": torch.stack(images_position, dim=0).float(),
                "name": dirx,
                "images_ref_paths": images_ref_paths
            }

        except (FileNotFoundError, OSError, ValueError) as e:
            # Robust fallback: randomly sample another index to prevent DataLoader crash
            return self.__getitem__(random.randint(0, len(self.data) - 1))


if __name__ == "__main__":
    dataset = TextureDataset(json_path=["train_examples/examples_train_Objaverse_Texverse.json"])
    print("images_cond", dataset[0]["images_cond"].shape)
    print("images_albedo", dataset[0]["images_albedo"].shape)
    print("images_mr", dataset[0]["images_mr"].shape)
    print("images_normal", dataset[0]["images_normal"].shape)
    print("images_position", dataset[0]["images_position"].shape)
    print("name", dataset[0]["name"])
