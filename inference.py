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





# ----------------------------------------------------------------------------
# MODIFICATION NOTICE:
# This file was modified by NI-Tex.
# Changes include: 
# - Fused GeoRender (Stage 1), Diffusion Inference (Stage 2), and Texture Baking (Stage 3).
# - Integrated Super-Resolution (Enhance) pipeline prior to UV baking for high-fidelity textures.
# - Optimized output directory structure for cleanly separated PBR texture maps.
# - Implemented explicit VRAM clearing across all stages to prevent Out-Of-Memory (OOM) errors.
# - Enhanced CLI arguments for flexible custom file paths or automated dataset processing.
# ----------------------------------------------------------------------------

import os
import sys
import gc
import json
import shutil
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from diffusers.utils import logging as diffusers_logging

warnings.filterwarnings("ignore")
diffusers_logging.set_verbosity(50)

# Project specific imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.train_util import instantiate_from_config
from DifferentiableRenderer.MeshRender import MeshRender
from utils.simplify_mesh_utils import remesh_mesh
from utils.pipeline_utils import ViewProcessor
from utils.uvwrap_utils import mesh_uv_wrap
from utils.image_super_utils import imageSuperNet  # Super Resolution Model
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb

try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    pass

from hunyuanpaintpbr.unet.model import HunyuanPaint 


# ==========================================================================================
# HELPER FUNCTIONS
# ==========================================================================================

def load_image(pil_img, color, image_size=512):
    """Loads a single image, handles alpha channel, and converts to tensor."""
    if isinstance(pil_img, (str, Path)):
        pil_img = Image.open(str(pil_img))
        
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
        
    pil_img = pil_img.resize((image_size, image_size))
    image = np.asarray(pil_img, dtype=np.float32) / 255.0
    
    if image.shape[2] == 3:
        image = image[:, :, :3]
        alpha = np.ones_like(image)
    else:
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)
        
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
    
    return image, alpha

def load_multiview_tensors(folder_path, expected_views=10, view_size=512):
    """Loads a directory of multiview maps into a single batched tensor."""
    valid_extensions = ('.png', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
    
    tensors = []
    for f in files[:expected_views]:
        img_path = os.path.join(folder_path, f)
        img = Image.open(img_path).convert("RGB")
        img_t = TF.to_tensor(img) 
        img_t = TF.resize(img_t, [view_size, view_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        tensors.append(img_t)
        
    if not tensors:
        raise ValueError(f"No valid images found in {folder_path}")
        
    return torch.stack(tensors).unsqueeze(0)


# ==========================================================================================
# CONFIGURATION
# ==========================================================================================

class Hunyuan3DPaintConfig:
    """Configuration class managing rendering, baking, super-resolution, and camera views."""
    def __init__(self, orth_scale: float = 1.35, max_num_view: int = 10, resolution: int = 512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resolution = resolution
        self.max_selected_view_num = max_num_view
        self.orth_scale = orth_scale
        
        # Rendering & Baking Settings
        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2   # High-res projection size
        self.texture_size = 1024 * 4  # Final UV texture map size
        self.bake_exp = 4
        self.merge_method = "fast"

        # Model Checkpoints
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        # Predefined optimal camera viewing angles (Azimuth & Elevation)
        self.candidate_camera_azims = [0, 180, 90, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1.0, 0.5, 0.1, 0.1, 0.05, 0.05]

        # Add diagonal views for maximum surface coverage
        for azim in [45, 135]:
            self.candidate_camera_azims.extend([azim, -azim])
            self.candidate_camera_elevs.extend([0, 0])
            self.candidate_view_weights.extend([0.01, 0.01])


# ==========================================================================================
# STAGE 1: GEOMETRY RENDER PIPELINE
# ==========================================================================================

def run_stage1_georender(mesh_path: Path, output_dir: Path, conf: Hunyuan3DPaintConfig):
    """Renders geometric features (normal & position maps) from optimal viewpoints."""
    print("\n--- [STAGE 1] Geometry Rendering ---")
    
    render = MeshRender(
        default_resolution=conf.render_size,
        texture_size=conf.texture_size,
        bake_mode=conf.bake_mode,
        raster_mode=conf.raster_mode,
        orth_scale=conf.orth_scale
    )
    view_processor = ViewProcessor(conf, render)

    normal_dir = output_dir / "geometry" / "normal"
    position_dir = output_dir / "geometry" / "position"
    normal_dir.mkdir(parents=True, exist_ok=True)
    position_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and wrapping UVs for mesh: {mesh_path.name}")
    mesh = trimesh.load(str(mesh_path), process=False)
    mesh = mesh_uv_wrap(mesh)
    render.load_mesh(mesh=mesh)

    print("Calculating optimal views and rendering maps...")
    elevs, azims, weights = view_processor.bake_view_selection(
        conf.candidate_camera_elevs, conf.candidate_camera_azims,
        conf.candidate_view_weights, conf.max_selected_view_num,
    )
    
    normal_maps = view_processor.render_normal_multiview(elevs, azims, use_abs_coor=False)
    position_maps = view_processor.render_position_multiview(elevs, azims)

    for i in range(conf.max_selected_view_num):
        img_filename = f"{i:03d}_0001.png" 
        normal_maps[i].save(normal_dir / img_filename)
        position_maps[i].save(position_dir / img_filename)
        
    print(f"Geometry maps saved to: {output_dir / 'geometry'}")
    
    del render, view_processor, mesh
    gc.collect()
    torch.cuda.empty_cache()
    
    return normal_dir, position_dir, elevs, azims, weights


# ==========================================================================================
# STAGE 2: DIFFUSION TEXTURE GENERATION
# ==========================================================================================

def run_stage2_diffusion(config_path, image_prompt_path, normal_dir, position_dir, output_dir, device):
    """Generates 2D texture representations using the HunyuanPaint diffusion model."""
    print("\n--- [STAGE 2] Multiview Texture Generation (Validation Mode) ---")
    
    config = OmegaConf.load(config_path)
    ckpt_path = config.get("resume_from") 
    
    print("Extracting model parameters from config...")
    init_kwargs = OmegaConf.to_container(config.model.params, resolve=True)

    print("Instantiating base model...")
    model = HunyuanPaint(**init_kwargs)

    # Patch UNet conv_in to accept the required number of channels
    nc = init_kwargs.get("noise_in_channels")
    if nc is not None:
        print(f"Modifying UNet conv_in to accept {nc} channels...")
        model_unet = model.unet.unet if hasattr(model.unet, "unet") else model.unet
        
        with torch.no_grad():
            old_conv = model_unet.conv_in
            new_conv = torch.nn.Conv2d(
                nc, old_conv.out_channels, old_conv.kernel_size, 
                old_conv.stride, old_conv.padding
            )
            new_conv.weight.zero_()
            new_conv.bias.zero_()
            model_unet.conv_in = new_conv

    print(f"Loading weights from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Load state dict with strict=False to tolerate missing DINO features
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    
    if hasattr(model, 'pipeline'):
        model.pipeline.to(device)

    output_dir = Path(output_dir)
    model.logdir = str(output_dir)
    os.makedirs(os.path.join(model.logdir, "textures"), exist_ok=True)

    resolution = model.view_size
    N_views = model.num_view
    
    print("Loading image and geometry tensors into VRAM...")
    
    bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ref_tensor, _ = load_image(image_prompt_path, color=bg_color, image_size=resolution)
    ref_tensor = ref_tensor.unsqueeze(0).unsqueeze(0).to(device)

    normal_tensor = load_multiview_tensors(normal_dir, N_views, resolution).to(device)
    if normal_tensor.dim() == 4:
        normal_tensor = normal_tensor.unsqueeze(0)
        
    position_tensor = load_multiview_tensors(position_dir, N_views, resolution).to(device)
    if position_tensor.dim() == 4:
        position_tensor = position_tensor.unsqueeze(0)

    dummy_target = torch.zeros((1, N_views, 3, resolution, resolution), device=device)

    # Construct batch mapping
    batch = {
        "images_cond": ref_tensor,
        "images_normal": normal_tensor,
        "images_position": position_tensor,
        "images_ref_paths": [[str(image_prompt_path)]], 
        "name": ["demo_output"] 
    }
    for pbr_token in model.pbr_settings:
        batch[f"images_{pbr_token}"] = dummy_target

    print("Running Diffusion Pipeline (Denosing via validation_step)...")
    
    # Ensure cohesive bf16 precision across the entire pipeline
    model.to(device, dtype=torch.bfloat16)
    if hasattr(model, 'pipeline'):
        model.pipeline.to(device, dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.validation_step(batch, batch_idx=0)

    del model, normal_tensor, position_tensor, ref_tensor, dummy_target, batch
    gc.collect()
    torch.cuda.empty_cache()

    # Organize generated assets
    print("\nReorganizing generated textures to the target format...")
    shutil.copy(image_prompt_path, output_dir / "image_prompt.png")
    
    base_name = "demo_output"
    suffix = os.path.splitext(os.path.basename(image_prompt_path))[0].split('_')[0]
    temp_output_path = output_dir / "images_val" / "images" / f"{base_name}_{suffix}"
    
    pbr_types = ["albedo", "mr"]
    for group_idx in range(N_views):
        for view_idx, pbr_name in enumerate(pbr_types):
            src_file = temp_output_path / f"{group_idx+1}_{view_idx+1}.png"
            dst_folder = output_dir / "textures" / pbr_name
            dst_folder.mkdir(parents=True, exist_ok=True)
            
            dst_file = dst_folder / f"view_{group_idx:03d}.png"
            if src_file.exists():
                shutil.move(str(src_file), str(dst_file))
                
    # Cleanup validation temp directory
    val_dir_to_remove = output_dir / "images_val"
    if val_dir_to_remove.exists():
        shutil.rmtree(val_dir_to_remove)
        
    print(f"Extraction complete! Final assets are properly sorted in: {output_dir}")
    return output_dir / "textures"


# ==========================================================================================
# STAGE 3: TEXTURE ENHANCEMENT & BAKING
# ==========================================================================================

def run_stage3_baking(mesh_path: Path, texture_dir: Path, output_dir: Path, elevs, azims, weights, conf: Hunyuan3DPaintConfig, do_enhance: bool = True):
    """Enhances 2D textures via Super-Resolution and bakes them onto the 3D mesh UV map."""
    print(f"\n--- [STAGE 3] Texture Baking & Inpainting (Enhance: {do_enhance}) ---")
    
    print("Loading generated PBR maps from disk...")
    albedo_imgs = []
    mr_imgs = []
    
    for i in range(conf.max_selected_view_num):
        albedo_path = texture_dir / "albedo" / f"view_{i:03d}.png"
        mr_path = texture_dir / "mr" / f"view_{i:03d}.png"
        
        albedo_imgs.append(Image.open(albedo_path).convert("RGBA"))
        if mr_path.exists():
            mr_imgs.append(Image.open(mr_path).convert("RGBA"))

    if do_enhance:
        print("Initializing Super Resolution Model...")
        super_model = imageSuperNet(conf)
        
        print("Enhancing Albedo maps...")
        for i in range(len(albedo_imgs)):
            albedo_imgs[i] = super_model(albedo_imgs[i])
            
        if mr_imgs:
            print("Enhancing Metallic-Roughness (MR) maps...")
            for i in range(len(mr_imgs)):
                mr_imgs[i] = super_model(mr_imgs[i])
                
        # Destroy Super Resolution model to reclaim VRAM for Baking
        del super_model
        gc.collect()
        torch.cuda.empty_cache()

    for i in range(len(albedo_imgs)):
        albedo_imgs[i] = albedo_imgs[i].resize((conf.render_size, conf.render_size))
    for i in range(len(mr_imgs)):
        mr_imgs[i] = mr_imgs[i].resize((conf.render_size, conf.render_size))

    print("Initializing Mesh Renderer & UV Wrapper...")
    render = MeshRender(
        default_resolution=conf.render_size,
        texture_size=conf.texture_size,
        bake_mode=conf.bake_mode,
        raster_mode=conf.raster_mode,
        orth_scale=conf.orth_scale
    )
    view_processor = ViewProcessor(conf, render)

    mesh = trimesh.load(str(mesh_path), process=False)
    mesh = mesh_uv_wrap(mesh)
    render.load_mesh(mesh=mesh)

    print("Baking Albedo texture to UV map...")
    texture_albedo, mask_albedo = view_processor.bake_from_multiview(albedo_imgs, elevs, azims, weights)
    mask_albedo_np = (mask_albedo.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    
    texture_albedo = view_processor.texture_inpaint(texture_albedo, mask_albedo_np)
    render.set_texture(texture_albedo, force_set=True) 

    if mr_imgs:
        print("Baking Metallic-Roughness texture to UV map...")
        texture_mr, mask_mr = view_processor.bake_from_multiview(mr_imgs, elevs, azims, weights)
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_mr = view_processor.texture_inpaint(texture_mr, mask_mr_np)
        render.set_texture_mr(texture_mr)

    print("Exporting final 3D models...")
    obj_path = output_dir / "results.obj"
    glb_path = output_dir / "results.glb"
    
    render.save_mesh(str(obj_path), downsample=True)

    with open(output_dir / "bake_weight.txt", "w", encoding="utf-8") as f:
        data_dict = {
            "selected_camera_azims": azims,
            "selected_camera_elevs": elevs,
            "selected_view_weights": weights
        }
        for key, value in data_dict.items():
            f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")

    convert_obj_to_glb(str(obj_path), str(glb_path))
    print(f"\n✅ Pipeline Complete! Fully textured 3D asset ready at: {glb_path}")


# ==========================================================================================
# MAIN ENTRY
# ==========================================================================================

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser(description="NI-Tex Unified End-to-End Inference Pipeline")
    
    # Option A: Standard automated mode (requires specific folder structure)
    parser.add_argument("--name", type=str, default=None, 
                        help="Target asset name. Automatically searches in 'asset/cases/{name}/'")
    
    # Option B: Flexible custom mode (allows providing explicit file paths)
    parser.add_argument("--mesh_path", type=str, default=None, help="Explicit path to input .glb or .obj mesh")
    parser.add_argument("--image_prompt", type=str, default=None, help="Explicit path to the reference image prompt")
    parser.add_argument("--output_dir", type=str, default="InferenceResults", help="Base directory to save results")
    
    # Pipeline Configurations 
    parser.add_argument("--base", type=str, default="cfgs/inference.yml", help="Path to model config")
    parser.add_argument("--orth_scale", type=float, default=1.35, help="Orthogonal scale for rendering")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use (e.g., '0' or '0,1')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_enhance", action="store_true", help="Disable Super-Resolution enhancement before baking to save time.")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # Resolve inputs dynamically
    if args.name:
        mesh_path = Path(f"asset/cases/{args.name}/mesh.glb")
        image_prompt_path = Path(f"asset/cases/{args.name}/image_prompt.png")
        output_dir = Path(args.output_dir) / args.name
    else:
        if not args.mesh_path or not args.image_prompt:
            raise ValueError("You must provide either '--name' OR both '--mesh_path' and '--image_prompt'.")
        mesh_path = Path(args.mesh_path).resolve()
        image_prompt_path = Path(args.image_prompt).resolve()
        output_dir = Path(args.output_dir) / mesh_path.stem

    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing required mesh: {mesh_path}")
    if not image_prompt_path.exists():
        raise FileNotFoundError(f"Missing required image prompt: {image_prompt_path}")
        
    global_conf = Hunyuan3DPaintConfig(orth_scale=args.orth_scale, max_num_view=10, resolution=512)
    do_enhance = not args.no_enhance

    # ---------------------------------------------------------
    # Execution Flow
    # ---------------------------------------------------------
    
    normal_dir, position_dir, elevs, azims, weights = run_stage1_georender(
        mesh_path=mesh_path, 
        output_dir=output_dir, 
        conf=global_conf
    )
    
    texture_dir = run_stage2_diffusion(
        config_path=args.base,
        image_prompt_path=image_prompt_path,
        normal_dir=normal_dir,
        position_dir=position_dir,
        output_dir=output_dir,
        device=device
    )
    
    run_stage3_baking(
        mesh_path=mesh_path,
        texture_dir=texture_dir,
        output_dir=output_dir,
        elevs=elevs,
        azims=azims,
        weights=weights,
        conf=global_conf,
        do_enhance=do_enhance
    )