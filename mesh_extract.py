import os
import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import math
import numpy as np
from scene.cameras import Camera
from gaussian_renderer import render
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json
from PIL import Image
import cv2


def load_camera(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            args.source_path, args.images, args.eval
        )
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](
            args.source_path, args.white_background, args.eval
        )
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)


def compute_normal_map(depth_map):
    """Compute normal map from depth map."""
    zx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    zy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    normal_map = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    n = np.linalg.norm(normal_map, axis=2)
    normal_map /= n[:, :, np.newaxis]
    return normal_map


def normalize_depth(depth_np, lower_percentile=1, upper_percentile=99):
    """Normalize depth values, excluding extremities."""
    min_depth = np.percentile(depth_np[depth_np > 0], lower_percentile)
    max_depth = np.percentile(depth_np[depth_np > 0], upper_percentile)
    normalized = np.clip(depth_np, min_depth, max_depth)
    normalized = (normalized - min_depth) / (max_depth - min_depth)
    return (normalized * 255).astype(np.uint8)


def extract_mesh(dataset, pipe, checkpoint_iterations=None):
    gaussians = GaussianModel(dataset.sh_degree)
    output_path = os.path.join(dataset.model_path, "point_cloud")
    iteration = 0
    if checkpoint_iterations is None:
        for folder_name in os.listdir(output_path):
            iteration = max(iteration, int(folder_name.split("_")[1]))
    else:
        iteration = checkpoint_iterations
    output_path = os.path.join(
        output_path, "iteration_" + str(iteration), "point_cloud.ply"
    )

    gaussians.load_ply(output_path)
    print(f"Loaded gaussians from {output_path}")

    kernel_size = dataset.kernel_size

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_cam_list = load_camera(dataset)

    depth_list = []
    color_list = []
    alpha_thres = 0.5

    # Create directories to save depth maps, normal maps, and full data
    depth_save_dir = os.path.join(dataset.model_path, "depth_maps")
    normal_save_dir = os.path.join(dataset.model_path, "normal_maps")
    full_depth_save_dir = os.path.join(dataset.model_path, "full_depth_data")
    full_normal_save_dir = os.path.join(dataset.model_path, "full_normal_data")
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(normal_save_dir, exist_ok=True)
    os.makedirs(full_depth_save_dir, exist_ok=True)
    os.makedirs(full_normal_save_dir, exist_ok=True)

    for idx, viewpoint_cam in enumerate(viewpoint_cam_list):
        # Rendering offscreen from that camera
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        rendered_img = (
            torch.clamp(render_pkg["render"], min=0, max=1.0)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        color_list.append(np.ascontiguousarray(rendered_img))
        depth = render_pkg["median_depth"].clone()
        if viewpoint_cam.gt_mask is not None:
            depth[(viewpoint_cam.gt_mask < 0.5)] = 0
        depth[render_pkg["mask"] < alpha_thres] = 0
        depth_np = depth[0].cpu().numpy()
        depth_list.append(depth_np)

        # Use the original image name from camera info
        original_name = viewpoint_cam.image_name

        # Save visualized depth map
        depth_filename = f"{original_name}.png"
        depth_path = os.path.join(depth_save_dir, depth_filename)

        # Normalize depth for visualization, excluding extremities
        depth_normalized = normalize_depth(depth_np)
        depth_image = Image.fromarray(depth_normalized)
        depth_image.save(depth_path)
        print(f"Saved visualized depth map to {depth_path}")

        # Save full depth data as .pt file
        full_depth_filename = f"{original_name}.pt"
        full_depth_path = os.path.join(full_depth_save_dir, full_depth_filename)
        torch.save(depth, full_depth_path)
        print(f"Saved full depth data to {full_depth_path}")

        # Compute full normal map
        full_normal_map = compute_normal_map(depth_np)

        # Save full normal map data as .pt file
        full_normal_filename = f"{original_name}.pt"
        full_normal_path = os.path.join(full_normal_save_dir, full_normal_filename)
        torch.save(torch.from_numpy(full_normal_map), full_normal_path)
        print(f"Saved full normal map data to {full_normal_path}")

        # Save visualized normal map
        normal_map_vis = ((full_normal_map + 1) * 0.5 * 255).astype(np.uint8)
        normal_filename = f"{original_name}.png"
        normal_path = os.path.join(normal_save_dir, normal_filename)
        normal_image = Image.fromarray(normal_map_vis)
        normal_image.save(normal_path)
        print(f"Saved visualized normal map to {normal_path}")

    torch.cuda.empty_cache()
    voxel_size = 0.002
    o3d_device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=o3d_device,
    )
    for color, depth, viewpoint_cam in zip(color_list, depth_list, viewpoint_cam_list):
        depth = o3d.t.geometry.Image(depth)
        depth = depth.to(o3d_device)
        color = o3d.t.geometry.Image(color)
        color = color.to(o3d_device)
        W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
        fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.0))
        fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.0))
        intrinsic = np.array(
            [[fx, 0, float(W) / 2], [0, fy, float(H) / 2], [0, 0, 1]], dtype=np.float64
        )
        intrinsic = o3d.core.Tensor(intrinsic)
        extrinsic = o3d.core.Tensor(
            (viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64)
        )
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, 1.0, 8.0
        )
        vbg.integrate(
            frustum_block_coords, depth, color, intrinsic, extrinsic, 1.0, 8.0
        )

    mesh = vbg.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(
        os.path.join(dataset.model_path, "recon.ply"), mesh.to_legacy()
    )
    print("done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)
