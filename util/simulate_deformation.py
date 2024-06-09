import os

import numpy as np
import torch
import SimpleITK as sitk
from scipy.interpolate import RBFInterpolator

from torchrbf import RBFInterpolator as RBFInterpolatorGPU


def find_control_points(mask):
    # Identify all non-zero (prostate) points
    z_indices, y_indices, x_indices = np.where(mask == 1)

    # Middle of the prostate in the depth (z) direction
    z_mid = z_indices[len(z_indices) // 2]

    # For the sagittal plane (mid z-plane)
    sagittal_plane = mask[z_mid, :, :]
    y_sagittal, x_sagittal = np.where(sagittal_plane == 1)

    y_sagittal = np.sort(y_sagittal)
    x_sagittal = np.sort(x_sagittal)

    # Central left and right points in the sagittal plane
    central_left = (z_mid, y_sagittal[len(y_sagittal) // 2], x_sagittal[0])
    central_right = (z_mid, y_sagittal[len(y_sagittal) // 2], x_sagittal[-1])

    x_indices = np.sort(x_indices)
    x_mid = x_indices[len(x_indices) // 2]

    # For the coronal plane (mid x-plane)
    coronal_plane = mask[:, :, x_mid]
    z_coronal, y_coronal = np.where(coronal_plane == 1)

    z_coronal = np.sort(z_coronal)
    y_coronal = np.sort(y_coronal)

    # Central left and right points in the coronal plane
    central_left_coronal = (z_coronal[0], y_coronal[len(y_coronal) // 2], x_mid)
    central_right_coronal = (z_coronal[-1], y_coronal[len(y_coronal) // 2], x_mid)

    central_top = (z_coronal[len(z_coronal) // 2], y_coronal[0], x_mid)
    central_bottom = (z_coronal[len(z_coronal) // 2], y_coronal[-1], x_mid)

    return central_top, central_bottom, central_left, central_right, central_left_coronal, central_right_coronal


def get_world_coordinates(image_path, voxel_coordinates):
    # Load the image using SimpleITK
    image = sitk.ReadImage(image_path)

    # Ensure coordinates are in the correct format (tuple of int)
    transformed_coordinates = [
        tuple(int(x) for x in voxel) for voxel in voxel_coordinates
    ]

    # Transform voxel coordinates to physical (world) coordinates
    world_coordinates = [
        image.TransformIndexToPhysicalPoint(voxel) for voxel in transformed_coordinates
    ]

    return world_coordinates


def update_control_points(top, bottom, left, right, left_coronal, right_coronal):
    # Convert to numpy arrays for vector operations
    top = np.array(top)
    bottom = np.array(bottom)
    left = np.array(left)
    right = np.array(right)
    left_coronal = np.array(left_coronal)
    right_coronal = np.array(right_coronal)

    # Calculate the vector from bottom to top
    bottom_to_top_vector = top - bottom

    # move the top point upwards by random percentage between 20 and 50 % of the distance between the top and bottom points
    move_factor = np.random.uniform(0.35, 0.4)
    # Move the bottom point upwards 20% towards the top point
    new_bottom = bottom + move_factor * bottom_to_top_vector

    # Calculate the vector from left to right
    left_to_right_vector = right - left

    # Move left and right points outwards by 5% of their distance
    new_left = left - 0.05 * left_to_right_vector
    new_right = right + 0.05 * left_to_right_vector

    # Calculate the vector from left to right coronal
    left_to_right_coronal_vector = right_coronal - left_coronal

    # Move left and right coronal points outwards by 5% of their distance
    new_left_coronal = left_coronal - 0.05 * left_to_right_coronal_vector
    new_right_coronal = right_coronal + 0.05 * left_to_right_coronal_vector

    return new_bottom, new_left, new_right, new_left_coronal, new_right_coronal


def get_range_from_image(image_path):
    img = sitk.ReadImage(image_path)
    origin = np.array(img.GetOrigin())
    final = np.array(img.GetDirection()).reshape((3, 3)) @ np.array(img.GetSize()) * np.array(img.GetSpacing()) + origin
    c = np.stack((origin, final))
    origin = c.min(axis=0)
    final = c.max(axis=0)
    return origin, final


def calculate_deformation_gpu(non_deformed_points, deformed_points, image_path, subfolder_path, device='cuda'):
    origin, final = get_range_from_image(image_path)
    ddf_spacing = [6, 6, 6]

    # add margin to the origin and final, in the direction of each axis based on origin-final direction
    origin = origin - 0.1 * (final - origin)
    final = final + 0.1 * (final - origin)

    # create a grid of points with the same spacing as the DDF
    x = torch.linspace(origin[0], final[0], int((final[0] - origin[0]) // ddf_spacing[0]))
    y = torch.linspace(origin[1], final[1], int((final[1] - origin[1]) // ddf_spacing[1]))
    z = torch.linspace(origin[2], final[2], int((final[2] - origin[2]) // ddf_spacing[2]))

    dtype = torch.float32

    grid_points = torch.meshgrid(x, y, z, indexing='ij')
    flat_points = torch.stack(grid_points, dim=-1).reshape(-1, 3).to(device).to(dtype)

    # Calculate the deformation field based on the non-deformed and deformed points
    node_def = non_deformed_points - deformed_points
    node_def = torch.tensor(node_def, device=device).to(dtype)
    non_deformed_mesh = torch.tensor(non_deformed_points, device=device).to(dtype)

    def_grid_x = interp_1d_deformation_gpu(node_def, non_deformed_mesh, flat_points, grid_points, axis=0)
    def_grid_y = interp_1d_deformation_gpu(node_def, non_deformed_mesh, flat_points, grid_points, axis=1)
    def_grid_z = interp_1d_deformation_gpu(node_def, non_deformed_mesh, flat_points, grid_points, axis=2)

    full_grid = torch.stack([def_grid_x, def_grid_y, def_grid_z], dim=3).to(torch.float64)
    full_grid = torch.permute(full_grid, dims=[2, 1, 0, 3])

    full_grid = full_grid.cpu().numpy().astype(np.float64)

    img = sitk.GetImageFromArray(full_grid)
    img.SetOrigin(origin)
    img.SetDirection(sitk.ReadImage(image_path).GetDirection())
    img.SetSpacing(ddf_spacing)

    field_path = os.path.join(subfolder_path, "def_field.mha")
    sitk.WriteImage(img, field_path)
    print("deformation saved")

    return img


def apply_deformation(image_path, tx, interpolation=sitk.sitkLinear, minimum_intensity=0.0):
    image = sitk.ReadImage(image_path)
    # Create a resampler object
    resampler = sitk.ResampleImageFilter()
    # Set the resampler parameters
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(interpolation)
    resampler.SetDefaultPixelValue(np.double(minimum_intensity))
    resampler.SetTransform(tx)
    # Apply the transformation
    transformed_image = resampler.Execute(image)
    return transformed_image


def interp_1d_deformation_gpu(node_def, non_deformed_mesh, flat_points, grid_points, axis, sampling_freq=50):
    interp = RBFInterpolatorGPU(non_deformed_mesh, node_def[:, axis], device='cuda')
    y_node = interp(non_deformed_mesh)
    if (y_node - node_def[:, axis]).abs().mean() > 0.2:
        interp = RBFInterpolatorGPU(non_deformed_mesh, node_def[:, axis], device='cuda', smoothing=0.2)
    y_flat = interp(flat_points)
    y_grid = torch.reshape(y_flat, grid_points[0].shape)
    return y_grid


def interp_1d_deformation(node_def, non_deformed_mesh, x_flat, x_grid, axis, sampling_freq=50):
    interp = RBFInterpolator(non_deformed_mesh.points[::sampling_freq, :], node_def[::sampling_freq, axis])
    y_flat = interp(x_flat)
    y_grid = np.reshape(y_flat, list(x_grid.shape)[1:])
    return y_grid


def simulate_deformation(mask: torch.Tensor, label_path: str, image_path: str, image_min: float):
    mask = mask.squeeze().cpu().numpy()
    # Finding the control points
    control_points = find_control_points(mask)
    world_coordinates = get_world_coordinates(label_path, control_points)
    control_points_names = ['Central Top', 'Central Bottom', 'Central Left', 'Central Right', 'Central Coronal Left',
                            'Central Coronal Right']

    # for name, point, world_point in zip(control_points_names, control_points, world_coordinates):
    #     print(f"{name}: {point}")
    #     print(f"{name} (world): {world_point}")

    # Update the control points
    new_bottom, new_left, new_right, new_left_coronal, new_right_coronal = update_control_points(*world_coordinates)
    new_top = world_coordinates[0]

    new_world_coordinate = [new_top, new_bottom, new_left, new_right, new_left_coronal, new_right_coronal]
    new_world_coordinate = np.array(new_world_coordinate)
    world_coordinates = np.array(world_coordinates)

    deformation_field = calculate_deformation_gpu(world_coordinates, new_world_coordinate, label_path, "/tmp/")
    tx = sitk.DisplacementFieldTransform(deformation_field)

    if image_min < 0:
        image_min = 0.0

    deformed_image = apply_deformation(image_path, tx, minimum_intensity=image_min)
    deformed_label = apply_deformation(label_path, tx, interpolation=sitk.sitkNearestNeighbor)

    return deformed_image, deformed_label
