import numpy as np
import open3d as o3d
from skimage.morphology import skeletonize
from skan.csr import skeleton_to_csgraph, Skeleton
""" voxelizes the mesh and normalizes it to a unit cube via open3d """
def voxelize_mesh(mesh, voxel_size=0.005):

    # Normalize mesh
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    # Use open3d to voxelize the mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)


    return voxel_grid


""" performs thinning on voxelized object via skimage.morphology """
def thinning(voxels):
    # Create empty 3D grid and fill occupied voxels
    voxel_indices = [voxel.grid_index for voxel in voxels.get_voxels()]
    voxel_indices = np.array(voxel_indices)

    # Determine grid size
    grid_shape = voxel_indices.max(axis=0) + 1
    volume = np.zeros(grid_shape, dtype=np.uint32)

    # Assign label '1' to all foreground voxels
    for idx in voxel_indices:
        volume[tuple(idx)] = 1

    binary_grid = np.zeros(grid_shape, dtype=np.uint8)
    for idx in voxel_indices:
        binary_grid[tuple(idx)] = 1

    # Skeletonize
    skeleton = skeletonize(binary_grid)

    return skeleton


""" TODO: extract centerlines from a thinn sheets of a skeleton as a graph"""
def visualize_centerlines(skeleton):
    skeleton_graph = Skeleton(skeleton)
    paths = skeleton_graph.paths_list()
    shape = skeleton.shape

    points = []
    lines = []
    offset = 0

    for path in paths:
        coords_zyx = np.array(np.unravel_index(path, shape)).T
        coords_xyz = coords_zyx[:, [2, 1, 0]]

        points.extend(coords_xyz)

        # Create line segments between consecutive points
        lines.extend([[i + offset, i + 1 + offset] for i in range(len(coords_xyz) - 1)])
        offset += len(coords_xyz)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))

    color = [1.0, 0.2, 0.2]  # red-ish
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def visualize_skeleton(skeleton, voxel_size=0.005):
    # Visualize slices or point cloud of skeleton
    skeleton_points = np.argwhere(skeleton)

    # Convert to Open3D PointCloud for visualization
    points = skeleton_points.astype(np.float32) * voxel_size
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def main():
    # Load mesh
    bunny = o3d.data.BunnyMesh()
    #pathToMesh = "assets/harness.obj"
    path_to_mesh = bunny.path
    mesh = o3d.io.read_triangle_mesh(path_to_mesh)
    # An important parameter, if chosen too high, the resulting skeleton will have a point cloud representation
    # The ideal shape of skeleton has to be thin and connected via edges from the control points
    # Using the value of 0.005 results in a volumetric skeleton of a standford bunny which is not what you usually want from a thinning algorithm
    voxel_size = 0.005
    
    voxels = voxelize_mesh(mesh, voxel_size)
    # Perform thinning on the voxelized object
    thinned_voxels = thinning(voxels)

    # Actual centerlines
    center_lines = visualize_centerlines(thinned_voxels)

    # Visualize the skeleton
    pcd = visualize_skeleton(thinned_voxels)
    #Fix: draw skeleton over original mesh and not next to it
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([voxels, pcd, center_lines])



if __name__ == "__main__":
    main()