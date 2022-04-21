import numpy as np
import pandas as pd
import KeyencePointCloud
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh

def Compare(STL_c, STL_ref):

    # Read in STL files
    mesh_c = mesh.Mesh.from_file(STL_c)
    mesh_ref = mesh.Mesh.from_file(STL_ref)
    mesh_ref.translate([19.5, 21.5, 0.0])

    # Stack points of reference mesh for distance calculation
    mesh_ref_stacked = np.vstack([mesh_ref.points[:, 0:3], mesh_ref.points[:, 3:6], mesh_ref.points[:, 6:9]])

    # Find vectors and norms of output stl
    vecs_c = mesh_c.vectors
    norms_c = mesh_c.normals

    # Set Z level for vector filtering
    Z_filter_level = 0.5

    # Initialize surface area and Hausdorff distance list
    SurfArea_c = 0
    dist_lst = []

    # Iterate through vectors of output stl
    for vec in range(len(vecs_c)):

        # Filter out vectors of output stl mesh with Z coordinates ~0
        if all(thing > Z_filter_level for thing in vecs_c[vec][:, 2]):

            filtered_vector = vecs_c[vec]

            # Calculate surface area of filtered vector
            SurfArea_c = SurfArea_c + 0.5*np.linalg.norm(norms_c[vec])

            # Calculate Hausdorff distance between each point of the filtered triangle of the output stl and the vectors of the reference stl
            for idx_c in range(len(filtered_vector)):

                dist_min = 1000.0
                for idx_ref in range(len(mesh_ref_stacked)):
                    dist = np.linalg.norm(filtered_vector[idx_c, :] - mesh_ref_stacked[idx_ref, :])
                    if dist_min > dist:
                        dist_min = dist
                dist_lst.append([dist_min, filtered_vector[idx_c, :], mesh_ref_stacked[idx_ref, :]])

    dist_lst = np.vstack(dist_lst)
    MaxHausdorffData = dist_lst[np.argmax(dist_lst[:, 0]), :]

    # Calculate surface area of reference STL
    SurfArea_ref = mesh_ref.areas.sum()

    # Calculate surface area difference
    SurfAreaDiff = np.abs(SurfArea_ref-SurfArea_c)

    # Calculate volume
    Volume_c, cog_c, inertia_c = mesh_c.get_mass_properties()
    Volume_ref, cog_ref, inertia_ref = mesh_ref.get_mass_properties()
    VolDiff = np.abs(Volume_ref-Volume_c)

    return mesh_c, mesh_ref, VolDiff, SurfAreaDiff, MaxHausdorffData
