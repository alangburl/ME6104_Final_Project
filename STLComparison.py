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

    # Calculate surface area of each STL
    SurfArea_c = mesh_c.areas.sum()
    SurfArea_ref = mesh_ref.areas.sum()
    SurfAreaDiff = np.abs(SurfArea_ref-SurfArea_c)

    Volume_c, cog_c, inertia_c = mesh_c.get_mass_properties()
    Volume_ref, cog_ref, inertia_ref = mesh_ref.get_mass_properties()
    VolDiff = np.abs(Volume_ref-Volume_c)

    # Stack points for distance calculation
    mesh_ref_stacked = np.vstack([mesh_ref.points[:, 0:3], mesh_ref.points[:, 3:6], mesh_ref.points[:, 6:9]])
    mesh_c_stacked = np.vstack([mesh_c.points[:, 0:3], mesh_c.points[:, 3:6], mesh_c.points[:, 6:9]])

    # Calculate Hausdorff distance between STLs for each vector
    dist_lst = []
    for idx_c in range(len(mesh_c_stacked)):
        dist_min = 1000.0
        for idx_ref in range(len(mesh_ref_stacked)):
            dist = np.linalg.norm(mesh_c_stacked[idx_c, :]-mesh_ref_stacked[idx_ref, :])
            if dist_min > dist:
                dist_min = dist
        dist_lst.append([dist_min, mesh_c_stacked[idx_c, :], mesh_ref_stacked[idx_ref, :]])

    dist_lst = np.vstack(dist_lst)
    MaxHausdorffData = dist_lst[np.argmax(dist_lst[:, 0]), :]

    return mesh_c, mesh_ref, VolDiff, SurfAreaDiff, MaxHausdorffData
