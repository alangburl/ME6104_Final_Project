import numpy as np
import pandas as pd
import KeyencePointCloud
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh

from NURBS_Calculations import NURBS
from Surface_Parser import Surface_Parser as SP
# Define file/filepath of point cloud data in CSV form
import STLComparison

CSV_1 = 'Z3_48.csv'
CSV_2 = 'Z3_71.csv'

# Define Z offset between scans
Z_offset_mm = 0.23*25.4

# Define resolution multiplier. This has the effect of reducing the data resolution by a factor of n
n_res = 32

# Define resolution to use from scan data. Lower resolution --> less accuracy but less processing time
Res_mm = 0.0125*n_res

# Define number of points for linear interpolation between scans
n_interpolate = 4

# Define tolerance (mm) to use for parsing data for interpolation between scans
tol_interpolate = 2

# Define search distance (mm) to use for zero plane calculation. This search distance is used to construct a square
# around each corner of side length d_zero_plane and all points within this square will be used for zero plane calculations.
d_zero_plane = 2.5  

# Set plot toggle ("on"/"off"). Note that plotting while using a low n_res value will not work since it is too many points for
# matplotlib to put on the scatter plot.
# plot_toggle = "on"
plot_toggle = "off"

# Convert CSV files to 3 x n arrays
PointCloud1 = KeyencePointCloud.CSV_Array_Convert(CSV_1, Res_mm, n_res)
print("Number of Point Cloud 1 pts: ", np.shape(PointCloud1)[0])

# Plot point cloud data
if plot_toggle == "on":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PointCloud1[:, 0], PointCloud1[:, 1], PointCloud1[:, 2], marker='.')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Point Cloud 1, Z: 0.0")

PointCloud2 = KeyencePointCloud.CSV_Array_Convert(CSV_2, Res_mm, n_res)
print("Number of Point Cloud 2 pts: ", np.shape(PointCloud2)[0])

# Add in Z offset to second point cloud Z data
PointCloud2[:, 2] = PointCloud2[:, 2]+Z_offset_mm

# Plot point cloud data
if plot_toggle == "on":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PointCloud2[:, 0], PointCloud2[:, 1], PointCloud2[:, 2], marker='.')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Point Cloud 2, Z: 5.8")
# Combine scans to form one point cloud that is aligned to point cloud.
RawStackedPointCloud, CombinedPointCloud, ZeroPlaneXYCorners, ZeroPlanePoints, GlobalZeroPlane, AlignedCombined_PC = KeyencePointCloud.CombinePC([PointCloud1, PointCloud2], n_interpolate, tol_interpolate, d_zero_plane, plot_toggle)

#parse the data and realign it to make a nice shaped array
parser=SP(AlignedCombined_PC)
_,_,_=parser.convert_2d(11.0)#set the hieght point to the trigger
#currently only just sets the closest do to sparcity of the data
new_points,weights=parser.interpolation2d() 
#number of evaluation points
numu,numw=50,50
nurbs=NURBS(new_points,numu,numw,3,3)
#get the tensor product
output=nurbs.nurbs_calc(weights)
#save the stl
nurbs.save_stl('output.stl')
#plot the final nurbs surface
fig,ax=nurbs.plot_surfaces('nurbs',1,'NURBS surface')
fig.savefig('NURBS_Fit.png',dpi=600)

# Plot raw combined point cloud data
if plot_toggle == "on":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(RawStackedPointCloud[:, 0], RawStackedPointCloud[:, 1], RawStackedPointCloud[:, 2], marker='.')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Raw Combined Point Cloud")

    # Use this view to compare to interpolated combined point cloud
    ax.view_init(elev=0., azim=0)

# Plot linearly interpolated combined point cloud data
if plot_toggle == "on":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(CombinedPointCloud[:, 0], CombinedPointCloud[:, 1], CombinedPointCloud[:, 2], marker='.')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Linearly Interpolated Combined Point Cloud")

    # Use this view to compare to raw combined point cloud
    ax.view_init(elev=0., azim=0)

# Plot linearly interpolated combined point cloud data with corner zero plan data outlined
if plot_toggle == "on":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(CombinedPointCloud[:, 0], CombinedPointCloud[:, 1], CombinedPointCloud[:, 2], marker='.')
    ax.scatter(ZeroPlaneXYCorners[:, 0], ZeroPlaneXYCorners[:, 1], [1, 1, 1, 1], color='red', s=20)
    ax.scatter(ZeroPlanePoints[:, 0], ZeroPlanePoints[:, 1],  ZeroPlanePoints[:, 2], color='green', s=20)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Linearly Interpolated Combined Point Cloud with Corner Zero Plane Data")

#     # Use this view to show zero plane data outlines
#     ax.view_init(elev=90., azim=-90)

# # Plot linearly interpolated combined point cloud data with global zero plane
if plot_toggle == "on":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(CombinedPointCloud[:, 0], CombinedPointCloud[:, 1], CombinedPointCloud[:, 2], marker='.')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]), np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = GlobalZeroPlane[0] * X[r,c] + GlobalZeroPlane[1] * Y[r,c] + GlobalZeroPlane[2]
    ax.plot_wireframe(X,Y,Z, color='k')

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Linearly Interpolated Combined Point Cloud with Global Zero Plane")

# # Plot fully aligned combined point cloud data
if plot_toggle == "on":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(AlignedCombined_PC[:, 0], 
                AlignedCombined_PC[:, 1], 
                AlignedCombined_PC[:, 2], marker='.')

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Fully Aligned Point Cloud")

    # Use this view to compare to raw combined point cloud
    # ax.view_init(elev=0., azim=0)

# STL Comparison between candidate and reference STL
# mesh_c, mesh_ref, VolDiff, SurfAreaDiff, MaxHausdorffData = STLComparison.Compare('FinalProjectPart_short.stl', 'FinalProjectPart_shorter.stl')

# Plot STL files
# figure = plt.figure()
# axes = mplot3d.Axes3D(figure)
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh_c.vectors, alpha=0.1, linewidths=1, facecolors='b'))
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh_ref.vectors, alpha=0.1, linewidths=1, facecolors='g'))
# axes.plot3D([MaxHausdorffData[1][0], MaxHausdorffData[2][0]], [MaxHausdorffData[1][1], MaxHausdorffData[2][1]], [MaxHausdorffData[1][2], MaxHausdorffData[2][2]], "red")
# scale = mesh_c.points.flatten()
# axes.auto_scale_xyz(scale, scale, scale)
# figure.suptitle("STL Plot")
# axes.set_xlabel("X")
# axes.set_ylabel("Y")
# axes.set_zlabel("Z")

# print("STL Comparison Data")
# print("Volume Difference (mm3): ", VolDiff)
# print("Surface Area Difference (mm2): ", SurfAreaDiff)
# print("Maximum Hausdorff Distance (mm): ", MaxHausdorffData[0])

plt.show()