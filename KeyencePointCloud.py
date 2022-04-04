import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Point Cloud is abbreviated to PC for brevity.

# Convert Keyence scanner CSV file to 3 x n array
def CSV_Array_Convert(filename, Res_mm, n):

    # Read in CSV file as pandas dataframe
    dataframe = pd.read_csv(filename, delimiter=',', header=None)
    dataset = dataframe.values

    # Convert dataset dimensions to range of X and Y coordinate values (mm)
    X_Range = np.linspace(0, 3200*0.0125, num=int(3200/n), endpoint=False)
    Y_Range = np.linspace(0, np.shape(dataset)[0]*0.0125, num=int(np.shape(dataset)[0]/n), endpoint=False)

    # Initialize array for n x 3 point cloud data array
    PCArray = []

    # Convert point cloud Z data in Z array and map to X and Y data
    for i in range(len(X_Range)):
        for j in range(len(Y_Range)):

            # If Z value is out of range don't use it
            if dataset[j*n, i*n] != -99999.9999:
                PCArray.append([X_Range[i], Y_Range[j], dataset[j*n, i*n]])

    return np.array(PCArray)

# Combine multiple point clouds and align
def CombinePC(PC_Data, n_interpolate, tol_interpolate, d_zero_plane, plot_toggle):

    # Get raw stacked point cloud data for use in graphical troubleshooting
    RawStacked_PC = np.vstack(PC_Data)

    ###################################################################################
    # Linearly interpolate gaps between highest Z values of first scan and lowest Z values of second scan

    # Parse out data from each PC for interpolatation
    HighZ_Scan1 = PC_Data[0][(PC_Data[0][:, 2] <= max(PC_Data[0][:, 2])) & (PC_Data[0][:, 2] >= max(PC_Data[0][:, 2])-tol_interpolate)]
    LowZ_Scan2 = PC_Data[1][(PC_Data[1][:, 2] >= min(PC_Data[1][:, 2])) & (PC_Data[1][:, 2] <= min(PC_Data[1][:, 2])+tol_interpolate)]

    InterpolatedCloud = []
    # Iterate through each point from HighZ_Scan1 and LowZ_Scan2.
    for point1 in HighZ_Scan1:

        # Find distances between each point pair
        Dists = np.sqrt(np.sum((point1-LowZ_Scan2)**2, axis=1))

        # Find index of minimum distance
        Min_Ind = np.argmin(Dists)

        # Then linearly interpolate n_interpolate points between them and store as additional array
        InterpX = np.linspace(point1[0], LowZ_Scan2[Min_Ind, :][0], num=n_interpolate+1, endpoint=False)
        InterpY = np.linspace(point1[1], LowZ_Scan2[Min_Ind, :][1], num=n_interpolate+1, endpoint=False)
        InterpZ = np.linspace(point1[2], LowZ_Scan2[Min_Ind, :][2], num=n_interpolate+1, endpoint=False)

        # Add interpolated points to InterpolatedCloud. Make sure to skip first point since it is an already existing point,
        for i in range(n_interpolate):

            InterpolatedCloud.append([InterpX[i+1], InterpY[i+1], InterpZ[i+1]])

    # Print number of interpolated points
    print("Number of Interpolated Points: ", len(InterpolatedCloud))

    # Add InterpolatedCloud to PC_Data
    PC_Data.append(InterpolatedCloud)
    ######################################################################################

    #####################################################################################
    # Join individual point cloud arrays; for XY points with multiple Z values, remove higher Z value as it
    # is likely to be an error.

    # Stack individual point cloud arrays
    Stacked_PC = np.vstack(PC_Data)

    # Determine if there is more than one Z value for each XY value.
    unq, count = np.unique(Stacked_PC[:, [0, 1]], axis=0, return_counts=True)
    RepeatedRows = unq[count > 1]

    # Iterate through repeated rows and delete the row that has the higher z value.
    for row in RepeatedRows:
        repeated_idx = np.argwhere(np.all(Stacked_PC[:, [0, 1]] == row, axis=1))
        RepeatIndex = repeated_idx.ravel()

        # Compare Z values of repeated rows and set the higher one to -1
        if (Stacked_PC[RepeatIndex[0], 2] > Stacked_PC[RepeatIndex[1], 2]):
            Stacked_PC[RepeatIndex[0], 2] = -1
        else:
            Stacked_PC[RepeatIndex[1], 2] = -1

    # Print number of repeated points
    Num_Repeat = len(np.where(Stacked_PC[:, 2] == -1)[0])
    print("Number of Removed Repeated Points: ", Num_Repeat)

    # Delete all rows with Z = -1 (higher value from repeated rows)
    InterpolatedCombined_PC = np.delete(Stacked_PC, np.where(Stacked_PC[:, 2] == -1)[0], axis=0)

    # Print final number of combined point cloud points
    print("Number of Combined Point Clouds pts: ", len(InterpolatedCombined_PC))
    ###############################################################################################

    ##############################################################################################
    # Align the combined and interpolated point cloud by using zero plane estimates from corners

    # Determine points to use for zero plane fitting. Use points from each corner of combined and interpolated
    # point cloud since that data is available.
    Min_Y_Data = InterpolatedCombined_PC[InterpolatedCombined_PC[:, 1] == min(InterpolatedCombined_PC[:, 1])]
    Max_Y_Data = InterpolatedCombined_PC[InterpolatedCombined_PC[:, 1] == max(InterpolatedCombined_PC[:, 1])]
    ZeroPlaneXYCorners = np.vstack([Min_Y_Data[Min_Y_Data[:, 0] == min(Min_Y_Data[:, 0])],
                             Min_Y_Data[Min_Y_Data[:, 0] == max(Min_Y_Data[:, 0])],
                             Max_Y_Data[Max_Y_Data[:, 0] == min(Max_Y_Data[:, 0])],
                             Max_Y_Data[Max_Y_Data[:, 0] == max(Max_Y_Data[:, 0])]])

    # Determine the rest of the zero plane points and fit plane to them
    ZeroPlanePoints = []
    CornerPlaneData = []

    for corner in ZeroPlaneXYCorners:

        if (corner[1] == min(InterpolatedCombined_PC[:, 1])) & (corner[0]<20):
            # Determine points within square of side d_zero_plane
            CornerData = InterpolatedCombined_PC[(InterpolatedCombined_PC[:, 0] <= corner[0]+d_zero_plane) &
                                                 (InterpolatedCombined_PC[:, 0] >= corner[0]) &
                                                 (InterpolatedCombined_PC[:, 1] <= corner[1]+d_zero_plane) &
                                                 (InterpolatedCombined_PC[:, 1] >= corner[1])]
            ZeroPlanePoints.append(CornerData)

            # Fit plane to corner data
            fit, residual = ZeroPlaneFit(CornerData, corner, plot_toggle)

            # Save fit plane data
            CornerPlaneData.append(fit)

        if (corner[1] == min(InterpolatedCombined_PC[:, 1])) & (corner[0]>20):
            CornerData = InterpolatedCombined_PC[(InterpolatedCombined_PC[:, 0] >= corner[0]-d_zero_plane) &
                                                 (InterpolatedCombined_PC[:, 0] <= corner[0]) &
                                                 (InterpolatedCombined_PC[:, 1] <= corner[1]+d_zero_plane) &
                                                 (InterpolatedCombined_PC[:, 1] >= corner[1])]
            ZeroPlanePoints.append(CornerData)

            # Fit plane to corner data
            fit, residual = ZeroPlaneFit(CornerData, corner, plot_toggle)

            # Save fit plane data
            CornerPlaneData.append(fit)

        if (corner[1] == max(InterpolatedCombined_PC[:, 1])) & (corner[0]<20):
            CornerData = InterpolatedCombined_PC[(InterpolatedCombined_PC[:, 0] <= corner[0]+d_zero_plane) &
                                                 (InterpolatedCombined_PC[:, 0] >= corner[0]) &
                                                 (InterpolatedCombined_PC[:, 1] >= corner[1]-d_zero_plane) &
                                                 (InterpolatedCombined_PC[:, 1] <= corner[1])]
            ZeroPlanePoints.append(CornerData)

            # Fit plane to corner data
            fit, residual = ZeroPlaneFit(CornerData, corner, plot_toggle)

            # Save fit plane data
            CornerPlaneData.append(fit)

        if (corner[1] == max(InterpolatedCombined_PC[:, 1])) & (corner[0]>20):
            CornerData = InterpolatedCombined_PC[(InterpolatedCombined_PC[:, 0] >= corner[0]-d_zero_plane) &
                                                 (InterpolatedCombined_PC[:, 0] <= corner[0]) &
                                                 (InterpolatedCombined_PC[:, 1] >= corner[1]-d_zero_plane) &
                                                 (InterpolatedCombined_PC[:, 1] <= corner[1])]
            ZeroPlanePoints.append(CornerData)
            ZeroPlanePoints = np.vstack(ZeroPlanePoints)

            # Fit plane to corner data
            fit, residual = ZeroPlaneFit(CornerData, corner, plot_toggle)

            # Save fit plane data
            CornerPlaneData.append(fit)

    # Print total number of zero plane points
    print("Number of zero plane points: ", len(ZeroPlanePoints))

    # Calculate global zero plane coefficients by averaging results from corner plane fits
    CornerPlaneData = np.vstack(CornerPlaneData)
    GlobalZeroPlane = np.array([np.average(CornerPlaneData[:, 0]),
                       np.average(CornerPlaneData[:, 1]),
                       np.average(CornerPlaneData[:, 2])])

    # Calculate global zero plane normal vector and X, Y, Z rotation angles. In ZeroPlaneFit function, the plane equation
    # is given as Ax + By + C = z. Thus the normal vector is <A, B, 1>
    GlobalNorm = [GlobalZeroPlane[0]/GlobalZeroPlane[2], GlobalZeroPlane[1]/GlobalZeroPlane[2], 1/GlobalZeroPlane[2]]
    GlobalNorm = GlobalNorm/np.linalg.norm(GlobalNorm)
    GlobalAngleX = (np.pi/2) - np.arccos(GlobalNorm[0])
    GlobalAngleY = (np.pi/2) - np.arccos(GlobalNorm[1])
    GlobalAngleZ = np.arccos(GlobalNorm[2])

    # Print zero plane data
    print("Zero Plane Norm: ", GlobalNorm)
    print("X Axis Rotation: ", GlobalAngleX)
    print("Y Axis Rotation: ", GlobalAngleY)
    print("Z Axis Rotation: ", GlobalAngleZ)

    # Calculate full rotation matrix. Assume rotations follow order of X, Y, Z axes (ie X' = Rz*Ry*Rx*X)
    R_x = np.array([1, 0, 0, 0, np.cos(-GlobalAngleX), -np.sin(-GlobalAngleX), 0, np.sin(-GlobalAngleX), np.cos(-GlobalAngleX)]).reshape((3, 3))
    print("R_x: \n", R_x)
    R_y = np.array([np.cos(-GlobalAngleY), 0, np.sin(-GlobalAngleY), 0, 1, 0, -np.sin(-GlobalAngleY), 0, np.cos(-GlobalAngleY)]).reshape((3, 3))
    print("R_y: \n", R_y)
    R_z = np.array([np.cos(-GlobalAngleZ), -np.sin(-GlobalAngleZ), 0, np.sin(-GlobalAngleZ), np.cos(-GlobalAngleZ), 0, 0, 0, 1]).reshape((3, 3))
    print("R_z: \n", R_z)

    R_full = np.matmul(R_x, R_y)
    R_full = np.matmul(R_z, R_full)

    # Apply rotation angle to InterpolatedCombined_PC
    RotatedCombined_PC = np.matmul(InterpolatedCombined_PC, R_full)

    # Transform InterpolatedCombined_PC down by average Z value of ZeroPlanePoints
    Z_avg = np.average(ZeroPlanePoints[:, 2])

    AlignedCombined_PC = np.column_stack((RotatedCombined_PC[:, 0], RotatedCombined_PC[:, 1], RotatedCombined_PC[:, 2]-Z_avg))
    ###############################################################################################

    return RawStacked_PC, InterpolatedCombined_PC, ZeroPlaneXYCorners, ZeroPlanePoints, GlobalZeroPlane, AlignedCombined_PC

# Fit plane to data
def ZeroPlaneFit(FitData, corner, plot_toggle):

    # Fit plane to corner data using least squares regression
    tmp_A = []
    tmp_b = []
    for i in range(len(FitData)):
        tmp_A.append([FitData[i, 0], FitData[i, 1], 1])
        tmp_b.append(FitData[i, 2])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    print("Fitted Plane for XY Corner: ({}, {})".format(corner[0], corner[1]))
    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("residual:", residual)

    # Plot fitted plane
    if plot_toggle == "on":
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(FitData[:, 0], FitData[:, 1], FitData[:, 2], color='b')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                          np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
        ax.plot_wireframe(X,Y,Z, color='k')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title("Corner Zero Plane Fit:  ({}, {})".format(corner[0], corner[1]))

    return np.array(fit).flatten(), residual
