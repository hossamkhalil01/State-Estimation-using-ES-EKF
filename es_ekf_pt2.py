"""Implementation for Error state EKF for (IMU , LIDAR and GNSS) sensors fusion"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle,\
                    skew_symmetric, Quaternion
from numpy.linalg import inv

#### 1. Data ###################################################################

################################################################################
# load the data from the pickle files.
# For parts 1 and 2, p1_data.pkl is used
# while for part 3 pt3_data.pkl is used.
################################################################################
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################
"""
Each element of the data dictionary is stored as an item from the data
dictionary, which we will store in local variables, described by the following:

gt: Data object containing ground truth. with the following fields:
a: Acceleration of the vehicle, in the inertial frame
v: Velocity of the vehicle, in the inertial frame
p: Position of the vehicle, in the inertial frame
alpha: Rotational acceleration of the vehicle, in the inertial frame

w: Rotational velocity of the vehicle, in the inertial frame
r: Rotational position of the vehicle, in Euler angles in the inertial frame
_t: Timestamp in ms.
imu_f: StampedData object with the IMU specific force data (in vehicle frame).
data: The actual data.
t: Timestamps in ms.
imu_w: StampedData object with the IMU rotational velocity (in vehicle frame).
data: The actual data
t: Timestamps in ms.
gnss: StampedData object with the GNSS data.
data: The actual data
t: Timestamps in ms.
lidar: StampedData object with the LIDAR data (positions only).
data: The actual data
t: Timestamps in ms.

    """
################################################################################

gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']


################################################################################
# extrinsic calibration rotation matrix C_li and translation vector t_i_li for
# transforming lidar data into IMU frame.
################################################################################

# Incorrect calibration rotation matrix (for the second part of the project)
# not accurate clibration is provided so tuning is needed
# corresponding to Euler RPY angles (0.05, 0.05, 0.05).

C_li = np.array([
     [ 0.9975 , -0.04742,  0.05235],
     [ 0.04992,  0.99763, -0.04742],
     [-0.04998,  0.04992,  0.9975 ]
])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li


#### 2. Constants ##############################################################

################################################################################
# Set sensors variances values
################################################################################

#Old values (used in part 1 ) you can use it to see how it doesn't work Now.

# var_imu_f = 0.10
# var_imu_w = 0.25
# var_gnss  = 0.01
# var_lidar = 1.00

# Tuned values
var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 50

###############################################################################
# Set up some constants.
################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian


#### 3. Initial Values #########################################################

################################################################################
# Set up some initial values for ES-EKF.
################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates(quaternions)
p_cov = np.zeros([imu_f.data.shape[0], 9, 9]) # covariance matrices

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0


#### 4. Measurement Update #####################################################


def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    """ This function performs the measurement update step for the ES-EKF
    regardless of the sensor


    Argms:
    - sensor_var: the sensor variance value in this case is a constant value
    - p_cov_check : prediction step covariance matrix.
    - y_k : measurements vector same shape as x state vector
    - p_check, v_check, q_check: precition state vectors for position (3x1),
      velocity (3x1) and orientation (4x1) respectively.

    return:
    - p_hat: updated position state vector.
    - v_hat: updated velocity state vector.
    - q_hat: updated orientation state vector.
    - p_cov_hat:  updated covariance matrix
     """


    # 3.1 Compute Kalman Gain
    K_k = (p_cov_check.dot(h_jac.T)).dot(inv((h_jac.dot(p_cov_check)\
        .dot(h_jac.T))+(sensor_var*np.identity(3))))

    # 3.2 Compute error state
    delta_x = K_k.dot(y_k-p_check)

    # 3.3 Correct predicted state
    #a. position correction
    p_hat = p_check + delta_x[0:3]

    #b. velocity correction
    v_hat = v_check + delta_x[3:6]

    #c. Orientation correction
    q_hat= Quaternion(euler=delta_x[6:]).quat_mult_left(q_check,out='np')

    # 3.4 Compute corrected covariance
    p_cov_hat = p_cov_check - (K_k.dot(h_jac).dot(p_cov_check))

    return p_hat, v_hat, q_hat, p_cov_hat



#### 5. Main Filter Loop #######################################################

################################################################################
# start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################

# start at 1 b/c we have initial prediction from gt
for k in range(1, imu_f.data.shape[0]):
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs

    #the ortation matrix of the previous quaternion (3x3)
    C_ns =  Quaternion(*q_est[k-1]).to_mat()

    #a. Position
    p_check = p_est[k-1] + (delta_t*v_est[k-1]) +\
            ((delta_t**2/2)*((C_ns.dot(imu_f.data[k-1]))+g))

    #b. Velocity
    v_check = v_est[k-1] + (delta_t*((C_ns.dot(imu_f.data[k-1]))+g))

    #c. Orientation
    q_check = Quaternion(\
            euler =(imu_w.data[k-1]*delta_t)).quat_mult_right(q_est[k-1])

    # 1.1 Linearize the motion model and compute Jacobians
    #a. F_(k-1) Jacobian size of (9x9)
    F_k_1 = np.identity(9)
    F_k_1[0:3,3:6] = np.identity(3) * delta_t
    F_k_1[3:6,6:9] = -delta_t*skew_symmetric (C_ns.dot(imu_f.data[k-1]))

    #b. L_(k-1) Jacobian (9x6)
    L_k_1 = l_jac.reshape(9,6)

    #c. Q_(k-1) jacobian (6x6)
    Q_k = np.identity(6)
    Q_k[0][0] = var_imu_f**2
    Q_k[1][1] = var_imu_f**2
    Q_k[2][2] = var_imu_f**2
    Q_k[3][3] = var_imu_w**2
    Q_k[4][4] = var_imu_w**2
    Q_k[5][5] = var_imu_w**2

    Q_k = ((delta_t**2)*Q_k).reshape(6,6)


    # 2. Propagate uncertainty
    p_cov_check =((F_k_1.dot(p_cov[k-1]).dot(F_k_1.T))+\
                (L_k_1.dot(Q_k).dot(L_k_1.T))).reshape(9,9)


    # 3. Check availability of GNSS and LIDAR measurements

    #Searching for the matched GNSS data
    if (gnss_i < gnss.t.shape[0] and gnss.t[gnss_i] <= imu_f.t[k]):

        y_k = gnss.data[gnss_i].T
        p_check,v_check,q_check,p_cov_check = measurement_update(var_gnss,
                                            p_cov_check, y_k, p_check, v_check,
                                            q_check)
        gnss_i += 1

    #Searching for the matched LIDAR data
    if (lidar_i < lidar.t.shape[0] and lidar.t[lidar_i] <= imu_f.t[k]):

        y_k = lidar.data[lidar_i].T
        p_check,v_check,q_check,p_cov_check = measurement_update(var_lidar,
                                            p_cov_check, y_k, p_check, v_check,
                                            q_check)
        lidar_i += 1

    # Update states (save)
    p_est[k] = p_check
    v_est[k] = v_check
    q_est[k] = q_check
    p_cov[k] = p_cov_check


#### 6. Results and Analysis ###################################################

################################################################################
#  plot the results. This plot
# Notice that the estimated trajectory continues past the ground truth.
# This is because this part of the trajectory's ground truth is not provided
################################################################################

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, 4, 8, 12, 16])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()


################################################################################
# Plot the error for each of the 6 DOF, with estimates for our uncertainty
# The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty.
################################################################################

error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()


#### 7. Submission #############################################################

################################################################################
# This section generates sumbission file for Coursera's platform.
################################################################################


# Pt. 2 submission
# p2_indices = [9000, 9400, 9800, 10200, 10600]
# p2_str = ''
# for val in p2_indices:
#     for i in range(3):
#         p2_str += '%.3f ' % (p_est[val, i])
# with open('pt2_submission.txt', 'w') as file:
#     file.write(p2_str)
