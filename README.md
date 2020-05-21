Part1 :implementing a state estimator using the IMU as an input and (GNSS + LIDAR) as a correction measurements

Part2 :neglecting the orientation between the GNSS frame and the LIDAR frame leads to errors in the estimation
(wrong rotation matrix) --> (Cov matrix needs to be tuned)

Part3 : Sensor dropout effect ( for a short time) (leads to drift)