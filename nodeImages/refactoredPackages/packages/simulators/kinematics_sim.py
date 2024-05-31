import numpy as np

def get_next_pose(icc_pos, d, cur_theta, theta_displacement):
    """
    Compute the new next position in global frame
    Input:
        - icc_pos: numpy array of ICC position [x,y] in global frame
        - d: distance from robot to the center of curvature
        - cur_theta: current yaw angle in radian (float)
        - theta_displacement: the amount of angular displacement if we apply w for 1 time step
    Return:
        - next_position:
        - next_orientation:
    """
    
    # First, let's define the ICC frame as the frame centered at the location of ICC
    # and oriented such that its x-axis points towards the robot
    
    # Compute location of the point where the robot should be at (i.e., q)
    # in the frame of ICC.
    x_new_icc_frame = d * np.cos(theta_displacement)
    y_new_icc_frame = d * np.sin(theta_displacement)
    
    # Build transformation matrix from origin to ICC
    T_oc_angle = -(np.deg2rad(90) - cur_theta) # 
    icc_x, icc_y = icc_pos[0], icc_pos[1]
    T_oc = np.array([
        [np.cos(T_oc_angle), -np.sin(T_oc_angle), icc_x],
        [np.sin(T_oc_angle), np.cos(T_oc_angle), icc_y],
        [0, 0, 1]
    ]) # Transformation matrix from origin to the ICC
    
    # Build transformation matrix from ICC to the point where the robot should be at (i.e., q)
    T_cq = np.array([
        [1, 0, x_new_icc_frame],
        [0, 1, y_new_icc_frame],
        [0, 0, 1]
    ]) # Transformation matrix from ICC to the point where the robot should be at (i.e., q)
    
    # Convert the local point q to the global frame
    T_oq = np.dot(T_oc, T_cq) # Transformation matrix from origin to q
    
    next_position = np.array([T_oq[0,2], T_oq[1,2]])
    next_orientation = np.degrees(cur_theta) + np.degrees(theta_displacement)
    return next_position, next_orientation

def drive(cur_pos, cur_angle, velocity, angular_velocity, wheel_dist, wheel_radius, dt):
    """
    Input:
        - cur_pos: numpy array of current position [x,y] in global frame
        - cur_angle: current yaw angle in degree (float)
        - velocity: linear velocity in m/sec (float)
        - angular_velocity: angular velocity in rad/sec (float)
        - wheel_dist: distance between left and right wheels in meters (i.e., 2L) (float)
        - wheel_radius: radius of the wheels in meters (i.e., R) (float)
        - dt: time step (float)
    Return:
        - next_position: numpy array of next position [x,y] in global frame
        - next_orientation: next yaw angle ()
    """
    
    # Convert angle to radian and rename some variables
    cur_theta = np.deg2rad(cur_angle)
    l = wheel_dist
    v = velocity
    w = angular_velocity

    # If angular velocity is zero, then there is no rotation
    if w == 0:
        new_x = cur_pos[0] + dt * v * np.cos(cur_theta)
        new_y = cur_pos[1] + dt * v * np.sin(cur_theta)
        cur_pos = np.array([new_x, new_y])
        cur_angle = cur_angle # does not change since we are moving straight
        return cur_pos, cur_angle
    
    # Compute the distance from robot to the center of curvature (i.e., d)
    d = v / w
    
    # Compute the amount of angular displacement if we apply w for 1 time step
    theta_displacement = w * dt 

    # Compute location of ICC in global frame
    icc_x = cur_pos[0] - d * (np.sin(cur_theta)) 
    icc_y = cur_pos[1] + d * (np.cos(cur_theta))
    icc_pos = np.array([icc_x, icc_y])
    
    # Compute next position and orientation given cx, cy, d, cur_theta, and theta_displacement
    next_position, next_orientation = get_next_pose(icc_pos, d, cur_theta, theta_displacement)
    
    return next_position, next_orientation

def integrate_kinematics(initial_pose:list, initial_vel:list, y_ref:float, controller) -> tuple:
    """
    Integrate kinematics starting from initial pose and vel and targeting y_ref
    for 60 seconds, with a timestep of 0.1s
    Input:
        - inital_pose: 3 elements list containing the initial position and orientation 
                        [x_0,y_0,theta_0], theta_0 in degrees
        - initial_vel: 2 elements list with initial linear and angular vel [v_0, omega_0]
        - y_ref: the target y position
    
    Return:
        - xs: list of x coordinates
        - ys: list of y coordinates
        - omegas: list of angular velocities
        - es: list of errors (y-y_ref)
        - angles: list of yaw angles
    """
    cur_pos = np.array(initial_pose[0:2])   # initial position of the robot
    cur_angle = initial_pose[2]             # initial yaw angle of the robot
    wheel_dist = 0.1                        # distance between left and right wheels in meters, i.e., 2L
    wheel_radius = 0.0318                   # radius of the wheels in meters, i.e., R

    # Define time variables
    initial_time = 0.0
    timestep = 0.1                                # time step in seconds
    t_max = 60
    n = int(t_max / timestep)

    v_0 = initial_vel[0] # assume velocity is constant in m/s

    # Create some lists to store stuff
    xs = [cur_pos[0]]
    ys = [cur_pos[1]]
    angles = [cur_angle]
    
    omegas = []                       
    es = []                             
    
    # Set integrator state
    e_int = 0
    e = initial_pose[1]-y_ref

    # Set the commanded parameters
    v_0 = 0.22
    prev_e_y = 0.0
    prev_int_y = 0.0

    for i in range(n):
        y_hat = cur_pos[1]

        v_0, omega, e, e_int = controller(v_0, y_ref, y_hat, prev_e_y, prev_int_y, delta_t=timestep)
        prev_e_y = e
        prev_int_y = e_int

        # simulate driving
        cur_pos, cur_angle = drive(cur_pos, cur_angle, v_0, omega, wheel_dist, wheel_radius, timestep) 
        
        # store trajectory, angular velocity, and error so we can plot them
        xs.append(cur_pos[0])
        ys.append(cur_pos[1])
        omegas.append(omega)
        es.append(e)
        angles.append(cur_angle)

    return xs,ys,omegas,es,angles