import numpy as np


# Trajectory functions

# Trajectory 1: Flat oval
def elliptical_trajectory(t):
    A_x = 50  # Amplitude in the x direction
    A_y = 20  # Amplitude in the y direction, different from x for an oval shape
    omega = 2 * np.pi  # Frequency of oscillation for one full loop
    phi = np.pi / 4  # Phase shift
    
    # Normalize t from [1, 10] to [0, 1]
    normalized_t = (t - 1) / 9  # t-1 shifts the start from 1 to 0, and dividing by 9 scales the range from 0-9 to 0-1
    
    # Equations for the trajectory using the normalized time
    x = A_x * np.sin(omega * normalized_t + phi)
    y = A_y * np.cos(omega * normalized_t + phi)
    z = 30  # Constant elevation
    
    return x, y, z

# Trajectory 2: Bent oval
def bent_elliptical_trajectory(t):
    A = 30  # Amplitude
    omega = 2 * np.pi  # Frequency of oscillation for one full loop
    phi = np.pi / 4  # Phase shift
    z_offset = 33  # Offset in the z direction
    
    # Normalize t from [1, 10] to [0, 1]
    normalized_t = (t - 1) / 9  # t-1 shifts the start from 1 to 0, dividing by 9 scales the range from 0-9 to 0-1
    
    # Equations for the trajectory using normalized time
    x = A * np.sin(omega * normalized_t + phi)
    y = A * np.cos(omega * normalized_t + phi)
    z = A * np.cos(omega * normalized_t) + z_offset  # Cosine oscillation with a constant offset to elevate the entire pattern
    
    return x, y, z

# Trajectory 4: Rotated Heart
def heart_trajectory_rotated(t, angle_degrees=30):
    # Calculate the rotation angle in radians
    angle_radians = np.radians(angle_degrees)
    
    # Original heart trajectory coordinates
    x_original = 16 * np.sin(t)**3
    y_original = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    z_original = 30  # Originally constant elevation
    
    # Apply rotation around the Y-axis
    x_rotated = x_original * np.cos(angle_radians) + z_original * np.sin(angle_radians)
    y_rotated = y_original  # Y remains the same as the rotation is around the Y-axis
    z_rotated = -x_original * np.sin(angle_radians) + z_original * np.cos(angle_radians)
    
    return x_rotated, y_rotated, z_rotated

# Trajectory 5: Square
def square_trajectory(t):
    # Define parameters for the trajectory
    L = 50        # Side length of the square
    period = 10    # Total time to complete one square
    
    # Normalize t to a modulo period for periodicity in the trajectory
    t_normalized = t % period
    
    # Define piecewise functions for x and y
    if t_normalized <= period / 4:
        x = L * t_normalized * 4 / period
        y = 0
    elif t_normalized <= period / 2:
        x = L
        y = L * (t_normalized - period / 4) * 4 / period
    elif t_normalized <= 3 * period / 4:
        x = L - L * (t_normalized - period / 2) * 4 / period
        y = L
    else:
        x = 0
        y = L - L * (t_normalized - 3 * period / 4) * 4 / period

    # Constant value for the trajectory's z-coordinate
    z = 20  # Fixed elevation at 30 units
    return x, y, z


# Trajectory 6: Rotated Square
def square_trajectory_rotated(t, angle_degrees=30):
    # Define parameters for the trajectory
    L = 50        # Side length of the square
    period = 10    # Total time to complete one square
    z_const = 20  # Fixed elevation
    z_offset = 20

    # Calculate the rotation angle in radians
    angle_radians = np.radians(angle_degrees)

    # Normalize t to a modulo period for periodicity in the trajectory
    t_normalized = t % period
    
    # Define piecewise functions for x and y
    if t_normalized <= period / 4:
        x = L * t_normalized * 4 / period
        y = 0
    elif t_normalized <= period / 2:
        x = L
        y = L * (t_normalized - period / 4) * 4 / period
    elif t_normalized <= 3 * period / 4:
        x = L - L * (t_normalized - period / 2) * 4 / period
        y = L
    else:
        x = 0
        y = L - L * (t_normalized - 3 * period / 4) * 4 / period

    # Original constant z-value before rotation
    z_original = z_const
    
    # Apply rotation around the Y-axis
    x_rotated = x * np.cos(angle_radians) + z_original * np.sin(angle_radians)
    y_rotated = y  # Y remains unchanged as rotation is around Y-axis
    z_rotated = -x * np.sin(angle_radians) + z_original * np.cos(angle_radians)

    return x_rotated, y_rotated, z_rotated + z_offset

# Trajectory 7: Archimedean Spiral
def archimedean_spiral_flat(t, a=0, b=3, z_const=10, turns=3):
    # The total angle for the desired number of turns
    total_angle = 2 * np.pi * turns
    # Radius increases linearly with t
    r = a + b * t
    
    # Parametric equations for the spiral
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    # Constant elevation
    z = z_const
    
    return x, y, z

# Trajectory 8: Rotated Archimedean Spiral
def archimedean_spiral_flat_rotated(t, a=0, b=3, z_const=33, turns=3, angle_degrees=33):
    # The total angle for the desired number of turns
    total_angle = 2 * np.pi * turns
    # Calculate the rotation angle in radians
    angle_radians = np.radians(angle_degrees)
    # Radius increases linearly with t
    r = a + b * t
    
    # Original spiral equations in the plane
    x_original = r * np.cos(t)
    y_original = r * np.sin(t)
    z_original = z_const  # Constant elevation before rotation
    
    # Apply rotation around the Y-axis
    x_rotated = x_original * np.cos(angle_radians) - z_original * np.sin(angle_radians)
    y_rotated = y_original  # Y remains the same as the rotation is around the Y-axis
    z_rotated = x_original * np.sin(angle_radians) + z_original * np.cos(angle_radians)
    
    return x_rotated, y_rotated, z_rotated


# Trajectory 9: Staright Line
def straight_line_across_origin(t, length=333):
    # Normalize t from [1, 10] to [0, 1]
    normalized_t = (t - 1) / 10
    
    # Calculate the start and end points based on the length
    start_x = -(length / 2)
    end_x = length / 2
    
    # Linear interpolation between start_x and end_x as normalized_t goes from 0 to 1
    x = start_x + (end_x - start_x) * normalized_t
    y = 0
    z = 30  # Constant elevation
    
    return x, y, z

# Trajectory 10: Rotated Straight Line
def straight_line_across_origin_rotated_and_lifted(t, length=333, angle_degrees=33, z_lift=20):
    # Normalize t from [1, 10] to [0, 1]
    normalized_t = (t - 1) / 10  # Correct normalization for mapping [1, 10] to [0, 1]

    # Calculate the start and end points based on the length
    start_x = -(length / 2)
    end_x = length / 2
    
    # Linear interpolation between start_x and end_x as normalized_t goes from 0 to 1
    x = start_x + (end_x - start_x) * normalized_t
    y = 0
    original_z = 30  # Initial constant elevation

    # Convert angle in degrees to radians for clockwise rotation
    angle_radians = np.radians(-angle_degrees)  # Negative for clockwise rotation

    # Apply rotation around the Y-axis
    x_rotated = x * np.cos(angle_radians) + original_z * np.sin(angle_radians)
    y_rotated = y  # y remains unchanged as rotation is around the Y-axis
    z_rotated = -x * np.sin(angle_radians) + original_z * np.cos(angle_radians)

    # Apply translation in the z direction
    z_rotated += z_lift
    
    return x_rotated, y_rotated, z_rotated

# Trajectory 11: Wavy circular loop
def wavy_circular_loop_trajectory(t):
    period = 10  # Total time to complete one loop
    omega = 2 * np.pi / period  # Adjust omega to complete the loop in 10 units of time

    A = 60  # Circle radius
    B = 10  # Wavy amplitude along z-axis

    # Normalize time t starting from t=1 to t=10
    t_normalized = (t - 1) % period

    # Equations for the trajectory
    x = A * np.cos(omega * t_normalized)
    y = A * np.sin(omega * t_normalized)
    z = B * np.sin(4 * omega * t_normalized)  # 4 waves along the loop
    
    return x, y, z

# Trajectory 12: Clover
def clover_trajectory(t):
    omega = 1.0  # Frequency of oscillation
    A = 50     # Amplitude
    x = A * np.sin(omega * t) * np.cos(t)
    y = A * np.sin(omega * t) * np.sin(t) -25
    z = A/5 * np.cos(omega * t) + 30

    return x, y, z

def figure_of_eight_trajectory(t):
    # Normalize time from [1, 10] to [0, 1]
    normalized_time = (t - 1) / 9

    # Define figure-of-eight trajectory equations
    omega = 2 * np.pi  # Frequency of oscillation for one full loop
    a = 50  # Amplitude
    # Update time to normalized_time in equations and adjust frequency omega
    x = (a * np.cos(omega * normalized_time)) / (1 + np.sin(omega * normalized_time)**2)
    y = (a * np.sin(omega * normalized_time) * np.cos(omega * normalized_time)) / (1 + np.sin(omega * normalized_time)**2)
    z = 8 * np.ones_like(x)  # Constant elevation

    return x, y, z
def star_trajectory(t):
    # Define star trajectory equations
    omega = 1.0  # Frequency of oscillation
    A = 50.0 
    center_x, center_y = 0, 0
    radius = A     # Amplitude
    x = center_x + radius * np.cos(omega * t)
    y = center_y + radius * np.sin(omega * t)
    z = 8*np.ones_like(x)
    return x, y, z


def clover_trajectory2(t):
    # Define clover trajectory equations
    omega = 1.0  # Frequency of oscillation
    A = 50     # Amplitude
    x = A * np.sin(omega * t) * np.sin(t)*np.sin(t)
    y = A * np.sin(omega * t) * np.cos(t)*np.sin(t)
    z = A/5 * np.cos(omega * t) + 30
    return x, y, z


def clover_trajectory3(t):
    # Define clover trajectory equations
    omega = 1.0  # Frequency of oscillation
    A = 50     # Amplitude
    x = 3*A/2 * np.sin(omega * t) * np.sin(t)*np.cos(t)
    y = A * np.sin(omega * t) * np.cos(t)*np.cos(t)
    z = A/5 * np.cos(omega * t)*np.sin(t) + 30
    return x, y, z