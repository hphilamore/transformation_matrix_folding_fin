import numpy as np
from math import cos, sin, radians, pi
import matplotlib.pyplot as plt

"""
Theory taken from: FORWARD KINEMATICS: THE DENAVIT-HARTENBERG CONVENTION
https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/current/Papers/chap3-forward-kinematics.pdf
"""

"""
------- INPUT PARAMETERS --------
"""

"""
Number of fin rays
(Must be even number)
"""
n_rays = 6

"""
Shape factor, where ray angle = pi/fin_angle_factor 
- can be greater than or equal to number of fin rays   
- must be even number
"""
fin_angle_factor = 8

"""
Spacing between each toe 
Choose one, comment out the other
"""
spacing = n_rays  # Equal spacing between each toe 

# If division factor is greater, spacing used for this number of rays (smaller angle between toes)
spacing = fin_angle_factor


def plot_fin(positions, rotations, 
             fin_base, scale=0.5):

    """
    Plots a 3D scatter plot with folding points (joints) along the fin edge connected by a line.
    The local coordinate axes are shown at each joint.
    The points are all connected to a single q to create a wrireframe of a folded surface.

    Parameters
    ----------
    positions : 3D position vector
    rotations: 3 x 3 orientation matrix
    fin_base: 3D cooridinates of the origin of the fin, calculated geometrically
    scale: scale of local coordinate axes

    """

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract joint positions
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    z = [p[2] for p in positions]

    # Plot the links
    ax.plot(x, y, z, '-o', color='black', label='Links')

    # # Projections onto XY, XZ, YZ planes 
    # ax.plot(x, y, 0, 'c--')  # XY plane
    # ax.plot(x, 0, z, 'c--')  # XZ plane
    # ax.plot(0, y, z, 'c--')  # YZ plane

    # Plot local axes at each joint
    for p, R in zip(positions, rotations):
        # Local axes
        x_vec = R[:,0]*scale
        y_vec = R[:,1]*scale
        z_vec = R[:,2]*scale

        ax.quiver(*p, *x_vec, color='r', linewidth=1)
        ax.quiver(*p, *y_vec, color='g', linewidth=1)
        ax.quiver(*p, *z_vec, color='b', linewidth=1)

        # Label local axes
        ax.text(*(p + x_vec), 'x', color='r', fontsize=10)
        ax.text(*(p + y_vec), 'y', color='g', fontsize=10)
        ax.text(*(p + z_vec), 'z', color='b', fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 1.5)

    # Plot wireframe by connecting joints to fin q
    print(f'Check each point is equal distance from fin q and equal to hinge length l={l}:')
    # for i, position in enumerate(positions):
    for i, (p, fb) in enumerate(zip(positions, fin_base)):

        # Get pairs of x, y and z coordinates to connect fin base to position 
        x_ = [fb[0], p[0]] 
        y_ = [fb[1], p[1]]
        z_ = [fb[2], p[2]]
        ax.plot(x_, y_, z_, 'r--')  # XY plane

        # # Check each point is equal euclidian distance from fin base
        mag = ((fb[0]-p[0])**2 + 
               (fb[1]-p[1])**2 + 
               (fb[2]-p[2])**2) ** (1/2)
        print(f'Distance of joint {i} from fin base: {round(mag, 2)}')

    # plt.show()

def dh_transform(a, alpha, d, theta):
    """
    Standard DH transform
    """
    ct = cos(theta); st = sin(theta)
    ca = cos(alpha); sa = sin(alpha)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,     sa,     ca,    d],
        [0,      0,      0,    1]
    ])

def Rx(g):
    """
    X rotation matrix
    """
    cg = cos(g); sg = sin(g)
    return np.array([
        [1, 0,   0, 0],
        [0, cg, -sg, 0],
        [0, sg,  cg, 0],
        [0, 0,   0, 1]
    ], dtype=float)

def Ry(b):
    """
    Y rotation matrix
    """
    cb = cos(b); sb = sin(b)
    return np.array([
        [ cb, 0, sb, 0],
        [  0, 1,  0, 0],
        [-sb, 0, cb, 0],
        [  0, 0,  0, 1]
    ], dtype=float)

def Rz(t):
    """
    Z rotation matrix
    """
    ct = cos(t); st = sin(t)
    return np.array([
        [ct, -st, 0, 0],
        [st,  ct, 0, 0],
        [ 0,   0, 1, 0],
        [ 0,   0, 0, 1]
    ], dtype=float)

def Tx(a):
    """
    X translation (link length)
    """
    return np.array([
        [1,0,0,a],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ], dtype=float)

def Tz(d):
    """
    Z translation (link offset)
    """
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,d],
        [0,0,0,1]
    ], dtype=float)

def transformation_matrix(beta, a, gamma, theta):
    """
    Build A_i = Ry(beta) @ Tx(a) @ Ry(gamma) @ Rz(theta) 
    where:
      - Ry(beta) rotates about local Y, by fixed offset beta, to set new X axis,
      - Tx(a) is translation in new X direction by link length a,
      - Ry(gamma) rotates about local Y, by fixed offset gamma, to set new Z axis
      - Rz(theta) is joint rotation about new Z by joint angle theta.

    Returns 4x4 numpy array.
    """
    return Ry(beta) @ Tx(a) @ Ry(gamma) @ Rz(theta)

def generate_transformation_params(beta, a, gamma, theta, spacing):
    """
    Generates the individual joint parameters to describe the fin folding 
    from State 1 (fully open), through State 2 (folded), to State 3 (fully folded)
    """
    # Set angle of outer folds
    outer_angle = theta

    # Calculate angle of inner folds:  
    scale_factor = 1 - (4 / spacing)
    inner_angle = -outer_angle / scale_factor
    
    # Inner folds will reach fully folded state before outer, so stop inner folds at this point 
    if inner_angle <= -pi:
        inner_angle = -pi

    # Joint parameters for first two joints
    params = [(beta, a, gamma, inner_angle),
              (beta, a, gamma, outer_angle)]

    # Number of times to repeat to describe full fin
    repeats = int(n_rays/2)

    # Joint parameters for all joints
    params = params * repeats
    
    return params

# Fin ray angle
ray_angle = 1/fin_angle_factor

# alpha = pi / n_rays
alpha = pi * ray_angle

# Length of hinge from fin base to fin edge
l = 1

# Chord of fin ray (modelled as link length)
a = 2 * l * sin(alpha/2)

# Offset angle of each fin ray relative to previous, for translation
beta = -alpha/2

# Offset angle relative to translation axes, for joint rotation 
gamma = -alpha/2

# Iterate through joint angles describing poses from fully open fin (State 1) to folded (State 2)
for i, theta in enumerate(np.linspace(0, pi, 10)):

    # Joint parameters for transformation matrix: beta, a, gamma, theta
    params = generate_transformation_params(beta, a, gamma, theta, spacing)

    # # Example: Fully opn fin (flat)
    # params = [(beta, a, gamma, 0)] * n_rays
    
    # # Example: Partially folded fin (zig-zag) 
    # params = [(beta, a, gamma, pi/4),
    #           (beta, a, gamma, -pi/4)] * int(n_rays/2)

    # # Example: Fully folded fin
    # params = [(beta, a, gamma, pi),
    #           (beta, a, gamma, -pi/2)] * int(n_rays/2)
    
    # Transformation matrix of first joint in kinematic chain (4 x 4 identity matrix)
    T = transformation_matrix(0, 0, 0, 0)

    # 3D vector representing position of the joint with respect to the inertial or base frame 
    positions = [T[:3,3].copy()]

    # 3 x 3 matrix describing orientation of the joint with respect to the inertial or base frame
    rotations = [T[:3,:3].copy()] 

    # Cooridnates of fin base with respect to the inertial or base frame
    fin_base = [[0.0, 0.0, l]] 

    # Apply the trasformation to each joint on the fin
    for (beta, a, gamma, theta) in params:

        # Update matrix T, the transform from base frame (frame 0) to frame i
        T = T @ transformation_matrix(beta, a, gamma, theta)

        # Coordinates of the fin base
        fb = T.copy() @ Tz(1)
        fin_base.append(fb[:3, 3])

        # Update the 3D position vector
        positions.append(T[:3,3].copy())

        # Update the 3 x 3 orientation matrix
        rotations.append(T[:3,:3].copy())

    # Plot the fin
    plot_fin(positions, 
            rotations, 
            fin_base,
            scale=0.1)

    plt.savefig(f'fin_pose_{i}.png')
    plt.show()






