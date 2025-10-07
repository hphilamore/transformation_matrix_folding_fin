import numpy as np
from math import cos, sin, radians, pi
import matplotlib.pyplot as plt

"""
Theory taken from: FORWARD KINEMATICS: THE DENAVIT-HARTENBERG CONVENTION
https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/current/Papers/chap3-forward-kinematics.pdf

TODO
Update geometric parameters calculated automatically (including fold angle) when number of rays is changed
Folding to state 3
Replace identity matrix with A0
Add calculation of base coordinates to paper and diagram
Do the working to check matrix product is correct
Model description with sketh in README 
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
Division factor, where ray angle = pi/division factor 
- can be greater than or equal to number of fin rays   
- must be even number
"""
division_factor = 8

"""
Spacing between each toe 
Choose one, comment out the other
"""
# Equal spacing between each toe 
spacing = n_rays   

# If division factor is greater, spacing used for this number of rays (smaller angle between toes)
# spacing = division_factor


def plot_fin(positions, rotations, fin_base, scale=0.5):

    """
    Plots a 3D scatter plot with folding points (joints) along the fin edge connected by a line.
    The local coordinate axes are shown at each joint.
    The points are all connected to a single origin to create a wrireframe of a folded surface.

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
    for origin, R in zip(positions, rotations):
        # Local axes
        x_vec = R[:,0]*scale
        y_vec = R[:,1]*scale
        z_vec = R[:,2]*scale

        ax.quiver(*origin, *x_vec, color='r', linewidth=1)
        ax.quiver(*origin, *y_vec, color='g', linewidth=1)
        ax.quiver(*origin, *z_vec, color='b', linewidth=1)

        # Label local axes
        ax.text(*(origin + x_vec), 'x', color='r', fontsize=10)
        ax.text(*(origin + y_vec), 'y', color='g', fontsize=10)
        ax.text(*(origin + z_vec), 'z', color='b', fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 1.5)

    # Plot wireframe by connecting joints to fin origin
    print(f'Check each point is equal distance from fin origin and equal to hinge length l={l}:')
    for i, position in enumerate(positions):

        # Get pairs of x, y and z coordinates to connect fin base to position 
        xyz = [[i, j] for i, j in zip(fin_base, position)]
        x_ = xyz[0]
        y_ = xyz[1]
        z_ = xyz[2]
        ax.plot(x_, y_, z_, 'r--')  # XY plane

        # # Check each point is equal euclidian distance from fin origin
        # mag = ((x_[0]-x_[1])**2 + 
        #        (y_[0]-y_[1])**2 + 
        #        (z_[0]-z_[1])**2) ** (1/2)
        # print(f'Distance of point {i} from fin origin: {round(mag, 4)}')
        
        mag = ((fin_base[0]-position[0])**2 + 
               (fin_base[1]-position[1])**2 + 
               (fin_base[2]-position[2])**2) ** (1/2)
        print(f'Distance of point {i} from fin origin: {round(mag, 2)}')

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
    """
    # Set angles for inner and outer folds
    inner_angle = theta
    outer_angle = -inner_angle * 1/2

    # Calculate outer angle to create equal spacing between fin rays  
    scale_factor = 1 - (4 / n_rays)
    scale_factor = 1 - (4 / (division_factor))
    scale_factor = 1 - (4 / spacing)
    outer_angle = -inner_angle * scale_factor


    # Joint parameters for first two joints
    params = [(beta, a, gamma, inner_angle),
              (beta, a, gamma, outer_angle)]

    # Number of times to repeat to describe full fin
    repeats = int(n_rays/2)

    # Joint parameters for all joints
    params = params * repeats
    
    return params


# """
# Number of fin rays
# (Must be even number)
# """
# n_rays = 6

# """
# Size of each ray as multiple of pi
# Division factor:
# - can be greater than or equal to number of fin rays   
# - must be even number
# """
# division_factor = 6

# # Fin ray angle
# ray_angle = 1/ray_division_factor


# """
# Spacing between each toe 
# Choose one, comment out the other
# """
# # Equal spacing between each toe 
# spacing = n_rays   

# # If division factor is greater, spacing used for this number of rays (smaller angle between toes)
# spacing = division_factor




# Fin ray angle
ray_angle = 1/division_factor

# alpha = pi / n_rays
alpha = pi * ray_angle

# Length of hinge from fin base to fin edge
l = 1

# Height of fin ray
h = l * cos(alpha/2)

# Chord of fin ray (modelled as link length)
a = 2 * l * sin(alpha/2)

print(f"a = {a}, h = {h}")

# 3D cooridinates of the fin base 
fin_base = [a/2, 0, h]
fin_base.append(1)
fin_base = np.array(fin_base)
fin_base = Ry(-alpha/2) @ fin_base
fin_base = fin_base[:3]
print(f"fin_base = {fin_base}")

# Offset angle of each fin ray relative to previous, for translation
# beta = -pi/16
beta = -alpha/2

# Chord length of each fin ray edge (modelled as link length)
# a = 0.39

# Offset angle relative to translation axes, for joint rotation 
# gamma = -pi/16
gamma = -alpha/2

# Iterate through a sequence of joint angles, from fully open fin, to fully folded
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
    
    # 4 x 4 identity matrix
    T = np.eye(4)

    # 3D vector representing position of the joint with respect to the inertial or base frame 
    positions = [T[:3,3].copy()]

    # 3 x 3 matrix describing orientation of the joint with respect to the inertial or base frame
    rotations = [T[:3,:3].copy()]  

    # Apply the trasformation to each joint on the fin
    for (beta, a, gamma, theta) in params:

        # Update matrix T, the transform from base frame (frame 0) to frame i
        T = T @ transformation_matrix(beta, a, gamma, theta)

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






