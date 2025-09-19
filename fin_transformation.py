import numpy as np
from math import cos, sin, radians, pi
import matplotlib.pyplot as plt

"""
Theory taken from: FORWARD KINEMATICS: THE DENAVIT-HARTENBERG CONVENTION
https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/current/Papers/chap3-forward-kinematics.pdf
"""

def plot_fin(positions, rotations, fin_origin, scale=0.5):

    """
    Plots a 3D scatter plot with folding points (joints) along the fin edge connected by a line.
    The local coordinate axes are shown at each joint.
    The points are all connected to a single origin to create a wrireframe of a folded surface.

    Parameters
    ----------
    positions : 3D position vector
    rotations: 3 x 3 orientation matrix
    fin_origin: 3D cooridinates of the origin of the fin, calculated geometrically
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
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.5)
    ax.set_zlim(0, 1.5)

    # Plot wireframe using position of fan origin
    print(f'Checking each point is equal euclidian distance from fin origin')
    for i, position in enumerate(positions):
        xyz = [[i, j] for i, j in zip(fin_origin, position)]
        x_ = xyz[0]
        y_ = xyz[1]
        z_ = xyz[2]
        ax.plot(x_, y_, z_, 'r--')  # XY plane

        # Check each point is equal euclidian distance from fin origin
        mag = ((x_[0]-x_[1])**2 + 
               (y_[0]-y_[1])**2 + 
               (z_[0]-z_[1])**2) ** 1/2
        print(f'Distance of point {i} from fin origin: {round(mag, 4)}')

    # plt.show()

def dh_transform(a, alpha, d, theta):
    ct = cos(theta); st = sin(theta)
    ca = cos(alpha); sa = sin(alpha)

    # Standard DH transform
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
    Build A_i = Ry(beta) @ Rx(gamma) @ Rz(theta) @ Tx(a) @ Tz(d)
    where:
      - beta rotates about local Y to set X direction,
      - gamma rotates about local X to set Z axis,
      - theta is the joint rotation about that Z,
      - Tx(a) and Tz(d) place link endpoint.
    Returns 4x4 numpy array.
    """
    return Ry(beta) @ Tx(a) @ Ry(gamma) @ Rz(theta)

def generate_transformation_params(b, a, g, inner_angle):
    """
    Generates the paramters to describe the fin folding 
    """
    outer_angle = -inner_angle/2

    # return outer_angle

    return([(b, a, g, outer_angle),
              (b, a, g, inner_angle),
              (b, a, g, outer_angle),
              (b, a, g, inner_angle),
              (b, a, g, outer_angle),
              (b, a, g, inner_angle),
              (b, a, g, outer_angle),
              (b, a, g, inner_angle),
              ])

# 3D cooridinates of the origin of the fin, calculated geometrically
fin_origin = [0.39/2, 0, 0.98]

# Chord length along each fin section
l = 0.39

# Angle of each chord, relative to previous
c = -pi/16

# Angle of each z rotation axis, relative to previous
g = -pi/16

# --- Parameters for transformation matrix: beta, a, gamma, theta ---
# Example: Open fan (completely flat)
# params = [(0, l, g, pi/4),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/4),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/4),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/4),
#                     (c, l, g, -pi/4),
#                     ]

# # Example: Zig-zag fold
# params = [(0, l, g, pi/4),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/4),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/4),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/4),
#                     (c, l, g, -pi/4),
#                     ]

# # Example: Zig-zag fold
# params = [(0, l, g, -pi/4),
#                     (c, l, g, pi/2),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/2),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/2),
#                     (c, l, g, -pi/4),
#                     (c, l, g, pi/2),
#                     ]

# # Example: Fully folded fin
# params = [(0, l, g, -pi/2),
#                     (c, l, g, pi),
#                     (c, l, g, -pi/2),
#                     (c, l, g, pi),
#                     (c, l, g, -pi/2),
#                     (c, l, g, pi),
#                     (c, l, g, -pi/2),
#                     (c, l, g, pi),
#                     ]

# Iterate through a series of states from fully open fin to fully folded
for i, angle in enumerate(np.linspace(0, pi, 10)):

    # Parameters for transformation matrix: beta, a, gamma, theta
    params = generate_transformation_params(c, l, g, angle)

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
            fin_origin, 
            scale=0.1)

    plt.savefig(f'fin_pose_{i}.png')
    plt.show()







