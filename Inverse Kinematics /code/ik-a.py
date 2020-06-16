import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize

from fk import fk


# arm link lengths
link_lengths = np.array([0.7, 1.0, 1.0])

# joint angle limits
bounds = [(-np.pi, np.pi)] * 3

# initial state (in configuration space)
q0 = np.array([0, 0, 1.86])

# goal state (in task space)
p_d  = np.array([0.1, 1.33, 0])

def objective(x):
    """
    Objective function that we want to minimize.

    :param x: state in configuration space, as numpy array with dtype
        np.float64
    :returns: a scalar value with type np.float64 representing the objective
        cost for x
    """
    # FILL in your code here
    curr_pos = fk(x,link_lengths)
    #print(type(curr_pos))
    diff = curr_pos - p_d
    #print(type(diff))
    variable = (np.linalg.norm(diff))**2

    return variable



def solve_ik(obj, q0, bnds):
    """
    Call the scipy solver.

    :param obj: objective function
    :param q0: initial guess for solution
    :param bnds: list of lower and upper bound tuples for each parameter
    :returns: solution state that minimizes the objective function
    """
    # show initial objective
    print('Initial SSE Objective: ' + str(objective(q0)))

    # call optimizer
    solution = minimize(obj, q0, method='SLSQP', bounds=bnds)
    x = solution.x

    # show final objective
    print('Final SSE Objective: ' + str(objective(x)))

    # print solution
    print('Solution')
    print('x1 = ' + str(x[0]))
    print('x2 = ' + str(x[1]))
    print('x3 = ' + str(x[2]))

    return x

def plot_solution(x):
    """
    Plot IK solution.

    :param x: solution state as a vector of joint angles
    :returns: None
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # start position
    ax.scatter(0, 0, 0, color='r', s=100)

    # desired position
    ax.scatter(p_d[0], p_d[1], p_d[2], color='g', s=100)

    # plot robot
    points1 = fk(x[:1], link_lengths[:1])
    plt.plot([0, points1[0]], [0, points1[1]], [0, points1[2]], color='k')
    for pp in range(1, len(x)):
       points0 = fk(x[:pp], link_lengths[:pp])
       points1 = fk(x[:pp+1], link_lengths[:pp+1])
       plt.plot([points0[0], points1[0]],
                [points0[1], points1[1]],
                [points0[2], points1[2]],
                color='k')

    # show result
    plt.show()

if __name__ == '__main__':
    solution = solve_ik(objective, q0, bounds)
    plot_solution(solution)
