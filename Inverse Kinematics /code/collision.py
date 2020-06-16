import numpy as np


def line_sphere_intersection(p1, p2, c, r):
    """
    Implements the line-sphere intersection algorithm.
    https://en.wikipedia.org/wiki/Line-sphere_intersection

    :param p1: start of line segment
    :param p2: end of line segment
    :param c: sphere center
    :param r: sphere radius
    :returns: discriminant (value under the square root) of the line-sphere
        intersection formula, as a np.float64 scalar
    """
    # FILL in your code here

    v1 = np.dot(((p2-p1)/np.linalg.norm(p2-p1)),(p1-c))
    v2 = (np.linalg.norm(p1 - c)**2 - (r**2))

    return -(v1**2-v2)
