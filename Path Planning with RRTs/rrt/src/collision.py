
from abc import ABC, abstractmethod

import numpy as np


class CollisionObject(ABC):
    """
    Abstract class for a parametrically defined collision object.
    """
    @abstractmethod
    def in_collision(self, target):
        """
        Checks whether target point is in collision. Points at the boundary of
        the object are in collision.

        :returns: Boolean indicating target is in collision.
        """
        pass


class CollisionBox(CollisionObject):
    """
    N-dimensional box collision object.
    """
    def __init__(self, location, half_lengths):
        """
        :params location: coordinates of the center
        :params half_lengths: half-lengths of the rectangle along each axis
        """
        self.location = np.asarray(location)
        self.half_lengths = np.asarray(half_lengths)
        self.ndim = self.location.shape[0]

    def in_collision(self, target):
        # FILL in your code here
        #distance = np.sqrt(sum((self.location - np.asarray(target))**2))
        #target = np.asarray(target)
        #distance = np.linalg.norm(self.location.reshape(-1,1) - target.reshape(-1,1))
        obs = []
        for i in range(self.ndim):
            if abs(target[i] - self.location[i]) <= self.half_lengths[i]:
                obs.append(True)
            else:
                obs.append(False)
        
        if False in obs:
            return False
        else:
            return True


class CollisionSphere(CollisionObject):
    """
    N-dimensional sphere collision object.
    """
    def __init__(self, location, radius):
        """
        :params location: coordinates of the center
        :params radius: radius of the circle
        """
        self.location = np.asarray(location)
        self.radius = radius

    def in_collision(self, target):
        # FILL in your code here
        #distance = np.sqrt(sum((self.location - np.asarray(target))**2))
        target = np.asarray(target)
        distance = np.linalg.norm(self.location.reshape(-1,1) - target.reshape(-1,1))
        if distance <= self.radius:
            return True
        else:
            return False
