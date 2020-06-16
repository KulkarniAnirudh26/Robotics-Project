
import numpy as np
from collision import CollisionBox, CollisionSphere

class RRT():
    """
    Simple implementation of Rapidly-Exploring Random Trees (RRT)
    """
    class Node():
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            """
            :param state: np.array of a state in the search space.
            :param parent: parent Node object.
            """
            self.state = np.asarray(state)
            self.parent = parent
            self.children = []

        def __iter__(self):
            """
            Breadth-first iterator.
            """
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def add_child(self, state):
            """
            Adds a new child at the given state.

            :param statee: np.array of new child node's statee
            :returns: child Node object.
            """
            child = RRT.Node(state=state, parent=self)
            self.children.append(child)
            return child


    def __init__(self,
                 start_state,
                 goal_state,
                 dim_ranges,
                 obstacles=[],
                 step_size=0.05,
                 max_iter=1000):
        """
        :param start_state: Array-like representing the start state.
        :param goal_state: Array-like representing the goal state.
        :param dim_ranges: List of tuples representing the lower and upper
            bounds along each dimension of the search space.
        :param obstacles: List of CollisionObjects.
        :param step_size: Distance between nodes in the RRT.
        :param max_iter: Maximum number of iterations to run the RRT before
            failure.
        """
        self.start = RRT.Node(start_state, None)
        self.goal = RRT.Node(goal_state, None)
        self.dim_ranges = dim_ranges
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter


        if (self.start.state.shape != self.goal.state.shape):
            raise AssertionError("Start and Goal states do not match dimension!")

    def build(self):
        """
        Build an RRT.

        In each step of the RRT:
            1. Sample a random point.
            2. Find its nearest neighbor.
            3. Attempt to create a new node in the direction of sample from its
                nearest neighbor.
            4. If we have created a new node, check for completion.

        Once the RRT is complete, add the goal node to the RRT and build a path
        from start to goal.

        :returns: A list of states that create a path from start to
            goal on success. On failure, returns None.
        """
        for k in range(self.max_iter):
            # FILL in your code here
            point_random = self._get_random_sample()
            nearest_node = self._get_nearest_neighbor(point_random)
            new_node = self._extend_sample(point_random,nearest_node)
            if new_node and self._check_for_completion(new_node):
                # FILL in your code here
                node3 = new_node.add_child(self.goal.state)
                path = self._trace_path_from_start(node3)
                return path
        
        return None
                

        print("Failed to find path from {0} to {1} after {2} iterations!".format(
            self.start.state, self.goal.state, self.max_iter))

    def _get_random_sample(self):
        """
        Uniformly samples the search space.

        :returns: A vector representing a randomly sampled point in the search
            space.
        """
        # FILL in your code here
        dim = self.start.state.shape
        A = np.arange(0,1,0.1)
        random_point = np.random.choice(A, dim)
        #print(random_point)
        return random_point

    def _get_nearest_neighbor(self, sample):
        """
        Finds the closest node to the given sample in the search space,
        excluding the goal node.

        :param sample: The target point to find the closest neighbor to.
        :returns: A Node object for the closest neighbor.
        """
        # FILL in your code here
        
        #Node.iter()
        max_distance = float('inf')
        for i in self.start:
            a = i.state
            distance = np.linalg.norm(a-sample)
            if (distance < max_distance):
                max_distance = distance
                neighbor = i
                
        return neighbor
        

    def _extend_sample(self, sample, neighbor):
        """
        Adds a new node to the RRT between neighbor and sample, at a distance
        step_size away from neighbor. The new node is only created if it will
        not collide with any of the collision objects (see
        RRT._check_for_collision)

        :param sample: target point
        :param neighbor: closest existing node to sample
        :returns: The new Node object. On failure (collision), returns None.
        """
        # FILL in your code here
        #intermediate_vector = (neighbor.state - sample)/np.sqrt(sum((neighbor.state - sample)**2))
        distance = np.linalg.norm(sample-neighbor.state)
        if distance <= self.step_size:
            new_state = sample
        else:
            new_state = neighbor.state + (self.step_size/distance)*(sample - neighbor.state)
        if self._check_for_collision(new_state):
            return None
        else:
            #Add node to the RRT
            #new_node = RRT.Node(new_state, neighbor)
            new_node = neighbor.add_child(new_state)
            return new_node

    def _check_for_completion(self, node):
        """
        Check whether node is within self.step_size distance of the goal.

        :param node: The target Node
        :returns: Boolean indicating node is close enough for completion.
        """
        # FILL in your code here
        if np.linalg.norm(node.state - self.goal.state)  <= self.step_size :
            return True
        else:
            return False 
            '''
    def recur(self,node,visited = None ):
            if visited == None:
                visited = []
            visited.append(node.state)
            for k in node.children:
                if np.array_equal(k.state,self.goal.state):
                    visited.append(k.state)
                    return visited 
                self.recur(k,visited)
            return visited

            stack = []
        visited = []
        stack.append(self.start)
        while stack:
            node1 = stack.pop()
            visited.append(node1.state)
            if np.array_equal(node1.state,self.goal.state):
                return visited
            else:
                for k in node1.children:
                    for i in visited:
                        if not np.array_equal(k,visited[i]):
                            stack.append(k
            '''

    def _trace_path_from_start(self, node=None):
        """
        Traces a path from start to node, if provided, or the goal otherwise.

        :param node: The target Node at the end of the path. Defaults to
            self.goal
        :returns: A list of states (not Nodes!) beginning at the start state and
            ending at the goal state.
        """
        # FILL in your code here
        #path = self.recur(self.start)
        path = []
        if node == None:
            node = self.goal
            while (node != None):
                path.append(node.state)
                node = node.parent
        else:
            while (node != None):
                path.append(node.state)
                node = node.parent

        return path[::-1]

    def _check_for_collision(self, sample):
        """
        Checks if a sample point is in collision with any collision object.

        :returns: A boolean value indicating that sample is in collision.
        """
        # FILL in your code here
        for k in self.obstacles:
            if k.in_collision(sample):
                return True
        return False
        




   
    
