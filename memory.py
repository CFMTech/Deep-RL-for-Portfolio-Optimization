from collections import deque
import numpy as np
import random


class Memory:
    """
    Class of the uniform experience replay memory.
    """

    def __init__(self, max_size):
        """
        Description
        -------------
        Constructor of class Memory.

        Attributes & Parameters
        -------------
        max_size : Int, the maximum size of the replay memory
        buffer   : collections.deque object of maximum length max_size, the container
                   representing the replay memory
        """

        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """
        Description
        -------------
        Add experience to the replay buffer.

        Parameters
        -------------
        experience : 5-tuple representing a transiction.
        """

        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Description
        -------------
        Randomly sample "batch_size" experiences from the replay buffer.

        Parameters
        -------------
        batch_size : Int, the number of experiences to sample.

        Returns
        -------------
        List containing the sampled experiences.
        """

        return random.sample(self.buffer, batch_size)


class Node:

    # This counter is a class attribute and will define indices of leaves in the array
    # where we store them.
    count = 0
    # Storing the index will help us deal with the reaplay buffer when it gets saturated.
    saturated = False  # This boolean will turn True when Node.count >= max_size

    def __init__(
        self,
        max_size,
        index_heap=None,
        l_child=None,
        r_child=None,
        children_heap=[],
        parent=None,
        parent_heap=None,
        value=0.0,
        sliding="oldest",
    ):
        """
        Description
        ------------------------
        Constructor of class Node, this class represents objects describing how nodes are
        interlinked in both a sum tree and a heap.

        Parameters & Attributes
        ------------------------
        max_size      : Int, maximum size of the replay buffer where we store the leaves.
        index         : Int, index of the leaf on the storing array (only useful when the
                        node is actually a leaf).
        index_heap    : Int, index of the leaf on the storing array of the heap.
        l_child       : None or object of class Node, Left Child in the sum tree.
        r_child       : None or object of class Node, Right Child in the sum tree.
        children_heap : Sorted List containing the children of the node in the heap.
                        Sorting makes method sift_down of Heap easy.
        parent        : None or object of class Node, parent in sum tree. Setting the
                        parent helps us updating the value of each node
                        starting from a changed leaf up to the root.
        parent_heap   : None or object of class Node, parent in the heap.
        value         : Float, sum over all values of the subtree spanned by this node as
                        a root (TD error magnitude in case of a leaf).
        sliding       : String in ['oldest', 'random'], when the tree gets saturated and a
                        new experience comes up.
                           - 'oldest' : Oldest leaves are the first to be changed.
                           - 'random' : Random leaves are changed.
        leaf          : Boolean, whether the node is a leaf in the sum tree or not (True
                        when both l_child and r_child are None).
        leaf_heap     : Boolean, whether the node is a leaf in the heap or not (True when
                        both l_child_heap and r_child_heap are None).
        level         : Int, it specifies the hierarchy of nodes in the sum tree starting
                        from the leaves (0) and up to the root.
        complete      : Boolean, whether the node has both of its children in the same
                        level or not.

        """

        self.max_size = max_size
        self.index = Node.count
        self.index_heap = index_heap
        self.l_child = l_child
        self.r_child = r_child
        self.children_heap = sorted(children_heap, reverse=True)
        self.parent = parent
        self.parent_heap = parent_heap
        self.value = value
        self.leaf = (l_child is None) & (r_child is None)
        self.leaf_heap = len(children_heap) == 0
        self.complete = False
        if self.leaf:
            # Set the leaf index to class attribute count.
            self.index = Node.count
            # Increment class attribute count to account for tree saturation.
            Node.count += 1
            self.level = 0  # Level 0 because it is a leaf.
            # Update class attribute count (tree saturation status).
            Node.saturated = Node.count >= self.max_size

        elif self.r_child is None:
            # Every node that is not a leaf has at least a left child, in case it does not
            # have a right child, the node's level is the increment by 1 of the level of
            # its left child.
            self.level = self.l_child.level + 1

        else:
            # In case the node has both children, it takes the increment by 1 of the
            # minimum level. The reason is that when the tree evolves
            # by adding new leaves, this node will eventually have its children change
            # until reaching the mentioned minimum level.
            self.level = min(self.l_child.level, self.r_child.level) + 1
            self.complete = self.l_child.level == self.r_child.level

    def reset_count():
        """
        Description
        ------------------------
        Class method, resets class attribute count to 0

        Parameters
        ------------------------

        Returns
        ------------------------
        """
        Node.count = 0
        Node.saturated = False

    def update_complete(self):
        """
        Description
        ------------------------
        Update the status (complete or not) of the current node, this can be triggered by
        an update of its children.

        Parameters
        ------------------------

        Returns
        ------------------------
        """

        assert not self.leaf, "Do not update the status of a leaf"
        if self.r_child is None:
            pass

        else:
            self.complete = self.l_child.level == self.r_child.level

    def update_level(self):
        """
        Description
        ------------------------
        Update the level of the current node, this can be triggered by an update of its
        children.

        Parameters
        ------------------------

        Returns
        ------------------------
        """

        # Since we obviously do not update the level of a leaf, the if self.leaf condition
        # can be omitted.
        if self.r_child is None:
            # Every node that is not a leaf has at least a left child, in case it does not
            # have a right child, the node's level is the increment by 1 of the level of
            # its left child.
            self.level = self.l_child.level + 1

        else:
            # In case the node has both children, it takes the increment by 1 of the
            # minimum level. The reason is that when the tree evolves by adding new
            # leaves, this node will eventually have its children change until reaching
            # the mentioned minimum level.
            self.level = min(self.l_child.level, self.r_child.level) + 1

    def update_value(self):
        """
        Description
        ------------------------
        Update the value of the node after setting its left and right children.

        Parameters
        ------------------------

        Returns
        ------------------------
        """

        self.value = self.l_child.value + self.r_child.value

    def update(self):
        """
        Description
        ------------------------
        Update level, status and value attributes of the node.

        Parameters
        ------------------------

        Returns
        ------------------------
        """

        self.update_level()
        self.update_complete()
        self.update_value()

    def update_leaf_heap(self):
        """
        Description
        ------------------------
        Update the attribute leaf_heap.

        Parameters
        ------------------------

        Returns
        ------------------------
        """

        self.leaf_heap = len(self.children_heap) == 0

    def set_l_child(self, l_child):
        """
        Description
        ------------------------
        Set the left child of the node.

        Parameters
        ------------------------
        l_child : Object of class Node, Left child.

        Returns
        ------------------------
        """

        self.l_child = l_child

    def set_r_child(self, r_child):
        """
        Description
        ------------------------
        Set the right child of the node.

        Parameters
        ------------------------
        r_child : Object of class Node, Right child.

        Returns
        ------------------------
        """

        self.r_child = r_child

    def set_children_heap(self, children_heap):
        """
        Description
        ------------------------
        Set the nodes' children in the heap.

        Parameters
        ------------------------
        children_heap : sorted List of the children nodes, set their parent to the current
        node.

        Returns
        ------------------------
        """

        self.children_heap = children_heap
        self.children_heap.sort(reverse=True)
        for child in children_heap:
            child.set_parent_heap(self)

    def replace_child_heap(self, child_origin, child_new):
        """
        Description
        ------------------------
        Replace a child among the children of the node in the heap.

        Parameters
        ------------------------
        child_origin : Object of class Node, the child we want to replace, set its parent
                       later according to the usage.
        child_new : Object of class Node, the new child, set its parent to the current
                    node.

        Returns
        ------------------------
        """

        assert child_origin in self.children_heap, (
            "The child you want to replace does not belong to "
            "the children of current node!"
        )
        for i, child in enumerate(self.children_heap):
            if child == child_origin:
                self.children_heap[i] = child_new

        self.children_heap.sort(reverse=True)
        child_new.set_parent_heap(self)

    def add_child_heap(self, child):
        """
        Description
        ------------------------
        Add a new child in the heap to the current node when it does not altready have two
        children.

        Parameters
        ------------------------
        child : Object of class Node, the child we want to add, set its parent to the
                current node.

        Returns
        ------------------------
        """

        assert len(self.children_heap) < 2, (
            "The node has already 2 children, "
            "you cannot add a child, consider replacing operation."
        )
        self.children_heap.append(child)
        self.children_heap.sort(reverse=True)
        child.set_parent_heap(self)

    def set_parent_heap(self, parent_heap):
        """
        Description
        ------------------------
        Set the nodes' children in the heap.

        Parameters
        ------------------------
        parent_heap : Object of class Node, the parent we would like to set to the node in
        the heap.

        Returns
        ------------------------
        """

        self.parent_heap = parent_heap

    def set_index_heap(self, index_heap):
        """
        Description
        ------------------------
        Set the index of the current node in the heap.

        Parameters
        ------------------------
        index_heap : Int, the index in the heap to set to the current node.

        Returns
        ------------------------
        """

        self.index_heap = index_heap

    def __lt__(self, node):
        """
        Description
        ------------------------
        Comparing method, this proves useful when sorting objects inside lists or heaps.
        In fact we define it such that the root of a heap containing multiple nodes is the
        node with the highest priority.

        Parameters
        ------------------------
        node : Object of class Node, another node we compare the current one with.

        Returns
        ------------------------
        """

        return self.value < node.value


def retrieve_leaf(node, s):
    """
    Description
    ------------------------
    Function describing the retrieval strategy of a leaf when sampling, starting from node
    and given the random number 0 <= s <= node.value

    Parameters
    ------------------------
    node : Object of class Node, the root of the subtree we consider.
    s    : Float s.t 0 <= s <= node.value used in sampling.

    Returns
    ------------------------
    Int, index of the retrieved leaf in the sum tree.
    """

    if node.leaf:
        return node.index

    elif node.l_child.value >= s:
        return retrieve_leaf(node.l_child, s)

    else:
        return retrieve_leaf(node.r_child, s - node.l_child.value)


# Vectorized retrieve_leaf
retrieve_leaf_vec = np.vectorize(retrieve_leaf, excluded=set([0]))


def retrieve_value(node):
    """
    Description
    ------------------------
    Retrieve the value of a node, this function is just intended to be vectorized.

    Parameters
    ------------------------
    node : Object of class Node, the considered node.

    Returns
    ------------------------
    Float, the value of the considered node.
    """

    return node.value


# Vectorized retrieve_valuew
retrieve_value_vec = np.vectorize(retrieve_value)


class Heap:
    def __init__(self):
        """
        Description
        ------------------------
        Constructor of class Heap.

        Parameters & Attributes
        ------------------------
        track : List, the table representation of the heap.
        root  : Object of class Node, root of the heap.
        """

        self.track = []
        self.root = None
        self.last_child = None

    def swap(self, child, parent):
        """
        Description
        ------------------------
        Swap the relation parent-child between two nodes, while keeping the tree intact.

        Parameters
        ------------------------
        child  : Object of class Node, the child node.
        parent : Object of class Node, the parent node.

        Returns
        ------------------------
        """

        # We need to keep track of the child node children, the parent node children to
        # make the suitable exchanges and also the parent's parent to replace parent in
        # its children by child.
        child_children_heap, parent_children_heap, grand_parent = (
            child.children_heap,
            parent.children_heap,
            parent.parent_heap,
        )
        # Swap the indices of child and parent in the heap.
        child_index_heap, parent_index_heap = child.index_heap, parent.index_heap
        child.set_index_heap(parent_index_heap)
        parent.set_index_heap(child_index_heap)
        # Swap the children.
        # The parent takes the children of its child.
        parent.set_children_heap(child_children_heap)
        # The child takes the children of its parent
        child.set_children_heap(parent_children_heap)
        # and replaces itself in its children by its parent.
        child.replace_child_heap(child, parent)
        if grand_parent is not None:
            # When grand parent exists, child takes the place of parent in the grand
            # parents' children
            grand_parent.replace_child_heap(parent, child)

        else:
            # When grand parent does not exist, it means parent is the root, we need then
            # to set child as the new root when swapped with parent.
            child.set_parent_heap(None)
            self.root = child

        # Now that child and parent are internally updated, update their positioning in
        # the tracking list.
        self.track[child.index_heap] = child
        self.track[parent.index_heap] = parent

    def sift_up(self, node):
        """
        Description
        ------------------------
        Update the structure of the heap when one of its internal nodes changes its value
        to a higher one.

        Parameters
        ------------------------
        node    : Object of class Node, the modified node.

        Returns
        ------------------------
        changed : Boolean, whether the heap has changed by performing the sift_up or not.
        """

        parent = node.parent_heap
        changed = False
        while (parent is not None) and (node > parent):
            self.swap(node, parent)
            parent = node.parent_heap
            # Entering the while loop means that we modify the heap.
            changed = True

        return changed

    def sift_down(self, node):
        """
        Description
        ------------------------
        Update the structure of the heap when one of its internal nodes changes its value
        to a lower one.

        Parameters
        ------------------------
        node : Object of class Node, the modified node.

        Returns
        ------------------------
        changed : Boolean, whether the heap has changed by performing the sift_up or not.
        """

        children = node.children_heap
        changed = False
        while (len(children) != 0) and (children[0] > node):
            self.swap(children[0], node)
            children = node.children_heap
            # Entering the while loop means that we modify the heap.
            changed = True

        return changed

    def update(self, node, value):
        """
        Description
        ------------------------
        Update the heap when changing the value of a node.

        Parameters
        ------------------------
        node  : Object of class Node, change de value of this node.
        value : Float, new value we assign to node

        Returns
        ------------------------
        """

        value_prev = node.value
        node.value = value
        self.sift_down(node) if value < value_prev else self.sift_up(node)

    def insert(self, node):
        """
        Description
        ------------------------
        Insert a new node to the heap.

        Parameters
        ------------------------
        node : Object of class Node, the new node to be inserted.

        Returns
        ------------------------
        """

        self.track.append(node)
        node.set_index_heap(len(self.track) - 1)
        if self.root is None:
            self.root = node

        else:
            parent = self.track[(node.index_heap - 1) // 2]
            parent.add_child_heap(node)
            # changed = self.sift_up(node)


class SumTree:
    def __init__(self, max_size):
        """
        Description
        ------------------------
        Constructor of class SumTree.

        Parameters & Attributes
        ------------------------
        max_size : Int, maximum number of leaves.
        sub_left : Object of class Node or None, root of the complete subtree built. It is
                   always the left child of the incoming root.
        parents : collections.deque object, container of parents nodes (helps build the
                  tree).
        children : collections.deque object, container of children nodes (helps build the
                   tree).
        complete : Boolean, True : - The number of leaves is a power of 2, parents &
                                     children are empty, sub_left is the root.
                            False : - The number of leaves isn't a power of 2, parents &
                                      children aren't empty, parents[0] is the root.
        """

        self.max_size = max_size
        self.sub_left = None
        self.parents = deque()
        self.children = deque()
        self.complete = False

    def add_leaf(self, node):
        """
        Description
        ------------------------
        Add a new leaf to the tree.

        Parameters
        ------------------------
        node : Object of class Node, the new leaf representing the transition to be added
        to the replay buffer.

        Returns
        ------------------------
        """

        if self.sub_left is None:
            # Add leaf initially to the empty tree.
            self.sub_left = node
            self.complete = True  # The tree is now complete.

        else:  # The tree is not empty
            root = Node(self.max_size, l_child=self.sub_left)  # Define the new root.
            self.sub_left.parent = root  # Set the corresponding parent.
            # Add the new root to the left of Parents container.
            self.parents.appendleft(root)
            # Add the new leaf to the right of Children container.
            self.children.append(node)
            self.complete = False  # The tree is not complete.
            if len(self.parents) >= 2:
                # Set the left child of last parent to children[-2]
                self.parents[-1].l_child = self.children[-2]
                # Set parents[-1] as the parent of children[-2]
                self.children[-2].parent = self.parents[-1]
                # Set the right child of last parent to children[-1]
                self.parents[-1].r_child = self.children[-1]
                # Set parents[-1] as the parent of children[-1]
                self.children[-1].parent = self.parents[-1]
                # Update the attributes of the last parent.
                self.parents[-1].update()
                while self.parents[-1].complete:
                    # Collapsing loop
                    node = self.parents.pop()  # Pop the last parent node.
                    self.children.pop()  # Pop the last child.
                    # Set the last parent (which is complete) to the last child,
                    self.children[-1] = node
                    # The three steps above are what I call a collapse.
                    # If we reach (len(self.parents) == 1) after the collapse,
                    if len(self.parents) == 1:
                        # we must break the while loop.
                        break

                    # Set the left child of last parent to children[-2]
                    self.parents[-1].l_child = self.children[-2]
                    # Set parents[-1] as the parent of children[-2]
                    self.children[-2].parent = self.parents[-1]
                    # Set the right child of last parent to children[-1]
                    self.parents[-1].r_child = self.children[-1]
                    # Set parents[-1] as the parent of children[-1]
                    self.children[-1].parent = self.parents[-1]
                    # Update the attributes of the last parent.
                    self.parents[-1].update()

                # In case we did not collapse every node (except the root)
                if len(self.parents) >= 2:
                    for i in range(-2, -len(self.parents), -1):
                        # Set the left child as we did before in the collapsing loop.
                        self.parents[i].l_child = self.children[i - 1]
                        # Set the corresponding parent.
                        self.children[i - 1].parent = self.parents[i]
                        # Set the right child to the next parent this time.
                        self.parents[i].r_child = self.parents[i + 1]
                        # Set the corresponding parent.
                        self.parents[i + 1].parent = self.parents[i]
                        # Update the attributes of the last parent.
                        self.parents[i].update()

                    # Treating the root independently.
                    # Since len(self.parents) >= 2, set its right child to the next
                    # parent,
                    self.parents[0].r_child = self.parents[1]
                    # Recall that the left child of the root is already set to
                    # self.sub_left .
                    # Update the attributes of the root.
                    self.parents[0].update()

                # In case every node (except the root) collapsed.
                else:
                    # Set the right child to the first child (the only child at this
                    # point).
                    self.parents[0].r_child = self.children[0]
                    # Set the corresponding parent.
                    self.children[0].parent = self.parents[0]
                    # Update the attributes of the root.
                    self.parents[0].update()
                    # Check if we can collapse the root.
                    if self.parents[0].complete:
                        root = self.parents.pop()  # Pop the root.
                        self.children.pop()  # Pop the last child.
                        # We have now a complete tree with root sub_left.
                        self.sub_left = root
                        self.complete = True  # The tree is complete.

            elif len(self.parents) == 1:
                # Set the right child to the first child (the only child at this point).
                self.parents[0].r_child = self.children[0]
                self.children[0].parent = self.parents[
                    0
                ]  # Set the corresponding parent.
                self.parents[0].update()
                # Check if we can collapse the root.
                if self.parents[0].complete:
                    root = self.parents.pop()  # Pop the root.
                    self.children.pop()  # Pop the last child.
                    # We have now a complete tree with root sub_left.
                    self.sub_left = root
                    self.complete = True  # The tree is complete.

    def sample_batch(self, batch_size=64):
        """
        Description
        ------------------------
        Sample batch size leaves according to the distribution expressed by their values.

        Parameters
        ------------------------
        batch_size : Int, the number of leaves to be sampled.

        Returns
        ------------------------
        np.array of shape (batch_size,) containing the indices of the leaves to be sampled
        """

        # Retrieve the root.
        root = self.sub_left if (len(self.parents) == 0) else self.parents[0]
        ss = np.random.uniform(0, root.value, batch_size)
        return retrieve_leaf_vec(root, ss)

    def update(self, leaf):
        """
        Description
        ------------------------
        Update the tree by propagating new value of a leaf up to the root.

        Parameters
        ------------------------
        leaf : Object of class Node, the leaf which value we have changed.

        Returns
        ------------------------
        """

        parent = leaf.parent
        parent.update_value()
        parent = parent.parent
        while parent is not None:
            parent.update_value()
            parent = parent.parent

    def retrieve_root(self):
        """
        Description
        ------------------------
        Retrieve the root node of the tree.

        Parameters
        ------------------------

        Returns
        ------------------------
        Object of class Node, the root of the tree.
        """

        return self.sub_left if len(self.parents) == 0 else self.parents[0]


# def retrieve_first(couple):
#    return couple[0]


def retrieve_first(couple):
    return couple[0]


retrieve_first_vec = np.vectorize(retrieve_first)


class PrioritizedMemory:
    """
    Class of the prioritized experience replay memory.
    """

    def __init__(self, max_size, sliding="oldest"):
        """
        Description
        -------------
        Constructor of class PrioritizedMemory.

        Parameters & Attributes
        ------------------------
        max_size : Int, the maximum size of the replay memory.
        sliding : String in ['oldest', 'random'], when the tree gets saturated and a new
                  experience comes up.
                        - 'oldest' : Oldest leaves are the first to be changed.
                        - 'random' : Random leaves are changed.
        buffer : 2D np.array of shape (2, max_size), the container representing the replay
                 memory.
                       - buffer[0, :] : experiences, 5-tuples representing transictions.
                       - buffer[1, :] : leaves, Objects of class Node representing the
                                        corresponding experiences.
        sliding : String in ['oldest', 'random'], when the tree gets saturated and a new
                  experience comes up.
                       - 'oldest' : Oldest leaves are the first to be changed.
                       - 'random' : Random leaves are changed.
        tree : SumTree object, the sum-tree which leaves represent the stored transitions.
        """

        self.max_size = max_size
        assert sliding in [
            "oldest",
            "random",
        ], "sliding parameter must be either 'oldest' or 'random'"
        self.sliding = sliding
        self.buffer = np.empty((2, max_size), dtype=object)
        self.tree = SumTree(max_size=max_size)  # Initialize Sum-Tree
        self.heap = Heap()  # Initialize Heap

    def update(self, index, priority):
        """
        Description
        -------------
        Change the priority of an already stored leaf and propagate the information up to
        the root of the sum tree.  We also need to update this priority and propagate its
        impact in the heap.

        Parameters
        -------------
        index : Int, index of the leaf we want to propagate the value up to the root from.
        priority : Float > 0, (|delta| + epsilon)^alpha with delta the TD error.

        Returns
        -------------
        """

        # node points to the node object we want to update.
        node = self.buffer[1, index]
        # Update the heap by either a sift_down or a sift_up.
        self.heap.update(node, priority)
        # Notice that the priority of node already changed when updated in the heap,
        self.tree.update(node)
        # thus directly running the sum tree update.

    def add(self, experience, priority):
        """
        Description
        -------------
        Add the tuple (experience, leaf) to the replay buffer.

        Parameters
        -------------
        experience : tuple (state, action, reward, next_state) representing a transition.
        priority   : Float > 0, (|delta| + epsilon)^alpha with delta the TD error.

        Returns
        -------------
        """

        # If the tree is not saturated.
        if not Node.saturated:
            # Create the leaf corresponding to the transition.
            leaf = Node(max_size=self.max_size, value=priority)
            # Fill the replay buffer.
            self.buffer[:, leaf.index] = np.array([experience, leaf], dtype=object)
            # Add new leaf to the tree.
            self.tree.add_leaf(leaf)
            # Add object leaf to the heap, notice that it is a leaf in the sum-tree,
            self.heap.insert(leaf)

        else:
            if self.sliding == "oldest":
                index = Node.count % self.max_size
                # We need to increment Node.count sor that we cycle again through
                Node.count += 1
                # indices from 0 to (self.max_size-1)
            elif self.sliding == "random":
                # No need to increment Node.count here since its value does not matter
                # anymore.
                index = np.random.randint(0, self.max_size)

            leaf = self.buffer[1, index]
            # When the tree gets saturated, replace the previous element in the buffer at
            # index
            self.buffer[:, index] = np.array([experience, leaf])
            # and point to the same previous leaf object in memory to change its value,
            self.update(index, priority)
            # then propagate the information up to the root.

    def sample(self, batch_size):
        """
        Description
        -------------
        Randomly sample "batch_size" experiences from the replay buffer.

        Parameters
        -------------
        tree : Object of class SumTree, we plug it as an agument of the method for memory
               efficiency purpose.
        batch_size : Int, the number of experiences to sample.

        Returns
        -------------
        np.array of shape (batch_size,) : array containing the sampled experiences.
        np.array of shape (batch_size,) : array containing the indices in the replay
                                          buffer of the sampled experiences.
        """

        # Sample indices using the tree.
        indices = self.tree.sample_batch(batch_size)
        # Retrieve the 1st element of each couple (experiences) as well as their indices
        # in the buffer (to access the leaves).
        return list(self.buffer[0, indices]), indices

    def highest_priority(self):
        """
        Description
        -------------
        Return the highest priority in the replay buffer.

        Parameters
        -------------

        Returns
        -------------
        0 < Float <= 1, the highest priority in the replay buffer.
        """

        priority = self.heap.root.value
        return priority

    def n_experiences(self):
        """
        Description
        -------------
        Return the number of experiences stored in the replay buffer so far.

        Parameters
        -------------

        Returns
        -------------
        Int, number of experiences
        """

        return len(self.heap.track)

    def sum_priorities(self):
        """
        Description
        -------------
        Return the sum of all priorities, useful to compute sampling probabilities from
        priorities

        Parameters
        -------------

        Returns
        -------------
        Float > 0, sum of all priorities.
        """

        root = self.tree.retrieve_root()
        return root.value

    def retrieve_priorities(self, indices):
        """
        Description
        -------------
        Return the priorities of experiences placed in indices of the replay buffer.

        Parameters
        -------------
        indices : 1D np.array, indices in the replay buffer of experiences of interest.

        Returns
        -------------
        1D np.array containing the priorities of experiences.
        """

        return retrieve_value_vec(self.buffer[1, indices])
