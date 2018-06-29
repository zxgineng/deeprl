import numpy as np

class PriorizedExperienceReplay:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = SumTree(memory_size)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # adj_pri = pri^alpha
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped td error

    def add(self, args):
        max_p = np.max(self.memory.tree[-self.memory.capacity:])    # max adj_pri of leaves
        if max_p == 0:
            max_p = self.abs_err_upper
        self.memory.add(max_p, args)  # set the max adj_pri for new adj_pri

    def get_batch(self, batch_size):
        leaf_idx, batch_memory, ISWeights = np.empty(batch_size, dtype=np.int32), np.empty(batch_size,dtype=object), np.empty(batch_size)
        pri_seg = self.memory.total_p / batch_size  # adj_pri segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # Pi = Prob(i) = softmax(priority(i)) = adj_pri(i) / ∑_i(adj_pri(i))
        # ISWeight = (N*Pj)^(-beta) / max_i[(N*Pi)^(-beta)] = (Pj / min_i[Pi])^(-beta)
        min_prob = np.min(self.memory.tree[self.memory.capacity-1:self.memory.capacity-1+self.memory.counter]) / self.memory.total_p
        for i in range(batch_size):
            # sample from each interval
            a, b = pri_seg * i, pri_seg * (i + 1)   # interval
            v = np.random.uniform(a, b)
            idx, p, data = self.memory.get_leaf(v)
            prob = p / self.memory.total_p
            ISWeights[i] = np.power(prob / min_prob, -self.beta)
            leaf_idx[i], batch_memory[i] = idx, data
        return leaf_idx, batch_memory, ISWeights

    def update_sum_tree(self, tree_idx, td_errors):
        priority = td_errors + self.epsilon  # avoid 0
        clipped_pri = np.minimum(priority, self.abs_err_upper)
        adj_pri = np.power(clipped_pri, self.alpha)
        for ti, p in zip(tree_idx, adj_pri):
            self.memory.update(ti, p)

    @property
    def length(self):
        return self.memory.counter

    def load_memory(self, memory):
        self.memory = memory

    def get_memory(self):
        return self.memory


class SumTree(object):
    """restore adjust priority in leaves and sum in nodes"""
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        self.counter = 0
        self.tree = np.zeros(2 * capacity - 1)  # for all nodes(n - 1) and all leaves(n)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1    # first leaf index
        self.data[self.data_pointer] = data  # update transition
        self.update(tree_idx, p)  # update tree

        if self.counter < self.capacity:
            self.counter += 1
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, adj_pri):
        change = adj_pri - self.tree[tree_idx]    # change between previous adj_pri and current adj_pri
        self.tree[tree_idx] = adj_pri
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # index of relative node of this leaf
            self.tree[tree_idx] += change   # add change to the sum node

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing adj_priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing adj_priority for transitions
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # total adj_pri
