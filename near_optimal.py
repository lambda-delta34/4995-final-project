from math import sqrt, pow, log, ceil
import numpy as np
import math

def gaussian_kernel(x, y, bandwidth=1):
    return np.exp(-1. * np.linalg.norm((x-y), ord=2) / bandwidth) / np.sqrt(2 * np.pi)


def kde(query, dataset, bandwidth):
    return np.mean([gaussian_kernel(query, dataset[i,:], bandwidth) for i in range(dataset.shape[0])])


class NearOptimal:
    def __init__(self, w, dim, n, bandwidth=1):

        # applying Thm 3.3 of the paper
        # I assume 1 + o(1) + 1/c^2 approx 1.2 + 1/c^2
        self.max_dim = ceil(pow(log(n), 2/3))
        self.coef = 4.0
        self.o1 = 0.2

        self.U =  ceil(pow(n, 1 + self.o1 + 1.0/(bandwidth ** 2)))

        self.need_projection = dim > self.max_dim

        self.dim = dim
        self.original_dim = dim
        if self.need_projection:
            self.dim = self.max_dim
            self.projection = np.random.normal(size=(self.max_dim, dim)) / sqrt(self.max_dim * 1.0)       


        self.w = w
        self.lhs = 1.0 / self.coef
        self.rhs = (self.coef - 1.0) / self.coef
        self.n = n
        self.bandwidth = bandwidth
        self.b = np.random.uniform(size=(self.U, self.dim))


    # for a grid, check if point intersect with a ball on a point on grid
    # return False, [] if such grid point does not exist
    def getShiftResult(self, point, b, check_hypercube_axis):
        shifted_pt = point + b
        floor_grid = np.floor(shifted_pt)
        relative_position = shifted_pt - floor_grid

        judgement = check_hypercube_axis(relative_position)
        num_axis_ok = np.linalg.norm(judgement, ord=1) 
        if num_axis_ok < self.dim:
            return False, []
        else:
            target_grid = (judgement + 1.0) / 2
            norm = np.linalg.norm(judgement - target_grid, ord=2) 
            if norm > self.lhs:
                return False, []
        return True, (floor_grid + target_grid).astype(int)
    

    # Return the hash of tuple of
    #  where query point intersect with a ball on a grid point.
    # Hashed tuple: (grid point(projected dim), i) if found
    #               (-1, -1, ..., -1) if not exist
    def hash(self, point):
        def axis_checker(value):
            if value >= self.rhs:
                return  1
            elif value <= self.lhs:
                return -1
            else:
                return 0
        axis_checker_vec = np.vectorize(axis_checker)

        target_pt = point
        if self.need_projection:
            target_pt = np.matmul(self.projection, np.reshape(target_pt, (self.original_dim, 1)))
            target_pt = np.reshape(target_pt, (1, self.dim))[0] / (self.coef * self.w)

        for i in range(self.n):
            ok, result = self.getShiftResult(target_pt, self.b[i, :], axis_checker_vec)
            if ok:
                return hash(tuple(np.append(result, i)))
        return hash(tuple(np.full(self.dim + 1, -1)))
    
class FastKDE_NearOptimal:
    def __init__(self, dataset, bandwidth, L):
        n_points = dataset.shape[0]
        dimension = dataset.shape[1]
        sizes = np.random.binomial(n_points, L*1./n_points, L)
        self.hashed_points = []
        self.lshs = []
        self.L = L
        self.bandwidth = bandwidth
        self.dataset = dataset

        for size in sizes:
            hash = NearOptimal(0.1, dimension, size, bandwidth)
            sample = np.random.choice(n_points, size, replace=False)
            result = {}
            for row_num in sample:
                row = dataset[row_num, :]
                row_hash = hash.hash(row)
                if row_hash not in result:
                    result[row_hash] = []
                result[row_hash].append(row_num)
            self.lshs.append(hash)
            self.hashed_points.append(result)

    def kde(self, query):
        estimators = []
        for j in range(self.L):
            query_hash = self.lshs[j].hash(query)

            if query_hash not in self.hashed_points[j]:
                estimators.append(0)
            else:
                bin_size = len(self.hashed_points[j][query_hash])
                point_ind = self.hashed_points[j][query_hash][np.random.randint(bin_size)]
                point = self.dataset[point_ind, :]
                estimators.append(gaussian_kernel(query, point, self.bandwidth) * bin_size * 1. / (self.bandwidth * self.L))
        return np.mean(estimators)