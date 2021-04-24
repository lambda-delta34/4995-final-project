from collections import defaultdict
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def kde_tf(query, dataset, kernel_tf, bandwidth=1):
    @tf.function(experimental_relax_shapes=True)
    def model_fn(arg):
        return kernel_tf(x=arg, y=query, bandwidth=bandwidth)
    return tf.math.reduce_mean(tf.vectorized_map(model_fn, dataset))

def laplacian_kernel(x, y, bandwidth=1):
    "Laplacian kernel"
    return np.exp(-1. * np.linalg.norm((x-y), ord=1) / bandwidth)


def laplacian_kernel_tf(x, y, bandwidth=1):
    return tf.math.exp(-1. * tf.norm((x - y), ord=1) / bandwidth)


class LaplaceLSH_tf:
    def __init__(self, dimension, bandwidth):
        poisson_param = dimension * 1. / bandwidth
        self.rep = np.random.poisson(poisson_param)
        self.axes = tf.random.uniform([self.rep], minval=0, maxval=dimension, dtype=tf.int32)
        self.thresholds = tf.random.uniform([self.rep], minval=0, maxval=1)
    def hash(self, point):
        return tf.gather(point, self.axes) < self.thresholds
    

class BinningLaplaceLSH_tf:
    def __init__(self, dimension, bandwidth):
        self.dimension = dimension
        self.delta = tf.random.gamma((1, dimension), 2, 1 / bandwidth)[0]
        weight = tf.random.uniform((1, dimension), 0, 1)[0]
        self.shift = tf.math.multiply(self.delta, weight)
        self.max_len = tf.math.floor(tf.nn.relu(tf.math.divide(1 - self.shift, self.delta)))

    def hash(self, point):
        pt_len = tf.nn.relu(tf.math.ceil((point - self.shift) / self.delta))
        return  tf.cast(tf.math.minimum(pt_len, self.max_len), tf.int32)

    
class FastLaplacianKDE_tf:
    def __init__(self, dataset, bandwidth, L):
        shape = tf.shape(dataset).numpy()
        n_points = shape[0]
        dimension = shape[1]
        sizes = np.random.binomial(n_points, L*1./n_points, L)
        self.hashed_points = []
        self.lshs = []
        self.L = L
        self.bandwidth = bandwidth
        self.dataset = dataset

        for size in sizes:
            hash_tf = BinningLaplaceLSH_tf(dimension, 2 * bandwidth)
            sample = np.random.choice(n_points, size, replace=False)
            result = {}
            for row_num in sample:
                row = dataset[row_num, :]
                row_hash = hash_tf.hash(row)
                hash_tuple_hash = hash(tuple(row_hash.numpy()))
                if hash_tuple_hash not in result:
                    result[hash_tuple_hash] = []
                result[hash_tuple_hash].append(row_num)
            self.lshs.append(hash_tf)
            self.hashed_points.append(result)
            
    def laplacian_kernel(self, x, y, bandwidth=1):
        return np.exp(-1. * np.linalg.norm((x-y), ord=1) / bandwidth)

    def kde(self, query):
        estimators = []
        for j in range(self.L):
            query_hash_tf = self.lshs[j].hash(query)
            query_hash = hash(tuple(query_hash_tf.numpy()))

            if query_hash not in self.hashed_points[j]:
                estimators.append(0)
            else:
                bin_size = len(self.hashed_points[j][query_hash])
                point_ind = self.hashed_points[j][query_hash][np.random.randint(bin_size)]
                point = self.dataset[point_ind, :]
                estimators.append(self.laplacian_kernel(query, point, 2*self.bandwidth) * bin_size * 1. / self.L)
        return np.mean(estimators)