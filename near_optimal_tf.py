## Our Attempt to Transform NearOptimal into GPU
## One issue is that tensorflow hates mixing its datatype with native Python datatype
## This code is inefficient in ths sense that GPU has to retarce our vectorized code
## which is slow



# import tensorflow as tf
# class NearOptimal:
#     def __init__(self, w, dim, n, bandwidth=1):
#         if bandwidth < 1:
#           print("Small bandwith may cause too much memory comsumption(n^(1/c^2))")

#         # applying Thm 3.3 of the paper
#         # I assume 1 + o(1) + 1/c^2 approx 1.2 + 1/c^2
#         self.max_dim = ceil(pow(log(n), 2/3))
#         self.coef = 4.0
#         self.o1 = 0.2

#         self.U = ceil(pow(n, 1 + self.o1 + 1.0/(bandwidth ** 2)))

#         self.need_projection = dim > self.max_dim

#         self.dim = dim
#         self.original_dim = dim
#         if self.need_projection:
#           self.dim = self.max_dim
#           self.projection = tf.random.normal((self.max_dim, dim)) / tf.sqrt(self.max_dim * 1.0)       


#         self.w = w
#         self.lhs = tf.constant(1.0 / self.coef, dtype=tf.float32)
#         self.rhs = tf.constant((self.coef - 1.0) / self.coef, dtype=tf.float32)
#         self.n = n
#         self.bandwidth = bandwidth
#         self.b = tf.random.uniform((self.U, self.max_dim))


#     def hash(self, point):
#       # Assuming input is a number in [0, 1]
#       # return -1 if less than 0.25, 1 if greater than 0.75, 0 otherwise
#       # for check if there is a chance that a point can intercept with a ball
#       # on grid
#       @tf.function(experimental_relax_shapes=True, input_signature=(tf.TensorSpec(shape=[], dtype=tf.float32),))
#       def check_hypercube_axis(arg):
#         value = arg[0]
#         if value >= self.rhs:
#           return  tf.constant([1], tf.float32)
#         elif value <= self.lhs:
#           return tf.constant([-1], tf.float32)
#         else:
#           return tf.constant([0], tf.float32)
#       target_pt = point
#       if self.need_projection:
#         col = tf.shape(point).numpy()[0]
#         target_pt = tf.matmul(self.projection, tf.reshape(target_pt, (col, 1)))
#         target_pt = tf.reshape(target_pt, (1, self.dim))[0]
      
#       for i in range(self.U):
#         ok, result = self.getShiftResult(target_pt, self.b[i, :], check_hypercube_axis)
#         if ok:
#           cube = result.numpy()
#           return hash(tuple(np.append(cube, i)))

#       return hash(tuple(np.full(self.dim + 1, -1)))

    
#     def getShiftResult(self, point, b, check_hypercube_axis):

#       shifted_pt = tf.add(tf.divide(point, self.coef * self.w), b)
#       floor_grid = tf.floor(shifted_pt)
#       relative_position = tf.subtract(shifted_pt, floor_grid)

#       relative_position_reshape = tf.reshape(relative_position, (self.dim, 1))

#       judgement = tf.reshape(tf.vectorized_map(check_hypercube_axis, relative_position_reshape), (1, self.dim))[0]

#       # at least one axis that is in the middle (-2w, 2w) and so no way for intersection
#       if tf.norm(judgement, ord=1) <  tf.constant(self.dim, dtype=tf.float32):
#         return False, []
#       else:
#         target_grid = tf.math.divide(tf.math.add(judgement, tf.constant(1.0, dtype=tf.float32)), tf.constant(2, dtype=tf.float32))

#         # once a point is in the cornet, check collision
#         if tf.norm(tf.subtract(target_grid, relative_position), ord=2) > self.w:
#           return False, []

#         return True, tf.cast(floor_grid + target_grid, dtype=tf.int32)