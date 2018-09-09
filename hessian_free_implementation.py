# Compare 1. Iterative Hessian Sketch, 2. Hessian-Free (HF) and possibly others.
# It would be nice to also compare with first order and accelerated versions
#
import numpy as np
import tensorflow as tf
epsilon = 0.00000005

# Import DATASETS
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Conjugate Gradient algorithm adapted from www.georgioudakis.com/blog/python.conjugate_gradient.html
# f is the loss function
def gradient(x):
    return tf.gradients(Loss, x)


def finite_differences(x,d):
    return (gradient(x + epsilon * d) - gradient(x)) / epsilon


# After defining whatever loss function
# We will define the Talor Approximation
# f(a0 + p) = (0.5*pt*B*p + gradientF(a0)t*p + f(a0)
# In order to get the optimal p
# we go from a0 to a1, we use CG

# CG Alg
# f(x) = 0.5*xt*A*x + bt + c
# In our case A = B and b = gradientf
# Evaluate -gradient(f) at any initial guess xo
# This is our initial direction do
# We move in that direction an amount alpha
# To find that amount alpha we use
# alpha = -(dit*(A*xi + b))/(dit*A*di)
# x1 = xo - alpha*gradientF(xo)
# Now we select d1 such that
# d1 = -gradientf(x1) + Bo*do
# where Bo = (gradientF(x1)t*a*do)/(dot*A*do)
# Now go back to finding alpha and repeat alg
def CG(b, x0, TOLERANCE=1.0e-10, MAX_ITERATIONS=100):
    """
    A function to solve [A]{x} = {b} linear equation system with the
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : array
        A real symmetric positive definite matrix.
        In our case this will be the Hessian. We want to avoid using this, so it will not be used.
        We will use a finite differences to calculate H*d by calling finite_differences(x,d) where x is the current
        point and d is the direction of movement
    b : vector
        The right hand side (RHS) vector of the system.
        In our case it will be the gradient of a specific point
        We need to be able to calculate gradients at more than
        Just that one point
        Therefore we will pass this as a function so we can
        evaluate the gradient of the function at that point
    x0 : vector
        The starting guess for the solution. Anything will do.
    MAX_ITERATIONS : integer
        Maximum number of iterations. Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    TOLERANCE : float
        Tolerance to achieve. The algorithm will terminate when either
        the relative or the absolute residual is below TOLERANCE.
    """

    #   Initializations
    x = x0
    d = -tf.gradient(Loss, x)
    r0 = b - finite_differences(x,d)


    #   Start iterations
    for i in range(MAX_ITERATIONS):
        a = float(np.dot(d.T, r0) / np.dot(d.T, finite_differences(x, d)))
        x = x - a * gradient(x)

        ri = r0 - np.dot(finite_differences(x, d), d)

        # print i, np.linalg.norm(ri)
        # Checks stopping condition
        if np.linalg.norm(ri) < TOLERANCE:
            return x

        # Otherwise go on to find new direction
        b = float(np.dot(gradient(x).T, finite_differences(x, d)))
        d = - gradient(x) + b * d
        r0 = ri
    return x


# Return H^-1*-grad f(x)
def NewtonStep(params, loss):
    deltax = CG(-tf.gradients(Loss,params), params)
    return deltax - params

# ----------------------------- Defining Feedforward Neural Network --------------------------- #
input_dim = [28, 28]
out_dim = 10
n_nodes_in_layer = 128
batch_size = 100
seed = 4567345

weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3])
hl = tf.contrib.layers.fully_connected(x, n_nodes_in_layer, activation_Fn=tf.nn.relu,
                                       weights_initializer=weights_initializer)
h2 = tf.contrib.layers.fully_connected(hl, n_nodes_in_layer, activation_Fn=tf.nn.relu,
                                       weights_initializer=weights_initializer)
h3 = tf.contrib.layers.fully_connected(h2, n_nodes_in_layer, activation_Fn=tf.nn.relu,
                                       weights_initializer=weights_initializer)
h4 = tf.contrib.layers.fully_connected(h3, n_nodes_in_layer, activation_Fn=tf.nn.relu,
                                       weights_initializer=weights_initializer)
prediction = tf.contrib.layers.fully_connected(h4, out_dim, activation_Fn=None)

label = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])  # Should be a one-hot vector.

Loss = 0.5 * tf.reduce_sum((label - prediction) ** 2)  # Mean Squared Error


# ----------------------------- Defining Feedforward Neural Network --------------------------- #


# take in datas (x,y) where x is the input data, y is the 1-hot encoding of the label
# Return loss of current params with respect to the data passed in.
def Update(in_vals, one_hot):
    params = tf.trainable_variables()
    v = NewtonStep(params, loss)
    params = params - v
	,loss_val = sess.run([params,Loss], feed_dict = {x: in_vals, label: one_hot})
	return loss_val

sess = tf.Session()
sess.run(tf.global_variables_initializer())

max_iters = 10 ** 5
batch_size = 32
for t in range(max_iters):
	# Sample data of size <batch_size>
	curr_loss = Update()

	print("Loss at iteration ",t+1,": ",curr_loss) # Print current loss


sess.close()
