import numpy as np
import tensorflow as tf
epsilon = 0.00000005
# Rosenbrock "Banana" function
# Define this as a tensorflow equation
# THIS IS NOT CORRECT but just keeping it as is for now
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
f = tf.Variable((1-x)^2 + 100(y-x^2)^2,name='f')

def gradient(x):
    return tf.gradients(f,x)

def finite_differences(x,d):
    return (gradient(x+epsilon*d)-gradient(x))/epsilon


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
def Taylor_Approx_and_CG(A, b, x0, TOLERANCE = 1.0e-10, MAX_ITERATIONS = 100):
    """
    A function to solve [A]{x} = {b} linear equation system with the
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : array
        A real symmetric positive definite matrix.
        In our case this will be the Hessian. We want to avoid using this
        So we will use a finite differences to calculate H*d
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
    r0 = b - finite_differences(x)
    d = -tf.gradient(f,x)

#   Start iterations
    for i in range(MAX_ITERATIONS):
        a = float(np.dot(d.T, r0)/np.dot(d.T,finite_differences(x,d)))
        x = x - a*gradient(x)

        ri = r0 - np.dot(finite_differences(x,a), d)

        # print i, np.linalg.norm(ri)
        # Checks stopping condition
        if np.linalg.norm(ri) < TOLERANCE:
            return x

        # Otherwise go on to find new direction
        b = float(np.dot(gradient(x).T,finite_differences(x,d)))
        d = - gradient(x) + b * d
        r0 = ri
    return x
