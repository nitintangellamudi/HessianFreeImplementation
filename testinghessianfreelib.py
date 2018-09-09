from __future__ import print_function
import hessianfree as hf
import numpy as np
from mnist import MNIST
import pickle


# mndata = MNIST('samples')
#
# images, labels = mndata.load_training()

def mnist(model_args=None, run_args=None):
    """Test on the MNIST (digit classification) dataset.

    Download dataset at http://deeplearning.net/data/mnist/mnist.pkl.gz

    :param dict model_args: kwargs that will be passed to the :class:`.FFNet`
        constructor
    :param dict run_args: kwargs that will be passed to :meth:`.run_epochs`
    """

    with open("mnist.pkl", "rb") as f:
        try:
            train, _, test = pickle.load(f)
        except UnicodeDecodeError:
            # python 3
            with open("mnist.pkl", "rb") as f2:
                train, _, test = pickle.load(f2, encoding="bytes")

    if model_args is None:
        ff = hf.FFNet([28 * 28, 1024, 512, 256, 32, 10],
                      layers=([hf.nl.Linear()] + [hf.nl.ReLU()] * 4 +
                              [hf.nl.Softmax()]),
                      use_GPU=False, debug=False)
    else:
        ff = hf.FFNet([28 * 28, 1024, 512, 256, 32, 10],
                      layers=([hf.nl.Linear()] + [hf.nl.ReLU()] * 4 +
                              [hf.nl.Softmax()]),
                      **model_args)

    inputs = train[0]
    targets = np.zeros((inputs.shape[0], 10), dtype=np.float32)
    targets[np.arange(inputs.shape[0]), train[1]] = 0.9
    targets += 0.01

    tmp = np.zeros((test[0].shape[0], 10), dtype=np.float32)
    tmp[np.arange(test[0].shape[0]), test[1]] = 0.9
    tmp += 0.01
    test = (test[0], tmp)

    if run_args is None:
        ff.run_epochs(inputs, targets,
                      optimizer=hf.opt.HessianFree(CG_iter=250,
                                                   init_damping=45),
                      minibatch_size=7500, test=test, max_epochs=125,
                      test_err=hf.loss_funcs.ClassificationError(),
                      plotting=True)
    else:
        CG_iter = run_args.pop("CG_iter", 250)
        init_damping = run_args.pop("init_damping", 45)
        ff.run_epochs(inputs, targets,
                      optimizer=hf.opt.HessianFree(CG_iter, init_damping),
                      test=test, test_err=hf.loss_funcs.ClassificationError(),
                      **run_args)

    output = ff.forward(test[0])
    print("classification error",
          hf.loss_funcs.ClassificationError().batch_loss(output, test[1]))
# or
# images, labels = mndata.load_testing()

mnist(model_args=None, run_args=None)



# hf.demos.xor()