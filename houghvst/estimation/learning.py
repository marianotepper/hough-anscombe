import lasagne
import numpy as np
import theano
import theano.tensor as T
from noise.utils import poisson_gaussian_noise


def build_net(input_var=None):
    from lasagne.init import Normal
    w = theano.shared(lasagne.utils.floatX(Normal(10, mean=0)((1, 16))))

    net = {}
    net['input'] = lasagne.layers.InputLayer(shape=(None, 1),
                                             input_var=input_var)
    net['linear'] = lasagne.layers.DenseLayer(net['input'], num_units=16,
                                         W=T.exp(w), nonlinearity=None)

    net['max'] = lasagne.layers.FeaturePoolLayer(net['linear'], 4, pool_function=T.max)
    net['min'] = lasagne.layers.FeaturePoolLayer(net['max'], 4, pool_function=T.min)
    net['output'] = net['min']
    return net


def iterate_minibatches(inputs, batchsize, replicas, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        inputs_noisy = [poisson_gaussian_noise(inputs[excerpt], sigma, alpha)
                        for _ in range(replicas)]
        yield np.vstack(inputs_noisy).T


def objective_fun(x_vst, original_shape):
    return (T.var(x_vst.reshape(original_shape), axis=-1) - 1) ** 2


sigma = 2
alpha = 5
replicas = 1000
batchsize = 50
max_range = 1

# Load the dataset
np.random.seed(1)

# u = max_range * np.random.rand(10000)
u = np.linspace(0, max_range, num=10000)

# Prepare Theano variables for inputs and targets
input_var = T.matrix('inputs')
# Create neural network model
net = build_net(input_var)

u_vst = lasagne.layers.get_output(net['output'])
loss = objective_fun(u_vst, (batchsize, replicas))
loss = loss.mean()

params = lasagne.layers.get_all_params(net['linear'], trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

train_fn = theano.function([input_var], loss, updates=updates)

num_epochs = 10
for epoch in range(num_epochs):
    train_err = 0
    train_batches = 0
    for inputs in iterate_minibatches(u, batchsize, replicas, shuffle=True):
        train_err += train_fn(inputs.flatten()[:, np.newaxis])
        train_batches += 1
        # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    print('Epoch {} of {}'.format(epoch + 1, num_epochs))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

params_values = [p.get_value() for p in net['linear'].get_params()]
w = np.squeeze(params_values[0])
bias = params_values[1]

fun = theano.function([input_var], lasagne.layers.get_output(net['output']))
u_noisy = np.hstack([poisson_gaussian_noise(u, sigma, alpha)
                     for _ in range(100)])[:, np.newaxis]
out = fun(u_noisy)
out = out.reshape((len(u), 100))


import matplotlib.pyplot as plt

print(u[::100])
plt.matshow(out[::100])

fig, axes = plt.subplots(1, 2)
for a, b in zip(w, bias):
    print(np.exp(a), b)
    axes[0].plot([0, max_range], [np.exp(a) * 0 + b, np.exp(a) * max_range + b])

out = fun(u[:, np.newaxis])
axes[1].plot(u, out)

plt.show()