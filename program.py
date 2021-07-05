import nerualnetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']


layer_sizes = (784,3,10)
x = np.ones((layer_sizes[0],1))

net = nn.NeuralNetwork(layer_sizes)
w = net.print_accuracy(training_images,training_labels)
for x in range(10):
    net = nn.NeuralNetwork(layer_sizes)
    w = (w + net.print_accuracy(training_images,training_labels))/2

print(w)


