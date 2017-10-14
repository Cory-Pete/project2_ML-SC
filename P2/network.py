from math import exp
from random import seed
from random import random

class Network():
    """ Setup the created network object """

    def __init__(self, dataset, hidden_layers):
        self.network = list()
        self.dataset = dataset
        self.inputs = len(dataset[0]) - 1
        self.outputs = len(set([row[-1] for row in dataset]))
        self.hidden_layer = [{'weights':[random() for i in range(self.inputs + 1)]} for i in range(hidden_layers)]
        self.network.append(self.hidden_layer)
        self.output_layer = [{'weights':[random() for i in range(hidden_layers + 1)]} for i in range(self.outputs)]
        self.network.append(self.output_layer)

    def train(self, learn_rate, epoch):
        for n_epoch in range(epoch):
            sum_error = 0
            for row in self.dataset:
                self.forward_propagate(row)
                outputs = self.outs
                #outputs = self.forward_propagate(row)
                expected = [0 for i in range(self.outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backprop_error(expected)
                self.update_weights(row, learn_rate)
            print('>n_epoch=%d, lrate=%.3f, error=%.3f' % (n_epoch, learn_rate, sum_error))

    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                print(neuron)
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                print(neuron['output'])
                new_inputs.append(neuron['output'])
            self.outs = new_inputs

    def backprop_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                        print(neuron['weights'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                self.trans_derive(neuron['output'])
                neuron['delta'] = errors[j] * self.transDerive

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def trans_derive(self, output):
        self.transDerive = output * (1.0 - output)

    def update_weights(self, row, learn_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][-1] += learn_rate * neuron['delta']

    def __str__(self):
        line = ""
        for layer in self.network:
            line += layer+ "\n"
        return line
