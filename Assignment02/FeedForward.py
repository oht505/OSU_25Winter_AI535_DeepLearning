"""
Name: Hyuntaek Oh
Email: ohhyun@oregonstate.edu
"""

from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
font = {'weight' : 'normal','size'   : 22}
matplotlib.rc('font', **font)
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

######################################################
# Q1 Implement Init, Forward, and Backward For Layers
######################################################

class SigmoidCrossEntropy:

    # Compute the cross entropy loss after sigmoid. The reason they are put into the same layer is because the gradient has a simpler form
    # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
    # labels -- batch_size x 1 vector of integer label id (0,1) where labels[i] is the label for batch element i
    #
    # TODO: Output should be a positive scalar value equal to the average cross entropy loss after sigmoid
    def forward(self, logits, labels):
        self.logits = logits
        self.labels = labels
        sigmoid = 1.0 / (1.0 + np.exp(-self.logits))
        avg_cross_entropy_loss = -np.mean(self.labels * np.log(sigmoid + 1e-8) + (1.0 - self.labels) * np.log(1.0 - sigmoid + 1e-8))
        return avg_cross_entropy_loss

    # TODO: Compute the gradient of the cross entropy loss with respect to the the input logits
    def backward(self):
        sigmoid = 1 / (1 + np.exp(-self.logits))
        computed_gradient = (sigmoid - self.labels) / self.labels.shape[0]
        # computed_gradient = (sigmoid - self.labels)
        return computed_gradient


class ReLU:

    # TODO: Compute ReLU(input) element-wise
    def forward(self, input):
        self.input = input
        return np.maximum(0, self.input)

    # TODO: Given dL/doutput, return dL/dinput
    def backward(self, grad):
        grad_input = grad * (self.input > 0)
        return grad_input

    # No parameters so nothing to do during a gradient descent step
    def step(self,step_size, momentum = 0, weight_decay = 0):
        return


class LinearLayer:

    # TODO: Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.velocity = np.zeros_like(self.weights)
        # self.velocity_bias = np.zeros_like(self.bias)
        self.learning_rate = 0.001

    # TODO: During the forward pass, we simply compute XW+b
    def forward(self, input):
        self.x = input
        return np.dot(self.x, self.weights) + self.bias


    # TODO: Backward pass inputs:
    #
    # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where
    #         the i'th row is the gradient of the loss of example i with respect
    #         to z_i (the output of this layer for example i)

    # Computes and stores:
    #
    # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
    #                       of the loss with respect to the weights of this layer.
    #                       This is an summation over the gradient of the loss of
    #                       each example with respect to the weights.
    #
    # self.grad_bias dL/dZ--     A (1 x output_dim) matrix storing the gradient
    #                       of the loss with respect to the bias of this layer.
    #                       This is an summation over the gradient of the loss of
    #                       each example with respect to the bias.

    # Return Value:
    #
    # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
    #               the i'th row is the gradient of the loss of example i with respect
    #               to x_i (the input of this layer for example i)

    def backward(self, grad):
        if self.x is None:
            raise ValueError("Forward pass must be called before backward"
                             )
        self.grad_weights = np.dot(self.x.T, grad)
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)
        grad_input = np.dot(grad, self.weights.T)

        return grad_input

    ######################################################
    # Q2 Implement SGD with Weight Decay
    ######################################################
    def step(self, step_size, momentum = 0.8, weight_decay = 0.0):
        # TODO: Implement the step
        self.velocity = momentum * self.velocity - step_size * self.grad_weights
        self.weights = self.weights + self.velocity - 2.0 * step_size * weight_decay * self.weights
        self.bias = self.bias - step_size * self.grad_bias

        # Bias applying momentum
        # self.velocity_bias = momentum * self.velocity_bias - step_size * self.grad_bias
        # self.bias += self.velocity_bias

######################################################
# Q4 Implement Evaluation for Monitoring Training
######################################################

# TODO: Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
    # X_val: (2000, 3072)
    # Y_val: (2000, 1)

    loss_fn = SigmoidCrossEntropy()
    total_loss = 0.0
    total_accuracy = 0.0
    total_error_rate = 0.0
    num_batches = 0

    # testing
    for i in range(0, X_val.shape[0], batch_size):
        X_batch = X_val[i : i + batch_size]
        Y_batch = Y_val[i : i + batch_size]

        logits = model.forward(X_batch)

        loss = loss_fn.forward(logits, Y_batch)
        total_loss += loss

        predictions = (1 / (1 + np.exp(-logits))) > 0.5
        accuracy = np.mean(predictions == Y_batch)
        testing_error_rate = 1 - accuracy
        total_accuracy += accuracy
        total_error_rate += testing_error_rate

        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = total_accuracy / num_batches
    avg_val_error_rate = total_error_rate / num_batches

    return avg_loss, accuracy, avg_val_error_rate

def params_accuracy_test(param_name, param_values,
                        X_train, Y_train,
                        X_test, Y_test,
                        input_dim, output_dim, num_layers):

    test_accuracies = []

    for value in param_values:
        if param_name == 'hidden_units':
            hidden_dims = value
            batch_size = 64
            step_size = 0.001
        elif param_name == 'batch_sizes':
            hidden_dims = 256
            batch_size = value
            step_size = 0.001
        elif param_name == 'step_sizes':
            hidden_dims = 256
            batch_size = 64
            step_size = value
        else:
            raise ValueError("Invalid param_name.")

        new_net = FeedForwardNeuralNetwork(input_dim, output_dim, hidden_dims, num_layers)

        max_epochs = 20
        momentum = 0.8
        weight_decay = 0.001
        for epoch in range(max_epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            loss_fn = SigmoidCrossEntropy()

            for i in range(0, X_train.shape[0], batch_size):
                # Gather batch
                X_batch = X_train[indices[i: i + batch_size]]
                Y_batch = Y_train[indices[i: i + batch_size]]

                # Compute forward pass
                logits = new_net.forward(X_batch)

                # Compute predictions & accuracy
                predictions = (1 / (1 + np.exp(-logits))) > 0.5
                accuracy = np.mean(predictions == Y_batch)

                # Compute loss
                loss = loss_fn.forward(logits, Y_batch)

                # Backward loss and networks
                loss_grad = loss_fn.backward()
                new_net.backward(loss_grad)

                # Take optimizer step
                new_net.step(step_size, momentum, weight_decay)

        # Evaluate performance on test.
        _, vacc, _ = evaluate(new_net, X_test, Y_test, batch_size)
        test_accuracies.append(vacc)

    for i in range(len(param_values)):
        print(f'{param_name}: {param_values[i]} -> accuracy: {test_accuracies[i]}')

    # Plot test accuracy
    plt.figure(figsize=(10,6))
    plt.plot(param_values, test_accuracies, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xticks(param_values, param_values)
    plt.xlabel(param_name)
    plt.ylabel('Test Accuracy(%)')
    plt.title(f'Test Accuracy vs {param_name}')
    plt.grid(True)
    plt.show()

def main():

    # TODO: Set optimization parameters (NEED TO SUPPLY THESE)
    batch_size = 16
    max_epochs = 20
    step_size = 0.001

    number_of_layers = 3
    width_of_layers = 128
    weight_decay = 0.05
    momentum = 0.8

    # Load data
    data = pickle.load(open('cifar_2class_py3.p', 'rb'))
    X_train = data['train_data']
    Y_train = data['train_labels']
    X_test = data['test_data']
    Y_test = data['test_labels']

    # Some helpful dimensions
    num_examples, input_dim = X_train.shape       # (10,000 , 3072)
    output_dim = 1 # number of class labels -1 for sigmoid loss

    # Data Normalization
    X_max_train = np.max(X_train, axis=0)
    X_train = (X_train - np.mean(X_train)) / X_max_train
    X_max_test = np.max(X_test, axis=0)
    X_test = (X_test - np.mean(X_test)) / X_max_test

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below. You can edit this as you please.
    net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    # Q2 TODO: For each epoch below max epochs
    for epoch in range(max_epochs):
        # Scramble order of examples
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        loss_fn = SigmoidCrossEntropy()
        total_train_loss = 0
        corrects = 0
        num_batches = 0

        # for each batch in data:
        for i in range(0, X_train.shape[0], batch_size):
            # Gather batch
            X_batch = X_train[indices[i : i + batch_size]]
            Y_batch = Y_train[indices[i : i + batch_size]]

            # Compute forward pass
            logits = net.forward(X_batch)

            # Compute predictions & accuracy
            predictions = (1 / (1 + np.exp(-logits))) > 0.5
            corrects += np.sum(predictions == Y_batch)

            # Compute loss
            loss = loss_fn.forward(logits, Y_batch)

            # Backward loss and networks
            loss_grad = loss_fn.backward()
            net.backward(loss_grad)

            # Take optimizer step
            net.step(step_size, momentum, weight_decay)

            # Accumulate loss
            total_train_loss += loss

            # Count the number of batches
            num_batches += 1

        # Book-keeping for loss / accuracy
        epoch_avg_loss = total_train_loss / num_batches
        epoch_avg_acc = corrects / num_examples
        losses.append(epoch_avg_loss)
        accs.append(epoch_avg_acc)
        epoch_train_error_rate = 1 - epoch_avg_acc

        # Evaluate performance on test.
        vloss, vacc, epoch_val_error_rate = evaluate(net, X_test, Y_test, batch_size)
        # print(f'tloss: {tloss:.4f}, vacc: {vacc:.4f}')
        val_losses.append(vloss)
        val_accs.append(vacc)

        ###############################################################
        # Print some stats about the optimization process after each epoch
        ###############################################################
        # epoch_avg_loss -- average training loss across batches this epoch
        # epoch_avg_acc -- average accuracy across batches this epoch
        # vacc -- testing accuracy this epoch
        ###############################################################

        logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%       Train error rate:  {:8.4}%       Val error rate:  {:8.4}%".format(epoch, epoch_avg_loss, epoch_avg_acc*100, vacc*100, epoch_train_error_rate*100, epoch_val_error_rate*100))

    # Find maximum accuracy
    max_vacc = np.max(val_accs)
    print(f'max epochs: {max_epochs}, batch size: {batch_size}, learning rate: {step_size}')
    print(f'number of hidden units: {width_of_layers}, momentum: {momentum}, weight decay: {weight_decay}')
    print(f'max validation accuracy: {max_vacc*100}%')
    ###############################################################
    # Code for producing output plot requires
    ###############################################################
    # losses -- a list of average loss per batch in training
    # accs -- a list of accuracies per batch in training
    # val_losses -- a list of average testing loss at each epoch
    # val_acc -- a list of testing accuracy at each epoch
    # batch_size -- the batch size
    ################################################################

    # Plot training and testing curves
    fig, ax1 = plt.subplots(figsize=(16,9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    # ax1.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_losses))], val_losses,c="red", label="Val. Loss")
    ax1.plot(range(len(val_losses)), val_losses, c="red", label="Val. Loss")
    ax1.set_xticks(range(max_epochs))
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    #ax1.set_ylim(-0.01,3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'

    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    # ax2.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_accs))], val_accs,c="blue", label="Val. Acc.")
    ax2.plot(range(len(val_losses)), val_accs, c="blue", label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01,1.01)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.show()

    ############################################################
    # Q5 Tuning Parameters: batch size, step size, hidden units
    ############################################################
    hidden_dims = [32, 64, 128, 256, 512]
    params_accuracy_test('hidden_units', hidden_dims,
                             X_train, Y_train,
                             X_test, Y_test,
                             input_dim, output_dim, number_of_layers)
    step_sizes = [0.001, 0.005, 0.01, 0.05, 0.1]
    params_accuracy_test('step_sizes', step_sizes,
                         X_train, Y_train,
                         X_test, Y_test,
                         input_dim, output_dim, number_of_layers)
    batch_sizes = [16, 32, 64, 128, 256]
    params_accuracy_test('batch_sizes', batch_sizes,
                         X_train, Y_train,
                         X_test, Y_test,
                         input_dim, output_dim, number_of_layers)


#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################
class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):

        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        else:
            # TODO: Please create a network with hidden layers based on the parameters
            self.layers = []

            # Input layer
            self.layers.append(LinearLayer(input_dim, hidden_dim))
            self.layers.append(ReLU())

            # Hidden layers
            for _ in range(num_layers-2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
                self.layers.append(ReLU())

            # Output layer
            self.layers.append(LinearLayer(hidden_dim, output_dim))
            self.loss_fn = SigmoidCrossEntropy()

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size, momentum, weight_decay):
        for layer in self.layers:
            layer.step(step_size, momentum, weight_decay)


def displayExample(x):
    r = x[:1024].reshape(32,32)
    g = x[1024:2048].reshape(32,32)
    b = x[2048:].reshape(32,32)

    plt.imshow(np.stack([r,g,b],axis=2))
    plt.axis('off')
    plt.show()


if __name__=="__main__":
    main()





