import math
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(177)

# CONSTANTS
BS = 100
LR = 0.001
num_neurons = 1024  # will be 1024
pixels = 784
epochs = 50


def read_labels(file_name):
    file = open(file_name, "rb")
    magic_number = int.from_bytes(file.read(4), "big")
    # print(magic_number)
    num_items = int.from_bytes(file.read(4), "big")
    print(num_items)

    labels = []

    for i in range(num_items):
        labels.append(int.from_bytes(file.read(1), "big"))

    file.close()
    # for i in range(len(labels)):
    #     print("label " + str(i) + " is " + str(labels[i]))

    return labels


def read_image(file_name):
    file = open(file_name, "rb")
    magic_number = int.from_bytes(file.read(4), "big")
    # print(magic_number)
    num_items = int.from_bytes(file.read(4), "big")
    print(num_items)
    num_rows = int.from_bytes(file.read(4), "big")
    # print(num_rows)
    num_cols = int.from_bytes(file.read(4), "big")
    # print(num_cols)

    print(num_cols * num_rows)

    images = [[0 for i in range(num_cols * num_rows)]
              for k in range(num_items)]

    for i in range(num_items):
        for j in range(num_cols * num_rows):
            temp = int.from_bytes(file.read(1), "big")
            images[i][j] = temp

    file.close()

    return images


#############################################################
# DATA
trainLabels = read_labels("train-labels.idx1-ubyte")

trainImages = read_image("train-images.idx3-ubyte")

testLabels = read_labels("t10k-labels.idx1-ubyte")

testImages = read_image("t10k-images.idx3-ubyte")

data = np.array(trainImages) / 255.0  # 60k x 784
data2 = np.array(testImages) / 255.0  # 10k x 784
m, n = data.shape  # m is the number of samples, n is the number of pixels
num_images = m  # N = 783, M= 60k
sol = np.array(trainLabels)  # sol is the solution vector of 0's - 9's
sol = np.reshape(sol, (1, sol.size)).T  # turning sol from a list to a vector
sol2 = np.array(testLabels)  # sol is the solution vector of 0's - 9's
# turning sol from a list to a vector
sol2 = np.reshape(sol2, (1, sol2.size)).T
# concatinate the image array with the solution vector so that the image is paired with a solution
stuff = np.concatenate((data, sol), axis=1)
stuff2 = np.concatenate((data2, sol2), axis=1)  # same for test data
# add a new col with all of the labels -> row 1 of stuff = [image1|sol]
trainData = stuff.T  # 785 x 60k
testData = stuff2.T  # 785 x 10k



#############################################################


def init_params():
    # create a 1024xpixels weight matrix - 1st layer of a uniform distribution from -0.5 to 0.5
    W1 = np.random.uniform(-0.5, 0.5, (num_neurons, pixels))
    # 1024x1 bias matrix - 1st layer
    b1 = np.random.uniform(-0.5, 0.5, (num_neurons, 1))
    # 10x1024 weight matrix - 3rd layer (output layer)
    W2 = np.random.uniform(-0.5, 0.5, (10, num_neurons))
    # 10x1 bias matrix - 3rd layer (output layer)
    b2 = np.random.uniform(-0.5, 0.5, (10, 1))
    return W1, b1, W2, b2


def ReLU(X):
    return np.maximum(X, 0)  # ReLU function


def softmax(Z):
    e = np.exp(Z)
    return e / np.sum(e, axis=0)  # softmax function


def forward_prop(X, W1, W2, b1, b2):
    # ROW x COL
    # X = 784 x BATCHSIZE, each column = 1 image;
    # W1 = 1024 x 784; each row is one node and the columns are the connections to the previous layer for that node
    # W2 = 10 x 1024; each row is one node in L2 and the columns are the connections to the previous layer for that node
    # b1 = 1024 x 1, each row is one neuron's bias (L1)
    # b2 = 10 x 1, each row is one neuron's bias (L2)
    Z1 = np.dot(W1, X) + b1  # Z1 = W1*X + b1; 1024 x BATCHSIZE
    A1 = ReLU(Z1)  # A1 is the ReLU transformation of Z1 1024 x BATCHSIZE
    Z2 = np.dot(W2, A1) + b2 # 10 x BATCHSIZE ; 10 is nodes in L2
    A2 = softmax(Z2) # 10 x BATCHSIZE
    return Z1, Z2, A1, A2


def deriv_ReLU(X):
    return X > 0  # derivative of ReLU function


def one_hot(Y):
    # Y consists of all the labels in 1 column => ith row is the ith solution
    one_hot_Y = np.zeros((Y.size, 10))  # one hot encode the solution; Num_IMAGES x 10; each sub array is size 10
    for i in range(Y.size):
        one_hot_Y[i][int(Y[i])] = 1 # for the ith row of result, set the Y[i] position to 1
    return one_hot_Y.T  # return row column


def back_prop(Z1, A1, Z2, A2, W2, X, Ys, batchsize):
    # Z1 is 1024 x batchsize
    # A1 is ''''
    # A1T is 100 x 1024
    # Z2 is 10 x 100 , not used
    # A2 is 10 x 100
    # W2 is 10 x 1024
    # X is 784 x batchsize
    # Ys is 10 x 100

    dZ2 = A2 - Ys  # computing the derivative of each layer based of equations
    dW2 = 1 / batchsize * dZ2 @ A1.T  # dot product alt notation
    # 1/batchsize [ entry i j is (row i of dz2 matched up with col j of A1T (summation))]
    db2 = 1 / batchsize * np.sum(dZ2, 1) # sum along columns // sum of each vector becomes an entry => total entries = num vectors
    # to keep as numpy vector (prev line turns npvect -> python list) ; (obj to reshape, [1 row, sizeof(obj) cols]).transpose
    db2 = np.reshape(db2, (1, db2.size)).T # this should be  rows 10 x 1 col
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1) # 1024 x 100
    dW1 = 1 / batchsize * dZ1.dot(X.T) # 1024 x 784
    db1 = 1 / batchsize * np.sum(dZ1, 1)
    db1 = np.reshape(db1, (1, db1.size)).T  # should be 1024 x 1

    # dZ2 is size of A2/Ys = 10 x 100
    # dW2 is size 10 x 1024 = dz2 dot A1T
    return dW2, db2, dW1, db1


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # W1 is 1024 x 784
    # W2 is 10 x 1024
    # b1 is 1024 x 1
    # b2 is 10 x 1
    # dW1 1024 x 784
    # db1 is 1024 x 1
    # db2 is 10 x 1
    # alpha is a number

    # vector scaling and then subtraction
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1  # recompute the new weights and bias
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    # a2 is for the entire set -> [2(M/F) by number of images ]
    # get the position of the maximum argument of every column (1 for row)
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    count = 0
    for i in range(predictions.size):  # over num images (all predictions)
        if Y[predictions[i], i] == 1:  # compares the prediction with the actual output
            count += 1  # prediction correct
    return count / predictions.size


### MAIN ####
def run(alpha, epochs, batchsize):
    W1, b1, W2, b2 = init_params()
    trainAcc = []
    testAcc = []
    trainLoss = []
    testLoss = []
    one_hot_Y = None
    for i in range(epochs):
        # shuffles data before the iteration begins
        np.random.shuffle(trainData.T)
        # one hot encode the solution from the solution row of the train data matrix
        one_hot_Y = one_hot(trainData[pixels])  # 10 x 60k -> the jth column is the jth image's solution
        print("Epoch:", i, "/", epochs)
        # accuracy derivation
        Z1, Z2, A1, A2 = forward_prop(trainData[:pixels, :], W1, W2, b1, b2)
        trainAcc.extend([get_accuracy(get_predictions(A2), one_hot_Y[:, :])])
        trainLoss.extend([np.sum(1 / 2 * (A2 - one_hot_Y[:, :]) ** 2) / np.size(A2)])
        Z1, Z2, A1, A2, = forward_prop(
            testData[:pixels, :], W1, W2, b1, b2)  # useless z1 z2 a1
        testAcc.extend(
            [get_accuracy(get_predictions(A2), one_hot(testData[pixels]))])
        # loss function's value (==cost) => changes to cross-entropy
        testLoss.extend(
            [np.sum(1 / 2 * (A2 - one_hot(testData[pixels])) ** 2) / np.size(A2)])

        print(f"{'Train Accuracy:':<15}{round(trainAcc[-1], 4):>10}",
              f"\n{'Test Accuracy:':<15}{testAcc[-1]:>10}")

        # end accuracy pre epoch
        for i in range(math.floor(  # go over the entire shuffled data set; entire for loop = 1 epoch
                num_images / batchsize)):  # number of loops is determined by num_images(size of test data) divided by the size of batch
            j = i * batchsize
            # j through k are the current batch being tested
            k = batchsize * (i + 1)
            # forward prop computed with every layer output saved
            Z1, Z2, A1, A2 = forward_prop(
                trainData[:pixels, j:k], W1, W2, b1, b2)
            # drop out mask with p = 0.40
            U1 = (np.random.rand(*A1.shape) < 0.4) / 1 # generate an array of shape A1.shape (1024, 100)
            # call back prop function, all rows; j through k cols
            dW2, db2, dW1, db1 = back_prop(
                Z1, A1 * U1, Z2, A2, W2, trainData[:pixels, j:k], one_hot_Y[:, j:k], batchsize)
            W1, b1, W2, b2 = update_params(
                W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)  # update the parameters

    # final accuracy derivation
    Z1, Z2, A1, A2 = forward_prop(trainData[:pixels, :], W1, W2, b1, b2)
    trainAcc.extend([get_accuracy(get_predictions(A2), one_hot_Y[:, :])]) # appends list
    trainLoss.extend(
        [np.sum(1 / 2 * (A2 - one_hot_Y[:, :]) ** 2) / np.size(A2)])
    Z1, Z2, A1, A2 = forward_prop(testData[:pixels, :], W1, W2, b1, b2)
    testAcc.extend(
        [get_accuracy(get_predictions(A2), one_hot(testData[pixels]))])
    testLoss.extend(
        [np.sum(1 / 2 * (A2 - one_hot(testData[pixels])) ** 2) / np.size(A2)])
    return trainAcc, testAcc, trainLoss, testLoss


def main():
    trainAcc, testAcc, trainLoss, testLoss = run(
        LR, epochs, BS)  # start the gradient descent function
    fig, axs = plt.subplots(2)
    print("Final Accuracy Train:", trainAcc[-1], "Test:", testAcc[-1])
    print("Max Accuracy Train:", max(trainAcc), "Test:", max(testAcc))

    # plottting
    axs[0].set_title('Accuracy Graph')
    axs[0].plot(trainAcc, 'r-', label='Train Accuracy')
    axs[0].plot(testAcc, 'b-', label='Test Accuracy')
    axs[0].grid()
    axs[1].set_title('Loss Graph')
    axs[1].plot(trainLoss, 'r-', label='Train Loss')
    axs[1].plot(testLoss, 'b-', label='Test Loss')
    axs[1].grid()
    axs[0].legend()
    axs[1].legend()
    plt.show()


main()
