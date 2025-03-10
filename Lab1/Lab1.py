import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontdict = {'fontsize' : 18})
    for i in range(x.shape[0]):
        if y[i, 0] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontdict = {'fontsize' : 18})
    for i in range(x.shape[0]):
        if pred_y[0, i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

def MSE(A3, Y):
    return np.mean((A3 - Y.T) ** 2)

def derivative_MSE(A3, Y):
    m = Y.shape[0]
    return (2/m)*(A3-Y.T)

def BCE(A3, Y):
    return -np.mean(Y.T * np.log(A3 + 1e-8) + (1 - Y.T) * np.log(1 - A3 + 1e-8))

def derivative_BCE(A3, Y):
    return (A3 - Y.T) / (A3 * (1 - A3) + 1e-8)

# forward pass
def forward(X):
    global Z1, A1, Z2, A2, Z3, A3
    Z1 = W1 @ X.T + b1 # @ means dot product
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)
    return A3

# backward pass
def backward(X, Y, A3, loss_type="BCE"):
    global W1, b1, W2, b2, W3, b3, lr

    m = X.shape[0]

    # loss func. derivative
    if loss_type == "BCE":
        dL_dA3 = derivative_BCE(A3, Y) # (1, m)
    elif loss_type == "MSE":
        dL_dA3 = derivative_MSE(A3, Y) # (1, m)
    dA3_dZ3 = derivative_sigmoid(A3)  # (1, m)

    # output layer neuron err equals to Z3's gradient
    dZ3 = dL_dA3 * dA3_dZ3  # (1, m)
    
    # calculate W3 & Z3's gradient
    dW3 = (dZ3 @ A2.T) / m  # (1, h2)
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m  # (1, 1)

    # layer2
    dA2_dZ2 = derivative_sigmoid(A2)  # (h2, m)
    dZ2 = (W3.T @ dZ3) * dA2_dZ2  # (h2, m)
    dW2 = (dZ2 @ A1.T) / m  # (h2, h1)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # (h2, 1)

    # layer1
    dA1_dZ1 = derivative_sigmoid(A1)  # (h1, m)
    dZ1 = (W2.T @ dZ2) * dA1_dZ1  # (h1, m)
    dW1 = (dZ1 @ X) / m  # (h1, 2)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # (h1, 1)

    # gradient decent
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3

def train(X_train, Y_train, epochs=100000, loss_type="BCE"):
    loss_history = []
    for i in range(epochs):
        A3 = forward(X_train)
        loss = BCE(A3, Y_train)
        loss_history.append(loss)
        backward(X_train, Y_train, A3, loss_type)

        if i % 500 == 0:
            if loss_type == "BCE":
                loss = BCE(A3, Y_train)
            elif loss_type == "MSE":
                loss = MSE(A3, Y_train)
            print(f"Epoch {i}, Loss: {loss:.4f}")
    # loss-epoch curve
    plt.plot(range(epochs), loss_history, label=f"{loss_type} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve ({loss_type} Loss)")
    plt.legend()
    plt.show()

def test(X_test, Y_test, loss_type="BCE"):
    A3 = forward(X_test)
    predictions = (A3 > 0.5).astype(int)
    correct = 0
    m = X_test.shape[0]  

    for i in range(m):
        ground_truth = Y_test[i, 0] 
        pred_value = A3[0, i] 

        if (pred_value > 0.5 and ground_truth == 1) or (pred_value <= 0.5 and ground_truth == 0):
            correct += 1
        print(f"Iter{i:02d} | Ground truth: {ground_truth:.1f} | prediction: {pred_value:.5f}")

    if loss_type == "BCE":
        loss = BCE(A3, Y_train)
    elif loss_type == "MSE":
        loss = MSE(A3, Y_train)
    accuracy = (correct / m) * 100
    print(f"loss={loss:.5f} accuracy={accuracy:.2f}%")

    return predictions

np.random.seed(42)

# weight initial
h1, h2 = 32, 32# hidden units
# learning rate
lr = 0.05
# three linear func. cause two hidden layers
W1 = np.random.randn(h1, 2)
b1 = np.zeros((h1, 1))
W2 = np.random.randn(h2, h1)
b2 = np.zeros((h2, 1))
W3 = np.random.randn(1, h2)
b3 = np.zeros((1, 1))

X_train, Y_train = generate_linear(n=100)
X_test, Y_test = generate_linear(n=100)

#X_train, Y_train = generate_XOR_easy()
#X_test, Y_test = generate_XOR_easy()
train(X_train, Y_train, epochs=5000, loss_type="BCE") # loss_type = BCE or MSE
pred_y = test(X_test, Y_test, loss_type="BCE")
show_result(X_test, Y_test, pred_y)