import numpy as np

def im2col(X, kernel_size, stride, pad):
    N, H, W, C = X.shape
    K = kernel_size
    S = stride
    P = pad

    H_out = (H + 2*P - K) // S + 1
    W_out = (W + 2*P - K) // S + 1

    X_pad = np.pad(X, ((0,0), (P,P), (P,P), (0,0)), mode='constant')
    strides = (X_pad.strides[0], S*X_pad.strides[1], S*X_pad.strides[2], X_pad.strides[1], X_pad.strides[2], X_pad.strides[3])
    shape = (N, H_out, W_out, K, K, C)
    windows = np.lib.stride_tricks.as_strided(X_pad, shape=shape, strides=strides)
    cols = windows.transpose(0, 1, 2, 4, 5, 3).reshape(N * H_out * W_out, -1).T
    return cols

def col2im(cols, X_shape, kernel_size, stride, pad):
    N, H, W, C = X_shape
    K = kernel_size
    S = stride
    P = pad
    H_out = (H + 2*P - K) // S + 1
    W_out = (W + 2*P - K) // S + 1

    cols_reshaped = cols.T.reshape(N, H_out, W_out, K, K, C)
    X_pad = np.zeros((N, H + 2*P, W + 2*P, C))

    for i in range(H_out):
        for j in range(W_out):
            y_start = i * S
            x_start = j * S
            y_end = y_start + K
            x_end = x_start + K
            X_pad[:, y_start:y_end, x_start:x_end, :] += cols_reshaped[:, i, j, :, :, :]

    if P != 0:
        X_grad = X_pad[:, P:-P, P:-P, :]
    else:
        X_grad = X_pad
    return X_grad

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self.W = np.random.randn(out_channels, kernel_size, kernel_size, in_channels) * 0.1
        self.b = np.zeros(out_channels)
        self.stride = stride
        self.pad = pad
        self.kernel_size = kernel_size
        self.cache = None

    def forward(self, X):
        N, H, W, C_in = X.shape
        K = self.kernel_size
        S = self.stride
        P = self.pad
        H_out = (H + 2*P - K) // S + 1
        W_out = (W + 2*P - K) // S + 1

        X_col = im2col(X, K, S, P)
        W_row = self.W.reshape(self.W.shape[0], -1)

        out = W_row @ X_col + self.b.reshape(-1, 1)
        out = out.reshape(self.W.shape[0], N, H_out, W_out).transpose(1, 2, 3, 0)
        self.cache = (X, X_col, W_row)
        return out

    def backward(self, dout):
        X, X_col, W_row = self.cache
        N, H, W, C_in = X.shape
        K = self.kernel_size
        S = self.stride
        P = self.pad

        dout_reshaped = dout.transpose(3, 1, 2, 0).reshape(self.W.shape[0], -1)
        dW = dout_reshaped @ X_col.T
        self.dW = dW.reshape(self.W.shape)
        self.db = np.sum(dout, axis=(0, 1, 2))

        dX_col = W_row.T @ dout_reshaped
        dX = col2im(dX_col, X.shape, K, S, P)
        return dX

class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, dout):
        X = self.cache
        return dout * (X > 0)

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, X):
        N, H, W, C = X.shape
        K = self.pool_size
        S = self.stride
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        windows = np.lib.stride_tricks.as_strided(
            X, 
            shape=(N, H_out, W_out, K, K, C),
            strides=(X.strides[0], S*X.strides[1], S*X.strides[2], X.strides[1], X.strides[2], X.strides[3])
        )
        windows_reshaped = windows.reshape(-1, K*K)
        max_values = np.max(windows_reshaped, axis=1)
        max_indices = np.argmax(windows_reshaped, axis=1)
        output = max_values.reshape(N, H_out, W_out, C)
        self.cache = (X.shape, max_indices, K, S)
        return output

    def backward(self, dout):
        X_shape, max_indices, K, S = self.cache
        N, H, W, C = X_shape
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        dX = np.zeros(X_shape)
        dY_flat = dout.flatten()

        idx = np.arange(N * H_out * W_out * C)
        n = idx // (H_out * W_out * C)
        remainder = idx % (H_out * W_out * C)
        i = remainder // (W_out * C)
        remainder = remainder % (W_out * C)
        j = remainder // C
        c = remainder % C

        y_start = i * S
        x_start = j * S
        window_y = max_indices[idx] // K
        window_x = max_indices[idx] % K

        y = y_start + window_y
        x = x_start + window_x

        np.add.at(dX, (n, y, x, c), dY_flat)
        return dX

class Flatten:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        self.cache = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache)

class Dense:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.b = np.zeros(out_features)
        self.cache = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, dout):
        X = self.cache
        self.dW = X.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = None

    def forward(self, X, y):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        probs = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        self.cache = (probs, y)
        N = X.shape[0]
        return -np.sum(np.log(probs[np.arange(N), y] + 1e-8)) / N

    def backward(self):
        probs, y = self.cache
        N = y.shape[0]
        dX = probs.copy()
        dX[np.arange(N), y] -= 1
        return dX / N

class CNN:
    def __init__(self):
        self.layers = [
            Conv2D(1, 6, 5, 1, 0),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(6, 16, 5, 1, 0),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(16*4*4, 120),
            ReLU(),
            Dense(120, 84),
            ReLU(),
            Dense(84, 10)
        ]
        self.loss = SoftmaxCrossEntropy()

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def compute_loss(self, X, y):
        logits = self.forward(X)
        return self.loss.forward(logits, y)

    def backward_loss(self):
        dout = self.loss.backward()
        self.backward(dout)

    def update_params(self, lr=0.001):
        for layer in self.layers:
            if isinstance(layer, (Conv2D, Dense)):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db