import numpy as np
import time
import matplotlib.pyplot as pl

class LogisticRegression:
    
    def __init__(self, regularization, k, n, method, lr, max_iter=5000):
        self.k = k
        self.n = n
        self.alpha = lr
        self.max_iter = max_iter
        self.method = method
        self.regularization = regularization
    
    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                #if i % 500 == 0:
                    #print(f"Loss at iteration {i}", loss)
            #print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                #if i % 500 == 0:
                    #print(f"Loss at iteration {i}", loss)
            #print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "stochastic":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                #if i % 500 == 0:
                    #print(f"Loss at iteration {i}", loss)
            #print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
        
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y*np.log(h)) / m + self.regularization(self.W) /(2*m)
        error = h - Y
        grad = self.softmax_grad(X, error)/m +self.regularization.derivation(self.W)/m
        return loss, grad
    


    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

    # Add functions accuracy, precision, recall and f1

    def accuracy(self, X_test, y_true):
        y_pred = self.predict(X_test)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_true)
    

    def precision(self, X_test, y_true):
        y_pred = self.predict(X_test)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        precisions = {}
        for cls in range(self.k):
            TP = np.sum((y_pred == cls) & (y_true == cls))
            FP = np.sum((y_pred == cls) & (y_true != cls))
            precisions[cls] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        return precisions
    
    def recall(self, X_test, y_true):
        y_pred = self.predict(X_test)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        recalls = {}
        for cls in range(self.k):
            TP = np.sum((y_pred == cls) & (y_true == cls))
            FN = np.sum((y_pred != cls) & (y_true == cls))
            recalls[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return recalls
    

    def f1_score(self, X_test, y_true):
        prec = self.precision(X_test, y_true)
        rec = self.recall(X_test, y_true)
        
        f1_scores = {}
        for cls in range(self.k):
            p, r = prec[cls], rec[cls]
            f1_scores[cls] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return f1_scores
    
    # Add functions macro precision, macro recall and macro f1
    
    def macro_precision(self, X_test, y_true):
        precisions = self.precision(X_test, y_true)  # dict {class: precision}
        return np.mean(list(precisions.values()))

    def macro_recall(self, X_test, y_true):
        recalls = self.recall(X_test, y_true)  # dict {class: recall}
        return np.mean(list(recalls.values()))

    def macro_f1(self, X_test, y_true):
        f1_scores = self.f1_score(X_test, y_true)  # dict {class: f1}
        return np.mean(list(f1_scores.values()))
    
    # Add weighted matrices

    def weighted_precision(self, X_test, y_true):
        # convert class labels to integers
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        precisions = self.precision(X_test, y_true)
        weights = [(y_true == cls).sum() / len(y_true) for cls in range(self.k)]
        return sum(weights[cls] * precisions[cls] for cls in range(self.k))

    def weighted_recall(self, X_test, y_true):
        # convert class labels to integers
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        recalls = self.recall(X_test, y_true)
        weights = [(y_true == cls).sum() / len(y_true) for cls in range(self.k)]
        return sum(weights[cls] * recalls[cls] for cls in range(self.k))
    
    def weighted_f1(self, X_test, y_true):
        # convert class labels to integers
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        f1_scores = self.f1_score(X_test, y_true)
        weights = [(y_true == cls).sum() / len(y_true) for cls in range(self.k)]
        return sum(weights[cls] * f1_scores[cls] for cls in range(self.k))
    



class Ridge:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class RidgeRegression(LogisticRegression):
    def __init__(self, k, n, method="minibatch", lr=0.01, l=0.001):
        self.regularization = Ridge(l)
        super().__init__(self.regularization, k, n, method, lr)
