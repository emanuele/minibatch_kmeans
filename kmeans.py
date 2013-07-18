import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

def kmeans(X, C):
    V = np.zeros(C.shape[0])
    for x in X:
        # idx = np.argmin(((C - x)*(C - x)).sum(1))
        idx = np.argmin(((C - x)**2).sum(1))
        # idx = np.argmin(pairwise_distances(C, x))
        V[idx] += 1
        eta = 1.0 / V[idx]
        C[idx] = (1.0 - eta) * C[idx] + eta * x
        
    return C
    

def mini_batch_kmeans(X, C, b, t):
    for i in range(t):
        # Sample a mini batch:
        X_batch = X[np.random.permutation(X.shape[0])[:b]]
        V = np.zeros(C.shape[0])
        idxs = np.empty(X_batch.shape[0])
        # Assign the closest centers without update for the whole batch:
        for j, x in enumerate(X_batch):
            # idxs[j] = np.argmin(((C - x)*(C - x)).sum(1))
            idxs[j] = np.argmin(((C - x)**2).sum(1))
        # idxs = np.argmin(pairwise_distances(C, X_batch), axis=0)

        # Update centers
        for j, x in enumerate(X_batch):
            V[idxs[j]] += 1
            eta = 1.0 / V[idxs[j]]
            C[idxs[j]] = (1.0 - eta) * C[idxs[j]] + eta * x

    return C


if __name__ == '__main__':

    np.random.seed(1)

    n = 10000
    d = 2
    X, y = make_blobs(n, d, centers=3)
    plot(X[:,0], X[:,1], 'ko')
    
    k = 3

    # In case we want to permute the order of X:
    # np.random.seed(0)
    # X = np.random.permutation(X)

    C_start = X[:k]
    plot(C_start[:,0], C_start[:,1], 'bo', markersize=10, label='initialization')

    C = kmeans(X, C_start)

    plot(C[:,0], C[:,1], 'ro', markersize=10, label='k-means')

    C_mbkm = mini_batch_kmeans(X, C_start, b=50, t=10)
    plot(C_mbkm[:,0], C_mbkm[:,1], 'go', markersize=10, label='mini-batch k-means')

    plt.legend(numpoints=1, loc='lower right')
