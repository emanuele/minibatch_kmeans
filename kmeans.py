import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree


def kmeans(X, C):
    """The Loyd's algorithm for the k-centers problems.

    X : data matrix
    C : initial centers
    """
    C = C.copy()
    V = np.zeros(C.shape[0])
    for x in X:
        idx = np.argmin(((C - x)**2).sum(1))
        V[idx] += 1
        eta = 1.0 / V[idx]
        C[idx] = (1.0 - eta) * C[idx] + eta * x

    return C


def mini_batch_kmeans(X, C, b, t, replacement=True):
    """The mini-batch k-means algorithms (Sculley et al. 2007) for the
    k-centers problem.

    X : data matrix
    C : initial centers
    b : size of the mini-batches
    t : number of iterations
    replacement: whether to sample batches with replacement or not.
    """
    C = C.copy()
    for i in range(t):
        # Sample a mini batch:
        if replacement:
            X_batch = X[np.random.permutation(X.shape[0])[:b]]
        else:
            X_batch = X[b*t:b*(t+1)]

        V = np.zeros(C.shape[0])
        idxs = np.empty(X_batch.shape[0], dtype=np.int)
        # Assign the closest centers without update for the whole batch:
        for j, x in enumerate(X_batch):
            idxs[j] = np.argmin(((C - x)**2).sum(1))

        # Update centers:
        for j, x in enumerate(X_batch):
            V[idxs[j]] += 1
            eta = 1.0 / V[idxs[j]]
            C[idxs[j]] = (1.0 - eta) * C[idxs[j]] + eta * x

    return C


def compute_labels(X, C):
    """Compute the cluster labels for dataset X given centers C.
    """
    # labels = np.argmin(pairwise_distances(C, X), axis=0) # THIS REQUIRES TOO MUCH MEMORY FOR LARGE X
    tree = KDTree(C)
    labels = tree.query(X, k=1, return_distance=False).squeeze()
    return labels


def compute_centroids(X, C):
    """Compute the centroids for dataset X given centers C. Note: centers
    C may not belong to X.
    """
    tree = KDTree(X)
    centroids = tree.query(C, k=1, return_distance=False).squeeze()
    return centroids


if __name__ == '__main__':

    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score

    np.random.seed(1)

    n = 10000
    d = 2
    X, y = make_blobs(n, d, centers=3)
    plt.plot(X[:,0], X[:,1], 'ko')

    k = 3

    # In case we want to permute the order of X:
    # np.random.seed(0)
    # X = np.random.permutation(X)

    C_init = X[:k]
    plt.plot(C_init[:,0], C_init[:,1], 'bo', markersize=10, label='initialization')

    C_kmeans = kmeans(X, C_init)

    plt.plot(C_kmeans[:,0], C_kmeans[:,1], 'ro', markersize=10, label='k-means')

    b = 50
    t = 10

    C_mbkm = mini_batch_kmeans(X, C_init, b=b, t=t, replacement=True)
    plt.plot(C_mbkm[:,0], C_mbkm[:,1], 'go', markersize=10, label='mini-batch k-means')

    C_mbkm_wr = mini_batch_kmeans(X, C_init, b=b, t=t, replacement=False)
    plt.plot(C_mbkm_wr[:,0], C_mbkm_wr[:,1], 'mo', markersize=10, label='mini-batch k-means w/o rep.')

    # from sklearn.cluster import MiniBatchKMeans
    # mbkm_skl = MiniBatchKMeans(n_clusters=k, max_iter=1, max_no_improvement=None, tol=0.0, batch_size=b, init=C_init, compute_labels=False)
    # mbkm_skl.fit(X)
    # C_mbkm_skl = mbkm_skl.cluster_centers_
    # plt.plot(C_mbkm_skl[:,0], C_mbkm_skl[:,1], 'co', markersize=10, label='mini-batch k-means SKL')

    plt.legend(numpoints=1, loc='lower right')

    labels_init = compute_labels(X, C_init)
    labels_kmeans = compute_labels(X, C_kmeans)
    labels_mbkm = compute_labels(X, C_mbkm)
    labels_mbkm_wr = compute_labels(X, C_mbkm_wr)
    print "Adjusted rand scores:"
    print "labels_kmeans, labels_init =", adjusted_rand_score(labels_kmeans, labels_init)
    print "labels_kmeans, labels_mbkm =", adjusted_rand_score(labels_kmeans, labels_mbkm)
    print "labels_kmeans, labels_mbkm_wr =", adjusted_rand_score(labels_kmeans, labels_mbkm_wr)

    plt.show()
