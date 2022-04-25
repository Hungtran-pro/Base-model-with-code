#code without sklearn
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11) # makes the random numbers predictable. But have to set it before every use!

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]] #covariance matrix
N = 500
#generate N examples from the means with the covariance matrix
X0 = np.random.multivariate_normal(means[0], cov, N) 
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate([X0, X1, X2], axis=0) #concatinate matrix with axis = 0 | 1 | None (flatern the matrix)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T #Convert the input to an numpy array

def kmeans_display(X, label):
    K = np.amax(label) + 1 #Caculate the maximum value of the input. Can set with 'axis' the maximum is selected over multiple axes
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()


kmeans_display(X, original_label) #display

def kmeans_init_center(X, k):
    #randomly pick k rows of X as initial centers.
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    #calculate pairwise distance btw data and centeres
    D = cdist(X, centers)
    #return index of the closet center
    return np.argmin(D, axis=1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        #collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        #take average
        centers[k, :]  = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    #return True if two sets of centers are the same
    return (set([tuple(a) for a in centers])) == (set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [kmeans_init_center(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(new_centers, centers[-1]):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print('Centers found by Kmeans algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])

#code with sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_labels = kmeans.predict(X)
kmeans_display(X, pred_labels)
kmeans_display(X, kmeans.labels_)
print((pred_labels == kmeans.labels_).all())