import numpy as np


class KMeans(object):
    """
    KMeans clustering algorithm.
    """

    def __init__(self, max_iters=500, k=5, K=None):
        """
        Initialize KMeans with max iterations.
        """
        if K is not None:
            k = K
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None
    
    def __init_centers(self, data):
        """
        Randomly pick K data points from the data as initial cluster centers.
        
        Arguments: 
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        centers = np.random.permutation(data)[0:self.k]
        return centers
    
    @staticmethod
    def __compute_distance(data, centers):
        """
        Compute the Euclidean distance between each data point and each center.
        
        Arguments:    
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        N = data.shape[0]
        K = centers.shape[0]
        distances = np.zeros((N, K))

        for k in range(K):
            difference = data - centers[k, :]  # [k, :] -> current center
            distances[:, k] = np.sqrt(np.sum(difference ** 2, axis=1))    
            
        return distances
    
    @staticmethod
    def __find_closest_cluster(distances):
        """
        Assign datapoints to the closest clusters.
        
        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def __compute_centers(self, data, cluster_assignments):
        """
        Compute the new centers by averaging the points in each cluster.
        
        Arguments:
            data: array of shape (N, D), the data points.
            cluster_assignments: array of shape (N,), the current cluster assignments.
            K: int, number of clusters.
        Returns:
            centers: array of shape (K, D), the new centers.
        """
        centers = np.zeros((self.k, data.shape[1]))  # Initialize centers
        for k in range(self.k):
            # Get all points that belong to cluster k
            points_in_cluster = data[cluster_assignments == k]
            if len(points_in_cluster) == 0:
                centers[k] = data[np.random.randint(data.shape[0])]
            else:
                centers[k] = np.mean(points_in_cluster, axis=0)  # Compute the mean of the points in this cluster
        return centers

    def __k_means(self, data, max_iter):
        """
        Main function that combines all the former functions together to build the K-means algorithm.
        
        Arguments: 
            data: array of shape (N, D) where N is the number of data samples, D is number of features.
            K: int, the number of clusters.
            max_iter: int, the maximum number of iterations.
        Returns:
            centers: array of shape (K, D), the final cluster centers.
            cluster_assignments: array of shape (N,), final cluster assignment for each data point.
        """
        # Initialize the centers
        centers = self.__init_centers(data)

        # Loop over the iterations
        for i in range(max_iter):
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{max_iter}...")
            
            old_centers = centers.copy()  # Keep in memory the centers of the previous iteration
            distances = self.__compute_distance(data, centers)
            cluster_assignments = self.__find_closest_cluster(distances)
            centers = self.__compute_centers(data, cluster_assignments)

            # End of the algorithm if the centers have not moved
            if np.all(centers == old_centers):
                print(f"K-Means has converged after {i+1} iterations!")
                break
        
        # Return the final centers and cluster assignments
        return centers, cluster_assignments
    
    @staticmethod
    def __assign_labels_to_centers(centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments: 
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns: 
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        cluster_center_label = np.zeros(centers.shape[0])
        for k in range(centers.shape[0]):
            points_in_cluster = true_labels[cluster_assignments == k]
            if len(points_in_cluster) == 0:
                cluster_center_label[k] = 0
            else:
                cluster_center_label[k] = np.argmax(np.bincount(points_in_cluster.astype(int)))  # Mode can be found using bincount and then argmax
        return cluster_center_label

    def __predict_with_centers(self, data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assigns points in data to their closest cluster, then uses the label
        of that cluster as prediction.

        Arguments: 
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns: 
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """
        new_labels = np.zeros(data.shape[0])
        distances = self.__compute_distance(data, centers)
        cluster_assignments = self.__find_closest_cluster(distances)
        new_labels = cluster_center_label[cluster_assignments]
        return new_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N, D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        self.centroids, cluster_assignments = self.__k_means(training_data, self.max_iters)
        self.best_permutation = self.__assign_labels_to_centers(self.centroids, cluster_assignments, training_labels)
        pred_labels = self.__predict_with_centers(training_data, self.centroids, self.best_permutation)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N, D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        test_labels = self.__predict_with_centers(test_data, self.centroids, self.best_permutation)
        return test_labels
