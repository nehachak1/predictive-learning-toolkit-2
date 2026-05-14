import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100, distance_metric="euclidean"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        if K <= 0: 
            raise ValueError("K must be a positive integer.")
        
        self.K = K 
        self.max_iters = max_iters
        self.distance_metric = distance_metric

        self.centers = None
        self.cluster_center_label = None

    def init_centers(self, data):
        """
        Randomly pick K data points from the data as initial cluster centers.

        Arguments:
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        if self.K > data.shape[0]: 
            raise ValueError("K cannot be greater than the number of samples.")
        
        indices = np.random.permutation(data.shape[0])[:self.K]
        centers = data[indices].copy()

        return centers

    def compute_distance(self, data, centers):
        """
        Compute the distance between each datapoint and each center.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        distances = np.zeros((data.shape[0], centers.shape[0]))

        for k in range(centers.shape[0]): 
            if self.distance_metric == "euclidean": 
                distances[:, k] = np.sqrt(np.sum((data - centers[k]) ** 2, axis = 1))
            elif self.distance_metric == "chi_square": 
                eps = 1e-12
                numerator = (data - centers[k]) ** 2
                denominator = data + centers[k] + eps
                distances[:, k] = np.sqrt(np.sum(numerator / denominator, axis = 1))
            elif self.distance_metric == "manhattan": 
                distances[:, k] = np.sum(np.abs(data - centers[k]), axis = 1)
            else: 
                raise ValueError("distance_metric must be 'euclidean', 'chi_square' or 'manhattan'.")

        return distances
        
    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        cluster_assignments = np.argmin(distances, axis = 1)

        return cluster_assignments
        
    def compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments:
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        centers = np.zeros((self.K, data.shape[1]))

        for k in range(self.K): 
            points_in_cluster = data[cluster_assignments == k]

            if len(points_in_cluster) == 0: 
                # If a cluster is empty, reinitialize it randomly
                random_index = np.random.randint(data.shape[0])
                centers[k] = data[random_index]
            else: 
                centers[k] = np.mean(points_in_cluster, axis = 0)

        return centers
                
    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.

        Arguments:
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
        centers = self.init_centers(data)

        for i in range(max_iter): 
            old_centers = centers.copy()

            distances = self.compute_distance(data, centers)
            cluster_assignments = self.find_closest_cluster(distances)
            centers = self.compute_centers(data, cluster_assignments)

            if np.allclose(centers, old_centers): 
                print(f"K-Means has converged after {i + 1} iterations!")
                break

        return centers, cluster_assignments
       
    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments:
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns:
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        cluster_center_label = np.zeros(centers.shape[0], dtype = int)

        for k in range(centers.shape[0]): 
            labels_in_cluster = true_labels[cluster_assignments == k]

            if len(labels_in_cluster) == 0: 
                cluster_center_label[k] = 0
            else: 
                labels_in_cluster = labels_in_cluster.astype(int)
                cluster_center_label[k] = np.bincount(labels_in_cluster).argmax()

        return cluster_center_label

    def predict_with_centers(self, data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assign points in data to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns:
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """
        distances = self.compute_distance(data, centers)
        cluster_assignments = self.find_closest_cluster(distances)

        new_labels = cluster_center_label[cluster_assignments]

        return new_labels

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        self.centers, cluster_assignments = self.k_means(
            training_data, 
            max_iter = self.max_iters
        )

        self.cluster_center_label = self.assign_labels_to_centers(
            self.centers, 
            cluster_assignments, 
            training_labels
        )

        pred_labels = self.predict_with_centers(
            training_data, 
            self.centers, 
            self.cluster_center_label
        )

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        if self.centers is None or self.cluster_center_label is None: 
            raise ValueError("Model must be fitted before calling predict().")
        
        pred_labels = self.predict_with_centers(
            test_data,
            self.centers, 
            self.cluster_center_label
        )

        return pred_labels
