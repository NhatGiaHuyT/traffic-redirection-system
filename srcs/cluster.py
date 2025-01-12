import numpy as np
from sklearn.cluster import KMeans
import cv2
import pickle
import matplotlib.pyplot as plt

class PathClusterer:
    def __init__(self, n_clusters=2, random_state=42):
        """
        Initializes the PathClusterer with the number of clusters and random state.

        Parameters:
            n_clusters (int): Number of clusters to form.
            random_state (int): Random state for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, features):
        """
        Fits the KMeans model to the provided features.

        Parameters:
            features (ndarray): Array of feature vectors.
        """
        self.kmeans.fit(features)

    def predict(self, features):
        """
        Predicts the cluster labels for the provided features.

        Parameters:
            features (ndarray): Array of feature vectors.

        Returns:
            ndarray: Cluster labels for each feature vector.
        """
        return self.kmeans.predict(features)

    def fit_and_predict(self, features):
        """
        Fits the KMeans model and predicts cluster labels in one step.

        Parameters:
            features (ndarray): Array of feature vectors.

        Returns:
            ndarray: Cluster labels for each feature vector.
        """
        self.fit(features)
        return self.predict(features)

    def get_line_directions(self, paths):
        """
        Computes direction vectors for each path using linear fitting.

        Parameters:
            paths (list of ndarray): List of paths, where each path is an array of (x, y) points.

        Returns:
            ndarray: Array of normalized direction vectors.
        """
        directions = []
        for path in paths:
            if len(path) < 2:
                continue
            [vx, vy, _, _] = cv2.fitLine(np.array(path, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            direction = np.array([vx[0], vy[0]])
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm
            directions.append(direction)
        return np.array(directions)

    def align_directions(self, directions):
        """
        Aligns direction vectors to ensure consistent orientation.

        Parameters:
            directions (ndarray): Array of direction vectors.

        Returns:
            ndarray: Array of aligned direction vectors.
        """
        aligned = []
        for dir_vec in directions:
            if np.dot(dir_vec, np.array([1, 0])) < 0:
                dir_vec = -dir_vec
            aligned.append(dir_vec)
        return np.array(aligned)

    def save(self, filename):
        """
        Saves the KMeans model to a file.

        Parameters:
            filename (str): Path to the file where the model will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.kmeans, f)

    def load(self, filename):
        """
        Loads the KMeans model from a file.

        Parameters:
            filename (str): Path to the file containing the saved model.
        """
        with open(filename, 'rb') as f:
            self.kmeans = pickle.load(f)

    def plot_grouped_paths(self, paths, labels, img_shape=(500, 500)):
        """
        Visualizes clustered paths on an image.

        Parameters:
            paths (list of ndarray): List of paths, where each path is an array of (x, y) points.
            labels (ndarray): Cluster labels for each path.
            img_shape (tuple): Shape of the output image (height, width).
        """
        colors = plt.cm.get_cmap('tab10', self.n_clusters)
        img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255
        for i, path in enumerate(paths):
            if len(path) < 2:
                continue
            color = tuple((np.array(colors(labels[i]))[:3] * 255).astype(int))
            for j in range(len(path) - 1):
                cv2.line(img, tuple(path[j]), tuple(path[j + 1]), color, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
