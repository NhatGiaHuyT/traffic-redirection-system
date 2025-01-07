import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class KMeansCluster:
    def __init__(self, tracked_data : dict = None, 
                 data_file:str = "tracked_data.pkl", 
                 n_clusters=6, camera_id = None):
        """
        Initialize the object.

        Args:
            - tracked_data (dict) : The tracked data `{object_id: [((x1, y1), v1), ((x2, y2), v2), ...]}`.
            - data_file (str) : The filename of the data file.
            - n_clusters (int) : The number of clusters.
            - camera_id (int) : The camera id.
        """
        
        if tracked_data is None and data_file is None:
            raise ValueError("You must provide either the tracked data or the data file")
        
        self.n_clusters_ = n_clusters
        self.id_ = camera_id
        self.colors_ = [
                'b',     # Blue
                'g',     # Green
                'r',     # Red
                'c',     # Cyan
                'm',     # Magenta
                'y',     # Yellow
                # 'k',     # Black
                # 'w',     # White
                'orange',
                'purple',
                'pink',
                'brown',
                'gray',
                'lightblue',
                'lightgreen',
                'lightcoral',
                'gold',
                'navy',
                'teal',
                'lime',
                'violet',
                'indigo',
                'maroon',
                'olive',
                'chocolate',
                'salmon',
                'khaki',
                'plum',
                'slategray',
                'coral',
                'darkgreen',
                'darkblue',
                'tan',
                'crimson',
                'steelblue',
                'sandybrown',
                'springgreen',
                'aqua',
                'fuchsia',
                'lavender',
                'seashell',
                'forestgreen',
                'royalblue',
                'powderblue',
                'honeydew',
                'peachpuff',
                'chartreuse',
                'palevioletred'
            ]
        self.kmeans_ = KMeans(n_clusters=self.n_clusters_, random_state=RANDOM_STATE)
        self.tracked_data_ = tracked_data or self.read_tracked_paths(data_file)
        self.data_size_ = len(self.tracked_data_)
        self.tracked_paths_2_paths_n_lines()



    def read_tracked_paths(self, filename:str) -> dict:
        """
        Read the tracked data from the data file

        Args:
        filename: str
            The filename of the data file

        Returns:
        tracked_data: dict
            The accumulated tracking data `{object_id: [((x1, y1), v1), ((x2, y2), v2), ...]}`
        """
        with open(filename, "rb") as f:
            tracked_data = pickle.load(f)
        
        return tracked_data 
    
    def tracked_paths_2_paths_n_lines(self):
        """
        Convert the tracked data to paths and lines
        """
        paths = []
        lines = []
        for value in self.tracked_data_.values():
            path = [point for point, _ in value]
            paths.append(np.array(path))

            fitted_line = cv2.fitLine(path, cv2.DIST_L2, 0, 0.01, 0.01) # vx, vy, x0, y0
            if np.dot(fitted_line[:2].flatten(), path[-1] - path[0]) < 0: # If the direction vector is opposite to the path
                    fitted_line[:2] *= -1
            lines.append(fitted_line.flatten())

        self.paths_ = paths
        self.minmax_range_paths_ = np.array([path.max(axis=0) - path.min(axis=0) for path in paths])
        self.lines_ = lines
    
    def get_line_intersection(self, line, point):
        """
        Get the intersection point of the line and an external point.
        """
        vx, vy, x0, y0 = line[:4]
        xp, yp = point
        t = -((vx * (x0 - xp)) + (vy * (y0 - yp))) / (vx**2 + vy**2)
        pos_c = x0 + t * vx, y0 + t * vy

        return pos_c

    def plot_vector(self, dv, point=None, color='b', t: int = 100):
        """
        Plot the direction vector.

        Args:
            - dv (array-like) : The direction vector.
            - point (array-like) : The external point.
            - color (str) : The color of the vector.
            - t (int) : The length of the vector.
        """
        vx, vy, x0, y0 = dv[:4]
        if point is not None:
            x1, y1 = self.get_line_intersection(dv, point)
            plt.quiver(x0, y0, x1 - x0, y1 - y0, angles='xy', scale_units='xy', scale=1, color=color)
        else:
            plt.quiver(x0, y0, x0 + t*vx, y0 + t*vy, angles='xy', scale_units='xy', scale=1, color=color)



    def fit(self, features, is_return: bool = False):
        """
        Fitting the feature vectors to the KMeans model and group the paths and lines via the labels.

        Args:
            - features (array-like) : The feature vectors.
            - is_return (bool) : Whether to return the grouped paths and lines.

        Returns:
            - grouped_paths (dict) : The grouped paths by the direction of the lines.
            - grouped_lines (dict) : The grouped lines by the direction of the lines.
        """
        self.kmeans_.fit(features.copy())
        
        if not is_return:
            return
        """Group the paths by the direction of the lines"""
        grouped_paths = dict()
        grouped_lines = dict()
        for i in range(self.n_clusters_):
            grouped_paths[i] = []
            grouped_lines[i] = []
            for j, line in enumerate(self.lines_):
                if self.kmeans_.labels_[j] == i:
                    grouped_paths[i].apped(self.paths_[j])
                    grouped_lines[i].append(line)

        return grouped_paths, grouped_lines
                
    
    def predict(self, features):
        """
        Predict the labels of the feature vectors.

        Args:
            - features (array-like) : The feature vectors.

        Returns:
            - labels (array-like) : The predicted labels.
        """
        return self.kmeans_.predict(features)
    
    def plot_grouped_paths(self, grouped_paths=None, figsize=(12, 8), img_shape=(1080, 1920)):
        """
        Plot the grouped paths.

        Args:
            - grouped_paths (dict) : The grouped paths by the direction of the lines.
            - figsize (tuple) : The output figure size.
            - img_shape (tuple) : The shape of the image.
        """
        if grouped_paths is None:
            grouped_paths = self.grouped_paths_
        
        plt.figure(figsize=figsize)
        for i, paths in grouped_paths.items():
            for path in paths:
                plt.plot(path[:, 0], path[:, 1], self.colors_[i % len(self.colors_)])
                plt.plot(path[-1, 0], path[-1, 1], 'v', color=self.colors_[i % len(self.colors_)])
        plt.xlim(0, img_shape[1])
        plt.ylim(img_shape[0], 0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    def save(self, filename):
        """
        Save the KMeans model to the file.

        Args:
            - filename (str) : The filename of the file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.kmeans_, f)
    
    def load(self, filename):
        """
        Load the KMeans model from the file.

        Args:
            - filename (str) : The filename of the cluster parameters.
        """
        with open(filename, 'rb') as f:
            self.kmeans_ = pickle.load(f)