import numpy as np
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

def cartesian_encoder(coord, r_E=6371):
    """Convert lat/lon to cartesian points on Earth's surface.

    Input
    -----
        coord : numpy 2darray (size=(N, 2))
        r_E : radius of Earth

    Output
    ------
        out : numpy 2darray (size=(N, 3))
    """
    def _to_rad(deg):
        return deg * np.pi / 180.

    theta = _to_rad(coord[:, 0])  # lat [radians]
    phi = _to_rad(coord[:, 1])    # lon [radians]

    x = r_E * np.cos(phi) * np.cos(theta)
    y = r_E * np.sin(phi) * np.cos(theta)
    z = r_E * np.sin(theta)

    return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)

def haversine(coord1, coord2):    
    def __to_rad(degrees):
        return degrees * 2 * np.pi / 360.

    lat1, lat2 = __to_rad(coord1[0]), __to_rad(coord2[0])
    dlat = __to_rad(coord1[0] - coord2[0])
    dlon = __to_rad(coord1[1] - coord2[1])

    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * \
        np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 

    return 6371e3 * c  # Meters

def group_time_distance(coords, r_C):
    """Group temporally adjacent points if they are closer than r_C.
    
    Input
    -----
        coords : array-like (shape=(N, 2))
        r_C : number (critical radius)
    
    Output
    ------
        groups : list-of-list
            Each list is a group of points
    """
    groups = []
    
    current_group = coords[0].reshape(-1, 2)
    for coord in coords[1:]:
        
        # Compute distance to current group
        dist = haversine(np.median(current_group, axis=0), coord)
    
        # Put in current group
        if dist <= r_C:
            current_group = np.vstack([current_group, coord])
        # Or start new group if dist is too large
        else:
            groups.append(current_group)
            current_group = coord.reshape(-1, 2)

    # Add the last group
    groups.append(current_group)

    return groups

def get_stationary_medoids(groups, min_size=1):
    """Convert groups of multiple points (stationary location events) to median-of-group points.
    
    Input
    -----
        groups : list-of-list
            Each list is a group of points
        min_size : int
            Minimum size of group to consider it stationary (default: 1)
            
    Output
    ------
        stat_coords : array-like (M, 2)
            Medioids of stationary groups
    """
    stat_coords = np.empty(shape=(0, 2))
    medoid_map = []
    i = 0
    for g in groups:
        if g.shape[0] > min_size:
            stat_coords = np.vstack([stat_coords, np.median(g, axis=0).reshape(1, -1)])
            medoid_map.extend([i] * len(g)); i += 1
        else:
            medoid_map.append(-1)
     
    return stat_coords, np.array(medoid_map)

def cluster_DBSCAN(coords, eps=5, min_samples=2):
    """Get labels of DBSCAN pass over a set of coordinates.
    
    Input
    -----
        coords : array-like (N, 2)
        eps : number (max. intercluster distance)
        min_samples : int (min. cluster samples)
    
    Output
    ------
        labels : 1d array
    """
    model_DBSCAN_all_static = DBSCAN(
        eps=eps, min_samples=min_samples, metric=haversine
    ).fit(coords)
    
    return model_DBSCAN_all_static.labels_

def optimal_sub_labeling(coords, K_max=10, var_dev=1, scale_lim=[2, 10], min_range=15, plot=False):
    """For a range of K values, return the labeling that minimizes that average KS statistic.
    
    Project points onto principle components to avoid multiple geo-problems.
    
    Input
    -----
        coords : array-like (N, 2)
        K_max : int (largest K to test)
        var_dev : number (component size deviation)
        scale_lim : list (, 2) (min-max range of cluster SD)
        min_range : number (min. range that will be clustered)
    
    Output
    ------
        out : 1d-array (point labels)
    """
    
    def _min_index(arr):
        return arr.index(min(arr))
    
    def _meters_to_dlat(m, R_E=6371e3):
        return m / R_E * 180 / np.pi

    def _dlat_to_meters(dlat, R_E=6371e3):
        return R_E * dlat * np.pi / 180

    def _meters_to_dlon(m, lat, R_E=6371e3):
        return m / R_E * 180 / np.pi / np.cos(lat * np.pi / 180)

    def _dlon_to_meters(dlon, lat, R_E=6371e3):
        return R_E * dlon * np.pi / 180 * np.cos(lat * np.pi / 180)
    
    def _clip_range(x, lim):
        return min([max([x, lim[0]]), lim[1]])
    
    def _norm_dist(p):
        return p * 1.0 / np.sum(p)

    def _compute_cumulative_dist(pmf_2d):
        cmf_2d = np.empty(shape=pmf_2d.shape)
        for i in range(pmf_2d.shape[0]):
            for j in range(pmf_2d.shape[1]):
                cmf_2d[i, j] = np.sum(pmf_2d[:i+1, :j+1])
        return cmf_2d
    
    def _cartesian_encoder(coord, r_E=6371):
        def __to_rad(deg):
            return deg * np.pi / 180.
        theta = __to_rad(coord[:, 0])  # lat [radians]
        phi = __to_rad(coord[:, 1])    # lon [radians]
        x = r_E * np.cos(phi) * np.cos(theta)
        y = r_E * np.sin(phi) * np.cos(theta)
        z = r_E * np.sin(theta)
        return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)
    
    def _pca_transform(X, n_components=None):
        return PCA(n_components=n_components).fit_transform(X)
    
    def _get_normal_pmf(mu, bins, SD, size=100000):
        X = np.random.multivariate_normal(mu, np.array([[SD[0], 0], [0, SD[1]]])**2, size=size)
        return _norm_dist(np.histogram2d(X[:, 0], X[:, 1], bins=bins, normed=True)[0])

    # Adjust K_max to never exceed max possible
    K_max = min(np.unique(coords, axis=0).shape[0]-1, K_max)
    
    # Center coordinates at (0, 0)
    coords = _pca_transform(_cartesian_encoder(coords, r_E=6371e3), 2)
    
    # The extend of the input data
    x_min, x_max = min(coords[:, 0]), max(coords[:, 0])
    y_min, y_max = min(coords[:, 1]), max(coords[:, 1])
    
    # Check if range is greater than min_range. If not, return single-cluster solution
    if x_max - x_min < min_range and y_max - y_min < min_range:
        return np.zeros(shape=(coords.shape[0], ), dtype=int)
    
    # The pmf bins
    bins = [
        np.arange(x_min, x_max, 2), # 2 m wide bins
        np.arange(y_min, y_max, 2)  # 2 m wide bins
    ]

    # Iteration solutions
    solutions = defaultdict(lambda: defaultdict(list))
    for K in range(1, K_max + 1):

        # Fit GMM with K components
        if K == 1:
            solutions[K]['labels'] = np.zeros(shape=coords.shape[0], dtype=int)
        else:
            solutions[K]['labels'] = KMeans(K).fit_predict(coords)

        # Loop over each kernel and increment the average KS score of the partition
        for k in range(K):
            
            # If there are no points in the k'th cluster, continue. (consider breaking)
            if not np.sum(solutions[K]['labels'] == k):
                solutions[K]['kernels'].append({'KS_score': np.nan, 'p': 0})
                continue
                
            # Only points in kth cluster
            coords_k = coords[solutions[K]['labels'] == k]
            
            # Mean and SD of that cluster
            mu_k = np.mean(coords_k, axis=0)
            SD_k = np.std(coords_k, axis=0)

            # Estimate the average SD of the data in meters
            scale = np.mean(SD_k)
            
            # Clip that to fit the acceptable range
            scale = _clip_range(scale, scale_lim)
            
            # Clip the SDs so they are within the scale_lim and respect the var_dev
            SD_k = [
                _clip_range(SD_k[0], lim=[scale-var_dev, scale+var_dev]),
                _clip_range(SD_k[1], lim=[scale-var_dev, scale+var_dev])
            ]


            # Get pmf and cmf of cluster.
            pmf_k = _norm_dist(np.histogram2d(coords_k[:, 0], coords_k[:, 1], bins=bins)[0])
            cmf_k = _compute_cumulative_dist(pmf_k)

            # Get the "target" cmf, as the accumulate of the perfect pmf.
            pmf_t = _get_normal_pmf(mu_k, bins, SD_k)
            cmf_t = _compute_cumulative_dist(pmf_t)

            # Store solution
            solutions[K]['kernels'].append({
                'KS_score': np.max(abs(cmf_k - cmf_t)),           # max dist. between (KS score)
                'p': coords_k.shape[0] * 1. / coords.shape[0]     # weight of kth kernel
            })
    
    # Compute the weighted average KS for all K
    KS_averages = [
        np.sum([k['p'] * k['KS_score'] for k in obj['kernels']])
        for obj in solutions.values()
    ]
    
    # Pick the K that yields the lowst avg. KS
    optimal_K = _min_index(KS_averages) + 1
    
    # Plot static
    if plot:
        plt.figure()
        plt.plot(range(1, K_max+1), KS_averages)
        plt.xlabel("K")
        plt.ylabel("Average KS")
        plt.show()
        
    return solutions[optimal_K]['labels']

def get_stop_location_labels(coords):
    """Get stop-location labels for time-sorted gps coordinates.
    
    Input
    -----
        coords : array-like (N, 2)
        
    Output
    ------
        out : array-like (N, )
    """
    
    def _reset_label_range(labels):
        """Reset labels to range between -1 and len(labels)."""
        unique_labels = set(labels) - {-1}
        labels_map = dict(zip(unique_labels, range(len(unique_labels))))
        labels_map[-1] = -1
        return np.array([labels_map[i] for i in labels])
    
    # 1. Get clusters of stationary points in time. Take medians
    groups = group_time_distance(coords, r_C=10)
    stat_medoids, medoid_map = get_stationary_medoids(groups, min_size=1)
    
    # 2. DBSCAN cluster the resulting stationary medoids
    labels_DBSCAN = cluster_DBSCAN(stat_medoids)
    
    # Map medoid labels back to coordinates
    coord_labels = np.ones(shape=coords.shape[0], dtype=int) * (-1)
    for gid, lab in enumerate(labels_DBSCAN):
        coord_labels[medoid_map == gid] = lab
        
    # 3. For each cluster, find subclusters with GMM
    sub_labels = np.ones(shape=coords.shape[0], dtype=int) * (-1)  # Default (-1) is outlier
    
    for lab in filter(lambda lab: lab >= 0, set(labels_DBSCAN)):  # Loop over non-outlier labels

        # Get the GMM subcluster labels
        sub_cluster_labels = optimal_sub_labeling(coords[coord_labels == lab, :], K_max=2)
        
        # Rename them so they don't conflict across clusters
        global_labels = np.array([int(str(lab)+str(sub_lab)) for sub_lab in sub_cluster_labels])

        # Store them
        sub_labels[coord_labels == lab] = global_labels

    # Rename the sublabels to a tight range
    sub_labels = _reset_label_range(sub_labels)
    
    return sub_labels