import cvxpy as cp
import numpy as np
import torch
import sklearn.metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Utility Functions ---
def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(array, k, axis=-1):
    """
    Get the k-th smallest value along the specified axis.

    Args:
        array (np.ndarray): Input array.
        k (int): Index of the k-th smallest value.

    Returns:
        np.ndarray: Array of k-th smallest values.
    """
    indices = np.argpartition(array, k, axis=axis)[..., :k]
    return np.take_along_axis(array, indices, axis=axis).max(axis=axis)


def compute_nearest_neighbour_distances(features, k):
    """
    Compute distances to the k-th nearest neighbors.

    Args:
        features (np.ndarray): Shape [N, feature_dim].
        k (int): Number of nearest neighbors.

    Returns:
        np.ndarray: Distances to the k-th nearest neighbors.
    """
    distances = compute_pairwise_distance(features)
    return get_kth_value(distances, k + 1)


def compute_linear_term(real_features, fake_features, k, metric="precision"):
    """
    Compute the linear term for evaluating the models.

    Args:
        real_features (np.ndarray): Real dataset features.
        fake_features (np.ndarray): Fake dataset features.
        k (int): Number of nearest neighbors.
        metric (str): 'precision' or 'density'.

    Returns:
        float: Linear metric value.
    """
    reals_nnd = compute_nearest_neighbour_distances(real_features, k)
    distances = compute_pairwise_distance(real_features, fake_features)

    if metric == "precision":
        return (distances < reals_nnd[:, None]).any(axis=0).mean()
    elif metric == "density":
        return (distances < reals_nnd[:, None]).sum(axis=0).mean() / k
    raise ValueError(f"Invalid metric: {metric}")


# --- Kernel Utilities ---
class KernelUtils:
    @staticmethod
    def gaussian_kernel(x, y, sigma):
        dist_sq = torch.sum((x - y) ** 2, dim=-1)
        return torch.exp(-0.5 * dist_sq / sigma ** 2)

    @staticmethod
    def frobenius_norm(X, Y, sigma=SIGMA, block_size=800):
        """
        Compute Frobenius norm between datasets `X` and `Y`.

        Args:
            X (np.ndarray): Dataset of shape [N, feature_dim].
            Y (np.ndarray): Dataset of shape [N, feature_dim].
            sigma (float): Gaussian kernel bandwidth.
            block_size (int): Block size for computation.

        Returns:
            float: Frobenius norm value.
        """
        X, Y = torch.tensor(X, device=DEVICE), torch.tensor(Y, device=DEVICE)
        sum_norm = 0.0

        for i in range(0, X.shape[0], block_size):
            for j in range(0, Y.shape[0], block_size):
                X_block = X[i : i + block_size]
                Y_block = Y[j : j + block_size]
                kernel_block = KernelUtils.gaussian_kernel(
                    X_block.unsqueeze(0), Y_block.unsqueeze(1), sigma
                ) ** 2
                sum_norm += kernel_block.sum().item()
        return sum_norm

    @staticmethod
    def scaled_kernel(kernel, sizes):
        """
        Scale a kernel matrix by dataset sizes.

        Args:
            kernel (np.ndarray): Kernel matrix.
            sizes (list[int]): Dataset sizes.

        Returns:
            np.ndarray: Scaled kernel matrix.
        """
        scale = 1 / np.array(sizes)
        return kernel * np.outer(scale, scale)


# --- Main Functions ---
def calculate_rke(models,sigma):
    """
    Calculate RKE for the given models.

    Args:
        models (dict[str, np.ndarray]): Dictionary of model outputs.

    Returns:
        np.ndarray: Scaled kernel matrix.
    """
    keys = list(models.keys())
    kernel = np.zeros((len(keys), len(keys)))

    for i, key_i in enumerate(keys):
        xi = models[key_i]
        for j, key_j in enumerate(keys):
            xj = models[key_j]
            kernel[i, j] = KernelUtils.frobenius_norm(xi, xj, sigma)

    sizes = [len(models[key]) for key in keys]
    return KernelUtils.scaled_kernel(kernel, sizes)


def calculate_precision(models, real):
    """
    Calculate precision for the given models.

    Args:
        models (dict[str, np.ndarray]): Dictionary of model outputs.
        real (np.ndarray): Real dataset features.

    Returns:
        np.ndarray: Precision values.
    """
    keys = list(models.keys())
    return np.array([compute_linear_term(real, models[key], 5, "precision") for key in keys])


def calculate_optimal_mixture(models, quadratic_calculator, has_linear=False, **kwargs):
    """
    Find the optimal mixture of models.

    Args:
        models (dict[str, np.ndarray]): Dictionary of model outputs.
        quadratic_calculator (function): Function to compute the quadratic term.
        has_linear (bool): Whether to include a linear term.
        kwargs: Additional arguments, e.g., real data and linear term calculator.

    Returns:
        np.ndarray: Optimal mixture coefficients.
    """
    num_models = len(models)
    alphas = cp.Variable(num_models, nonneg=True)
    kernel = quadratic_calculator(models,kwargs["sigma"])

    kernel += np.eye(kernel.shape[0]) * 1e-7  # Numerical stability
    objective = cp.quad_form(alphas, kernel)

    if has_linear:
        real_data = kwargs["real_data"]
        linear_term = kwargs["linear_term_calculator"](models, real_data)
        linear_term_weight = kwargs.get("linear_term_weight", 1.0)
        objective -= linear_term_weight * cp.sum(cp.multiply(linear_term, alphas))

    constraints = [cp.sum(alphas) == 1]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    return alphas.value