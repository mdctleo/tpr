import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt

def ks_test_kdes(kde1, kde2, num_samples=1000):
    """
    Perform the K-S test on two KDEs to see how different they are.

    Parameters:
    kde1 (KernelDensity): The first KDE.
    kde2 (KernelDensity): The second KDE.
    num_samples (int): The number of samples to generate from each KDE.

    Returns:
    float: The K-S statistic.
    float: The p-value of the test.
    """
    # Generate samples from the KDEs
    samples1 = kde1.sample(num_samples)
    samples2 = kde2.sample(num_samples)

    # Perform the K-S test
    ks_statistic, p_value = ks_2samp(samples1[:, 0], samples2[:, 0])

    return ks_statistic, p_value

def plot_bar_graph(x, y, x_label, y_label, title, path):

    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(x, y, color='b')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.6f}', ha='center', va='bottom')

    plt.tight_layout()

    plt.savefig(path)
    plt.close()

def plot_multiline_graph(x, ys, x_label, y_label, title, path, use_log_scale=False):
    plt.figure(figsize=(12, 6))
    for label, data in ys.items():
        plt.plot(x, data, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if use_log_scale:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_x_y_line_graph(x, y, x_label, y_label, title, path, highlight_point=None):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(path)
    if highlight_point:
        plt.scatter(*highlight_point, color='red', zorder=5)
        plt.annotate(f'Max Delta\n({highlight_point[0]}, {highlight_point[1]:.2f})', 
                     xy=highlight_point, 
                     xytext=(highlight_point[0], highlight_point[1] + 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.close()

def generate_evenly_spaced_visualizations(sorted_edges, num_visualizations=5):
    num_edges = len(sorted_edges)
    if num_edges == 0:
        print("No edges to visualize.")
        return
    
    if num_visualizations > num_edges:
        num_visualizations = num_edges

    interval = num_edges // num_visualizations

    for i in range(num_visualizations):
        index = i * interval
        if index < num_edges:
            sorted_edges[index][1].visualize_distribution()

def find_max_delta_point(x, y):
    deltas = np.diff(y)
    max_delta_index = np.argmax(np.abs(deltas))
    max_delta_point = (x[max_delta_index + 1], y[max_delta_index + 1])
    return max_delta_point

# Example usage
if __name__ == '__main__':
    # Generate some sample data
    np.random.seed(0)
    data1 = np.random.normal(loc=0, scale=1, size=1000)
    data2 = np.random.normal(loc=0.5, scale=1.5, size=1000)

    # Fit KDEs to the data
    kde1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data1[:, np.newaxis])
    kde2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data2[:, np.newaxis])

    # Perform the K-S test
    ks_statistic, p_value = ks_test_kdes(kde1, kde2)
    print(f"K-S statistic: {ks_statistic}")
    print(f"P-value: {p_value}")