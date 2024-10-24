#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 24/10/2024

# Standard library imports
import warnings

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_graph(y_pred, y_real, title) -> plt.Figure:
    """
    Plot the graph
    :param y_pred:
    :param y_real:
    :param title:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_pred, color="orange", label="Predicted")
    ax.plot(y_real, linestyle="dashed", linewidth=1, label="Actual")
    ax.grid(True, color="#666666", linestyle="-")
    ax.minorticks_on()
    ax.set_ylabel("Mean Air Temperature [°C]")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend()
    # plt.close()
    return fig


def plot_scatter(y_pred, y_real) -> plt.Figure:
    """
    Scatter plot
    :param y_pred:
    :param y_real:
    :return:
    """
    fig_scatter = plt.figure(figsize=(5, 5))
    plt.scatter(y_real, y_pred, alpha=0.3, color="#88c999")
    p1 = min(np.nanmin(y_real), np.nanmin(y_pred))
    p2 = max(np.nanmax(y_real), np.nanmax(y_pred))
    plt.plot([p1, p2], [p1, p2], "b-")
    plt.grid(True, color="#666666", linestyle="-")
    plt.minorticks_on()
    # Set x-axis and y-axis limits adaptively
    max_limit = max(np.nanmax(y_real), np.nanmax(y_pred))
    plt.xlim(0, max_limit)
    plt.ylim(0, max_limit)
    plt.xlabel("Real Power [kW]")
    plt.ylabel("Forecasted Power [kW]")
    plt.title("Predicted vs Actual power values")
    return fig_scatter


def error_distribution(y_pred, y_real) -> plt.Figure:
    """
    Plot the error distribution
    :param y_pred:
    :param y_real:
    :return:
    """
    error = y_pred - y_real
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.minorticks_on()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dens = sns.kdeplot(error, color="dodgerblue", lw=1, fill=True)
            if dens.collections:
                p = dens.collections[0].get_paths()[0]
                x = p.vertices[:, 0]
                y = p.vertices[:, 1]
                ax.fill_between(x, y, color="dodgerblue", alpha=0.3)
                dens.axvline(x=0, color="dodgerblue", linestyle="dashed")
                dens.set_xlim(-max(abs(error)), max(abs(error)))  # symmetric limits on x
                dens.set_title("Error distribution")
    except Exception as e:
        print(e)
        ax.hist(error, bins=30, color="dodgerblue", alpha=0.7)
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    return fig
