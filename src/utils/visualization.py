#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 20/10/2024

# Standard library imports
import warnings

# Third party imports
import matplotlib.pyplot as plt
import seaborn as sns


def plot_graph(ypred, ylab, title):
    """
    Plot the graph
    :param ypred:
    :param ylab:
    :param title:
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(ypred, color="orange", label="Predicted")
    ax.plot(ylab, linestyle="dashed", linewidth=1, label="Actual")
    ax.grid(visible=True, which="major", color="#666666", linestyle="-")
    ax.minorticks_on()
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # plt.xlim(left=0, right=306)
    ax.set_ylabel("Mean Air Temperature [°C]")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend()
    # plt.close()
    return fig


def plot_scatter(ypred, yreal):
    """
    Scatter plot
    :param ypred:
    :param yreal:
    :return:
    """
    scatter = plt.figure()
    plt.scatter(yreal, ypred, edgecolor="white", linewidth=1, alpha=0.05)
    plt.grid(True, which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xlabel("Actual Temperature [°C]")
    plt.ylabel("Predicted Temperature [°C]")
    plt.title(
        "Comparison among predicted and actual temperature \n in simulation environment using LSTM"
    )
    # plt.show()
    # plt.close()
    return scatter


def error_distribution(predictions, actuals):
    """
    Plot the error distribution
    :param predictions:
    :param actuals:
    :return:
    """
    error = predictions - actuals
    fig, ax = plt.subplots()
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
                dens.set_xlim(-5, 5)
                dens.set_title("Error distribution")
    except Exception as e:
        ax.hist(error, bins=30, color="dodgerblue", alpha=0.7)
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    return fig
