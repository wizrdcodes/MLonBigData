import os
import matplotlib.pyplot as plt

def title_xy_labels(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def next_plot_path(base="plot", folder="plots", ext="png"):
    os.makedirs(folder, exist_ok=True)
    i = 1
    while True:
        name = f"{base}_{i:03d}.{ext}"
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            return path
        i += 1

def show_tight_layout(show=True, save=True, folder="plots"):
    """
    show=True: display the plot window (or inline in notebooks)
    save=True: save a numbered plot into folder/
    """
    plt.tight_layout()
    if save:
        plt.savefig(
            next_plot_path(folder=folder), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

