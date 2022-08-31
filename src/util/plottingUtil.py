import matplotlib.pyplot as plt
import numpy as np

label_annotations = [
    "left eye center", "right eye center",
    "left eye inner corner", "left eye outer corner", "right eye inner corner", "right eye outer corner",
    "left eyebrow inner end", "left eyebrow outer end", "right eyebrow inner end", "right eyebrow outer end",
    "nose tip", "mouth left corner", "mouth right corner", "mouth center top lip", "mouth center bottom lip"
]


def plotImageWithLabels(im, y, plotTitle, legend=True, save=False, path="../../output/defaultOutput.png"):
    """Plots the given image with the given labels."""
    labels = np.array(label_annotations)[y[0::2] != -1]

    y = y[y != -1]
    x = y[0::2]
    y = y[1::2]

    fig, ax = plt.subplots(facecolor='white')
    fig.set_size_inches(10, 5)
    fig.set_dpi(100)

    ax.imshow(im.reshape(96, 96), cmap='gray', vmin=0, vmax=255)
    ax.set_title(plotTitle)
    ax.set_xlabel("x in px")
    ax.set_ylabel("y in px")

    if legend:
        scatter_plot = ax.scatter(x, y, marker="s", s=8, c=range(labels.size), cmap="tab20")
        handles, _ = scatter_plot.legend_elements(num=None)
        ax.legend(handles, labels, title="Facial Landmarks", loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.scatter(x, y, marker="s", s=8, c="g")

    if save:
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
