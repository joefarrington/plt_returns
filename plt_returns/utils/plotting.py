import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(
    df,
    x_col,
    y_col,
    val_col,
    title,
    subplots_kwargs={},
    heatmap_kwargs={},
):

    to_plot = df.pivot(index=y_col, columns=x_col, values=val_col)
    to_plot = to_plot.sort_index(ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), **subplots_kwargs)
    ax.set_title(title)
    return sns.heatmap(
        to_plot,
        annot=True,
        xticklabels=np.arange(0, 1.1, 0.1).round(1),
        square=True,
        fmt=".2f",
        **heatmap_kwargs
    )
