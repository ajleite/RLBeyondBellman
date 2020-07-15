''' Please note that both of these figures were later revised to remove
    extraneous labeling. '''

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def prep_spectrum_df(spectrum, base_dir, filename, grad):
    spec_filename = os.path.join(base_dir, filename + ".csv")
    if os.path.exists(spec_filename):
        return pd.read_csv(spec_filename, header=[0, 1], index_col=0)
    spectrum = np.reshape(spectrum, (spectrum.shape[0], -1))

    if grad:

        def normalize(x):
            x_min, x_max = np.min(x), np.max(x)
            x[x < 0] /= x_min
            x[x > 0] /= x_max
            return x

    else:
        normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    for i, spec in enumerate(spectrum):
        spectrum[i] = normalize(spec)

    # prep for df structure
    n_eQ = spectrum.shape[0] // 2
    spectrum = np.transpose(spectrum)
    headers = pd.MultiIndex.from_product(
        [["eQ", "DDPG"], np.arange(1, 1+n_eQ)], names=["algo", "run"]
    )
    indices = np.arange(np.shape(spectrum)[0])
    df = pd.DataFrame(spectrum, index=indices, columns=headers)

    df.to_csv(spec_filename)
    return df


def make_cluster_heatmap(spectrum, base_dir, filename, grad=False):
    # Load and prep dataset
    df = prep_spectrum_df(spectrum, base_dir, filename, grad)
    del spectrum

    # Create a categorical palette to identify the networks
    algo_pal = sns.husl_palette(8, s=0.45)[-2:]
    algo_lut = dict(zip(["eQ", "DDPG"], algo_pal))

    # Convert the palette to vectors that will be drawn on the side of the matrix
    algos = df.columns.get_level_values("algo")
    algo_colors = pd.Series(algos, index=df.columns).map(algo_lut)

    # Draw the full plot
    sns.clustermap(
        df.corr(),
        center=0,
        cmap="vlag",
        row_colors=algo_colors,
        col_colors=algo_colors,
        # col_cluster=False,
        linewidths=0.75,
        figsize=(4,4),
    )

    plt.savefig(os.path.join(base_dir, filename + ".pdf"))
    plt.savefig(os.path.join(base_dir, filename + ".png"))
    plt.show()


if __name__ == "__main__":

    Q_spectrum = np.load(
        "alife-results/best-Qmaps.npy"
    )
    make_cluster_heatmap(Q_spectrum, "alife-results", "figure_4_Q_spec_df")

    grad_spectrum = np.load(
        "alife-results/best-training-signals.npy"
    )
    make_cluster_heatmap(grad_spectrum, "alife-results", "figure_5_grad_spec_df", grad=True)
