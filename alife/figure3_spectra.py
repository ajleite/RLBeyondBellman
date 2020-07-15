#!/usr/bin/python3
import os

import numpy as np
import matplotlib.pyplot as plt

def make_spectrum_plot(shape, loc, dat, cmap, xlabel, ylabel, xticks, yticks, title):
    plt.subplot2grid(shape, loc)
    plt.imshow(dat, cmap=cmap, origin="lower", extent=(np.pi, -np.pi, -2, 2))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([-2.5, 0, 2.5], xticks)
    plt.yticks([-2, 0, 2], yticks)
    plt.title(title, fontsize=9)
    return plt.gca()


def plot_single_spectrum_figure(Q1, Q2, g1, g2, outfile, eQ_id=3):
    for i in range(len(g1)):
        Q1[i] = (Q1[i] - np.min(Q1[i])) / (np.max(Q1[i]) - np.min(Q1[i]))

        Q2[i] = (Q2[i] - np.min(Q2[i])) / (np.max(Q2[i]) - np.min(Q2[i]))

        g1[i] = np.array(g1[i])
        g1[i][g1[i] < 0] /= np.min(g1[i])
        g1[i][g1[i] > 0] /= np.max(g1[i])

        g2[i] = np.array(g2[i])
        g2[i][g2[i] < 0] /= np.min(g2[i])
        g2[i][g2[i] > 0] /= np.max(g2[i])

    plt.figure(figsize=[7.5, 4])
    idx = [0, 4, 8]
    titles = [r"$\omega = -8$", r"$\omega = 0$", r"$\omega = 8$"]

    # Q1 = [np.mean(Q1, axis=0)]
    # Q2 = [np.mean(Q2, axis=0)]
    # g1 = [np.mean(g1, axis=0)]
    # g2 = [np.mean(g2, axis=0)]
    # eQ_id = 0

    shape = [3, 4]
    for i in range(3):
        if i == 2:
            xlabel = r"$\theta$"
            xticks = [-2.5, 0, 2.5]
        else:
            xlabel = None
            xticks = []
        if i == 0:
            title = "DDPG\n" + titles[i]
        else:
            title = titles[i]
        axQ = make_spectrum_plot(
            shape,
            [i, 0],
            np.transpose(Q2[0][idx[i]]),
            "Blues",
            xlabel,
            r"$\tau$",
            xticks,
            [-2, 0, 2],
            title,
        )

        if i == 2:
            xlabel = r"$\theta$"
        else:
            xlabel = None
        if i == 0:
            title = "eQ\n" + titles[i]
        else:
            title = titles[i]
        _ = make_spectrum_plot(
            shape,
            [i, 1],
            np.transpose(Q1[eQ_id][idx[i]]),
            "Blues",
            xlabel,
            None,
            xticks,
            [],
            title,
        )

        if i == 2:
            xlabel = r"$\theta$"
        else:
            xlabel = None
        if i == 0:
            title = "DDPG\n" + titles[i]
        else:
            title = titles[i]
        axG = make_spectrum_plot(
            shape,
            [i, 2],
            np.transpose(g2[0][idx[i]]),
            "bwr",
            xlabel,
            None,
            xticks,
            [],
            title,
        )

        if i == 2:
            xlabel = r"$\theta$"
        else:
            xlabel = None
        if i == 0:
            title = "eQ\n" + titles[i]
        else:
            title = titles[i]
        _ = make_spectrum_plot(
            shape,
            [i, 3],
            np.transpose(g1[eQ_id][idx[i]]),
            "bwr",
            xlabel,
            None,
            xticks,
            [],
            title,
        )

    # plt.subplot2grid([4,4], [3, 0], colspan=2)
    # plt.colorbar(ax=axQ, cax=plt.gca(), orientation="horizontal")

    # plt.subplot2grid([4,4], [3, 2], colspan=2)
    # plt.colorbar(ax=axG, cax=plt.gca(), orientation="horizontal")

    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()


def plot_spectrum(spectrum, outfile, cmap="bwr"):
    # takes a spectrum with the following axes: (omega, theta, torque) as defined above
    # and saves it to a nicely formatted file.
    max_val = np.max(np.abs(spectrum))
    # if is_grad:
    #    cmp = "
    for idx, omega in enumerate(np.linspace(-8, 8, 9)):
        plt.subplot(3, 3, idx + 1)
        plt.imshow(
            np.transpose(spectrum[idx]),
            cmap=cmap,
            # vmin=-max_val,
            # vmax=max_val,
            origin="lower",
            extent=(np.pi, -np.pi, -2, 2),
        )
        plt.xlabel("theta")
        plt.ylabel("torque")
        plt.title(f"Omega = {omega}", fontsize=9)
        plt.tight_layout()
    if len(outfile) > 4:  # at least .png or .pdf
        plt.savefig(outfile)
    plt.close()


if __name__ == "__main__":
    Q_spectrum = np.load('alife-results/best-Qmaps.npy')
    grad_spectrum = np.load('alife-results/best-training-signals.npy')

    # spectra maps
    n_eQ = Q_spectrum.shape[0]//2
    eQ_id = (17*n_eQ)//20 # want it to be 17 normally, but fail gracefully
    plot_single_spectrum_figure(
        Q_spectrum[:n_eQ],
        Q_spectrum[n_eQ:],
        grad_spectrum[:n_eQ],
        grad_spectrum[n_eQ:],
        "alife-results/figure_3.png",
        eQ_id=eQ_id,
    )
