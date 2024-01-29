import logging
import os
from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML
from sklearn import preprocessing

from funman.server.query import FunmanResults


def animate_heat_map(my_df, frames):
    fig = plt.figure()

    vmin = my_df.min().min()
    vmax = my_df.max().max()
    # ax = sns.heatmap(data, vmin=0, vmax=1)
    # fig, ax = plt.subplots()
    # sns.heatmap(data, vmin=vmin, vmax=vmax, cmap="crest", ax=ax)

    def init():
        # plt.clf()
        fig.clear()
        data = my_df.loc[0, :]
        ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap="crest")
        ax.set_xlabel(data.columns[0][0])
        ax.set_ylabel(data.index.name)
        ax.set_title(data.columns[0][0])

    def animate(i):
        # plt.clf()
        fig.clear()
        data = my_df.loc[i, :]
        # ax.set_data(data)
        ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap="crest")
        ax.set_xlabel(data.columns[0][0])
        ax.set_ylabel(data.index.name)
        ax.set_title(f"{data.columns[0][0]}: time = {i}")

    anim = animation.FuncAnimation(
        fig,
        animate,
        interval=1000,
        frames=frames,
        init_func=init,
    )

    return anim


def plot_spatial_timeseries(
    results: FunmanResults,
    variables: Optional[List[str]] = None,
    outdir=None,
    fps=1,
):
    logging.getLogger("matplotlib.animation").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)

    df = results.dataframe(points=[results.parameter_space.true_points()[-1]])
    steps = len(df)
    parameters = results.model._parameter_names()
    vars = results.model._state_var_names()
    to_drop = (
        parameters
        + ["id", "label"]
        + [v for v in vars if variables is not None and v not in variables]
    )
    df = df.drop(columns=to_drop)

    # x = df.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # standard_scaler = preprocessing.StandardScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # # x_scaled = standard_scaler.fit_transform(x)
    # df = pd.DataFrame(x_scaled, columns =df.columns)

    df.columns = df.columns.str.split("_", expand=True)
    df = df.stack([1])
    df.index = df.index.set_names(["time"] + [df.columns[0][0]])

    anim_h = animate_heat_map(df, steps)
    if outdir:
        anim_h.save(
            os.path.join(outdir, "h.gif"),
            writer=animation.PillowWriter(fps=fps),
        )
    hh = HTML(anim_h.to_jshtml())
    dh = df.unstack().diff().fillna(0).stack([1]).rename(columns={"h": "dh"})
    anim_dh = animate_heat_map(dh, steps)
    if outdir:
        anim_dh.save(
            os.path.join(outdir, "dh.gif"),
            writer=animation.PillowWriter(fps=fps),
        )
    hdh = HTML(anim_dh.to_jshtml())

    return hh, hdh, anim_h, anim_dh
