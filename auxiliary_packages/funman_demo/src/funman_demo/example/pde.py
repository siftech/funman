import logging
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
    # fig = plt.figure()
    fig, ax = plt.subplots()
    data = my_df.loc[0, :]
    vmin = my_df.min().min()
    vmax = my_df.max().max()
    # ax = sns.heatmap(data, vmin=0, vmax=1)

    # def init():
    #     plt.clf()
    #     ax = sns.heatmap(data, vmin=0, vmax=1)

    def animate(i):
        plt.clf()
        data = my_df.loc[i, :]
        ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap="crest")
        return ax

    anim = animation.FuncAnimation(
        fig, animate, interval=1000, frames=frames  # init_func=init,
    )

    return anim


def plot_spatial_timeseries(
    results: FunmanResults, variables: Optional[List[str]] = None
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

    anim_h = animate_heat_map(df, steps)
    hh = HTML(anim_h.to_jshtml())
    dh = df.unstack().diff().fillna(0).stack([1]).rename(columns={"h": "dh"})
    anim_dh = animate_heat_map(dh, steps)
    hdh = HTML(anim_dh.to_jshtml())

    return hh, hdh, anim_h, anim_dh
