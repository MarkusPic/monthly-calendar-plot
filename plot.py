import matplotlib.pyplot as plt
import pandas as pd
import calendar
import seaborn as sns
from math import ceil


def monthly_calendar_figure(series, cols=3, height_cm=29.7, width_cm=42.0):
    """
    create a calendar with each month separate. week numbers as rows and day of the week names as columns

    :param series: pandas Timeseries with a daily DatetimeIndex
    :type series: pd.Series

    :param cols: number of columns of months in the figure
    :type cols: int

    :param width_cm: width of the plot in centimeter
    :type width_cm: float

    :param height_cm: height of the plot in centimeter
    :type height_cm: float

    :return: matplotlib figure with of the calendar
    :rtype: plt.Figure
    """
    ts = series.copy()
    # ts = series.resmaple('D')

    month_groupby = ts.groupby(ts.index.to_period('M'))
    rows = ceil(month_groupby.ngroups / cols)
    fig, axes = plt.subplots(rows, cols)
    # size is DIN A3
    fig.set_size_inches(h=height_cm / 2.54, w=width_cm / 2.54)
    axes = axes.T
    for (month, month_ts), ax in zip(month_groupby, axes.reshape(rows*cols, )): # month, month_ts
        df = pd.pivot(index=month_ts.index.weekofyear, columns=month_ts.index.weekday, values=month_ts)
        day = pd.pivot(index=month_ts.index.weekofyear, columns=month_ts.index.weekday, values=month_ts.index.day)

        cmap2 = plt.cm.get_cmap('RdYlGn')
        cmap2.set_gamma(100)
        ax = sns.heatmap(df, annot=day, square=False, linewidths=1, cmap=cmap2, vmin=None, vmax=None,
                         cbar=False, ax=ax, fmt='<8.0f',
                         annot_kws=dict(horizontalalignment='right',
                                        fontsize=8,
                                        verticalalignment='bottom'
                                        ))


        df.columns = list(calendar.day_abbr)

        cmap = plt.cm.get_cmap('RdYlGn_r')
        cmap.set_under('lightgray')

        ax = sns.heatmap(df, annot=True, square=False, linewidths=1, cmap=cmap, vmin=0.001, vmax=ts.max()+1,
                         cbar=False, ax=ax)
        ax.set_title(month.strftime('%Y - %B'))
        ax.xaxis.tick_top()
        ax.yaxis.set_ticks_position('none')
        ax.tick_params(axis=u'both', which=u'both', width=0, length=0.01)
        ax.tick_params(axis='y', which=u'both', rotation=0)

    # size is DIN A3
    fig.set_size_inches(h=height_cm / 2.54, w=width_cm / 2.54)
    return fig
