import matplotlib.pyplot as plt
import pandas as pd
import calendar
import copy
import seaborn as sns
from math import ceil


def monthly_calendar_figure(series, cols=3, cmap='RdYlGn_r', min_value=0.001, color_unter='lightgray', h=12, w=17):
    """
    create a calendar with each month separate. week numbers as rows and day of the week names as columns

    Args:
        series (pandas.Series): Timeseries with a daily DatetimeIndex
        cols (int): number of columns of months in the figure
        height_cm (float): width of the plot in centimeter, default=DIN A3
        width_cm (float): height of the plot in centimeter, default=DIN A3

    Returns:
        matplotlib.pyplot.Figure: figure with of the calendar
    """
    month_groups = series.groupby(series.index.to_period('M'))
    rows = ceil(month_groups.ngroups / cols)
    fig, axes = plt.subplots(rows, cols)

    fig.set_size_inches(h=h, w=w)
    # (left, bottom, right, top)
    fig.tight_layout(pad=1.0, #h_pad=h_pad, w_pad=w_pad,
                     rect=(0, 0, 1, 0.97)
                     )

    # from left column to right column
    axes = axes.T

    if isinstance(cmap, str):
        cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_under(color_unter)

    weekday_names = list(calendar.day_abbr)

    for (month, month_ts), ax in zip(month_groups, axes.reshape(rows * cols, )):  # month, month_ts
        print(month)
        month_df = month_ts.index.isocalendar()
        month_df['ts'] = month_ts
        month_df['day_of_month'] = month_ts.index.day
        month_df['year_week'] = month_df.year + month_df.week / 100
        df = pd.pivot(month_df, index='year_week', columns='day', values='ts')

        # from "1" to "Mon", ...
        df.columns = weekday_names
        ax = sns.heatmap(df, annot=True, square=False, linewidths=1, cmap=cmap, vmin=min_value, vmax=series.max() + 1,
                         cbar=False, ax=ax)

        ax.set_ylabel('')
        ax.set_title(month.strftime('%Y - %B'))
        ax.xaxis.tick_top()
        # [str(x) for x in ax.get_yticklabels()]
        ax.set_yticklabels(((df.index - df.index.astype(int)).values*100).round(0).astype(int))
        ax.set_ylim(6, 0)
        # ax.yaxis.set_ticks_position('none')
        ax.tick_params(axis=u'both', which=u'both', width=0, length=0.01)
        ax.tick_params(axis='y', which=u'both', rotation=0)

        # ---------------------------------
        # day annotation
        for week, dayofweek, day in month_df[['year_week', 'day', 'day_of_month']].values:
            row = df.index.tolist().index(week)
            # row = (week - month_df.year_week.min())*100
            col = dayofweek
            ax.text(col-1+0.1, row+0.2, f'{day:2d}', ha='left', va='top', size=7, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # day = pd.pivot(month_df, index='week', columns='day', values='day_of_month')
        # cmap2 = plt.cm.get_cmap('RdYlGn')
        # cmap2.set_gamma(100)
        # ax = sns.heatmap(df, annot=day, square=False, linewidths=1, cmap=cmap2, vmin=None, vmax=None,
        #                  cbar=False, ax=ax, fmt='0.0f',
        #                  annot_kws=dict(horizontalalignment='legt',
        #                                 fontsize=8,
        #                                 verticalalignment='top'
        #                                 ))

    # fig.tight_layout()
    return fig
