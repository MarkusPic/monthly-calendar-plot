import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import copy
from matplotlib.colors import Normalize

from math import ceil

from monthly_calendar_plot.plot import _get_txt_coords


def _icon_monthly_calendar_figure(series, cols=3, cmap='RdYlGn_r', min_value=0.001, color_unter='lightgray', color_bad='grey',
                                  transpose_figure=True, value_max=None,
                                  month_fmt='%Y - %B', annotation_fmt=None, value_fmt='_.0f'):
    """
    Create a calendar with each month separate.
    Week numbers as rows and day of the week names as columns.

    Args:
        series (pandas.Series): Timeseries with a daily DatetimeIndex
        cols (int): number of columns of months in the figure
        cmap (str): Name of the color-map.
        min_value (float): Smallest value to be colored.
        value_max (float, Optional): Highest value in the data. Default: max in series + 1.
        color_unter (str): Color of the days below the value of `min_value`.
        color_bad (str): color of the days with a NaN in it.
        transpose_figure (bool): True (default): row-wise top to bottom | False: columns-wise left to right.
        month_fmt (str): Format of the monthly title.
        annotation_fmt (dict): optional define custom annotation format of the text.
        value_fmt (str): Format of the values used for the annotation.

    Returns:
        matplotlib.pyplot.Figure: figure with of the calendar
    """
    month_groups = series.groupby(series.index.to_period('M'))
    rows = ceil(month_groups.ngroups / cols)

    fig = plt.figure(tight_layout=True)  # type: plt.Figure
    axes = fig.subplots(rows, cols)

    # fig.get_layout_engine().set(h_pad=0.1, w_pad=0.1)

    if value_max is None:
        value_max = series.max() + 1

    # from left column to right column
    if transpose_figure:
        axes = axes.T

    if isinstance(cmap, str):
        cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_under(color_unter)
    cmap.set_bad(color_bad)

    weekday_names = list(calendar.day_abbr)

    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(month_groups), desc='monthly_calendar_figure')
    except ImportError:
        progress_bar = None

    there_is_a_progress_bar = progress_bar is not None

    base_annotation_fmt = dict(ha='center', va='center', size=10, family='sans-serif')

    if annotation_fmt is not None:
        base_annotation_fmt.update(annotation_fmt)

    annotation_fmt = base_annotation_fmt

    norm = Normalize(vmin=min_value, vmax=value_max)

    def _get_txt_color(value):
        bg_color = cmap(norm(value))
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
        return ("black", "white")[0 if (luminance > 0.6) else 1]

    for (month, month_ts), ax in zip(month_groups, axes.reshape(rows * cols, )):  # month, month_ts
        if there_is_a_progress_bar:
            progress_bar.set_postfix_str(month, refresh=True)

        ax: plt.Axes

        month_df = month_ts.index.isocalendar()
        month_df['value'] = month_ts
        month_df['day_of_month'] = month_ts.index.day
        month_df['year_week'] = month_df.year + month_df.week / 100
        df = pd.pivot(month_df, index='year_week', columns='day', values='value')

        n_weeks = df.index.size

        # from "1" to "Mon", ...
        df.columns = weekday_names
        # left, right, bottom, top
        im = ax.imshow(df.values, cmap=cmap, vmin=min_value, vmax=value_max, aspect='auto', extent=(0, 7, n_weeks, 0))
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(0, 7+1), minor=True)
        ax.set_yticks(np.arange(0, n_weeks+1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

        ax.set_xticks(np.arange(0.5, 7))
        # ax.set_xticklabels(['~'] * len(weekday_names))
        ax.set_xticklabels(weekday_names)
        ax.set_ylabel('')
        ax.set_title(month.strftime(month_fmt))
        # ax.set_title('~~~~')
        ax.xaxis.tick_top()
        # [str(x) for x in ax.get_yticklabels()]
        ax.set_yticks(np.arange(0.5, n_weeks))
        # ax.set_yticklabels([x.get_text()[5:].lstrip('0') for x in ax.get_yticklabels()])
        ax.set_yticklabels(['--']*len(ax.get_yticks()))
        # ax.set_yticklabels(((df.index - df.index.astype(int)).values*100).round(0).astype(int))
        ax.set_ylim(6, 0)
        # ax.yaxis.set_ticks_position('none')
        ax.tick_params(axis=u'both', which=u'both', width=0, length=0.01)
        ax.tick_params(axis='y', which=u'both', rotation=0)

        # ---------------------------------
        # day annotation
        pad = 0.125
        for week, dayofweek, day, value in month_df[['year_week', 'day', 'day_of_month', 'value']].values:
            row = df.index.tolist().index(week)
            # row = (week - month_df.year_week.min())*100
            col = dayofweek

            # Day of the month
            # ax.text(col-1+pad, row+pad*2, f'{day:2d}', ha='left', va='top', size=7, family='monospace',
            #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, lw=0.2))

            ax.text(*_get_txt_coords(col, row, annotation_fmt, pad), '--', **annotation_fmt, color=_get_txt_color(value))

        if there_is_a_progress_bar:
            progress_bar.update()

    if there_is_a_progress_bar:
        progress_bar.close()

    return fig


if __name__ == '__main__':
    # Creating random dataset:
    index = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    np.random.seed(1)
    data = np.random.randint(0, 10, size=(len(index)))
    # making the data to a pandas.Series:
    daily = pd.Series(index=index, data=data)

    # actual plotting function:
    fig = _icon_monthly_calendar_figure(series=daily, cols=3, cmap='RdYlGn_r', min_value=0.001, color_unter='lightgray', color_bad='white', annotation_fmt=dict(ha='right'))
    # 1280×640px
    fig.set_dpi(100)
    fig.set_size_inches(h=6.4, w=12.8)
    fig.suptitle('monthly_calendar_plot', weight='bold', size=32)
    # fig.tight_layout(pad=0.01, rect=[0.1, 0.1, 0.8, 0.8])
    # 80 pt inches (1 inch = 100 pt)
    fig.savefig('icon.png', bbox_inches='tight', pad_inches=1)
