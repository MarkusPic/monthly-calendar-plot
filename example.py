from monthly_calendar_plot import monthly_calendar_figure
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Creating random dataset:
    index = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    np.random.seed(1)
    data = np.random.randint(0, 10, size=(len(index)))
    # making the data to a pandas.Series:
    daily = pd.Series(index=index, data=data)

    # actual plotting function:
    fig = monthly_calendar_figure(series=daily, cols=3, cmap='RdYlGn_r', min_value=0.001, color_unter='lightgray', color_bad='white')
    fig.set_size_inches(h=9, w=11)
    fig.savefig('example.png')
