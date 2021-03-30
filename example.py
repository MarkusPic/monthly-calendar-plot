from monthly_calendar_plot import monthly_calendar_figure
import pandas as pd
import numpy as np

if __name__ == '__main__':
    index = pd.date_range('2021-01-01', '2021-12-31', freq='D')
    np.random.seed(1)
    data = np.random.randint(0, 10, size=(len(index)))
    daily = pd.Series(index=index, data=data)
    fig = monthly_calendar_figure(series=daily, cols=3, cmap='RdYlGn_r', min_value=0.001, color_unter='lightgray',
                                  h=12, w=17)
    fig.savefig('example.png')
