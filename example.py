from monthly_calendar_plot import monthly_calendar_figure
import pandas as pd
import numpy as np

if __name__ == '__main__':
    index = pd.date_range('2019-01-01', '2019-12-31', freq='D')
    data = np.random.randint(0, 10, size=(len(index)))
    daily = pd.Series(index=index, data=data)
    fig = monthly_calendar_figure(series=daily)
    fig.tight_layout()
    fig.savefig('example.png', dpi=450)
