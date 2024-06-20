import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from pathlib import Path

comp_dir = Path('../dataset')

holidays_events = pd.read_csv(
    comp_dir / "holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date']
)
holidays_events = holidays_events.set_index('date').to_period('D')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date']
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    .loc['2017']
)

X = average_sales.to_frame(name='sales')
X["week"] = X.index.week
X["day"] = X.index.dayofweek

# Custom seasonal plot function for line trends
def custom_seasonal_line_plot(data, y, period, freq):
    plt.figure(figsize=(14, 7))
    data['period'] = data.index.map(lambda x: getattr(x, period))
    data['freq'] = data.index.map(lambda x: getattr(x, freq))
    avg_data = data.groupby('freq')[y].mean()
    plt.plot(avg_data, marker='o')
    plt.title(f'Seasonal Plot: {period} by {freq}')
    plt.xlabel(freq.capitalize())
    plt.ylabel('Average Sales')
    plt.grid(True)
    plt.show()

# Plot seasonal data with line trends
custom_seasonal_line_plot(X, y='sales', period='week', freq='day')
