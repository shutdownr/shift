# SHIFT: Interpretable Time Series Forecasting

SHIFT is an interpretable forecasting method for univariate time series forecasting. It relies on f-shapelets and subsequence similarity to make forecasts. 

## Benchmarks and demos
Clone the repository and install the required python libraries via pip

```shell
git clone https://github.com/shutdownr/shift.git
cd shift
python3 -m pip install -r requirements.txt
```

Create the data folder and manually download the data

```shell
mkdir data 
cd data
mkdir UTS
```

Download each of the following datasets and move them into /data/UTS<br>
Links for downloading the datasets:

- [CIF](https://zenodo.org/record/4656042)
- [NN5](https://zenodo.org/record/4656117)
- [Tourism](https://zenodo.org/record/4656096)
- [Weather](https://www.kaggle.com/datasets/thedevastator/weather-prediction?select=weather_prediction_dataset.csv)
- [M3 Monthly](https://forvis.github.io/data/M3_monthly_TSTS.csv)
- [M3 Quarterly](https://forvis.github.io/data/M3_quarterly_TSTS.csv)
- [M3 Yearly](https://forvis.github.io/data/M3_yearly_TSTS.csv)
- [M3 Other](https://forvis.github.io/data/M3_other_TSTS.csv)
- [M4 Hourly](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset?select=Hourly-train.csv)
- [M4 Weekly](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset?select=Weekly-train.csv)
- [M4 Yearly](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset?select=Yearly-train.csv)
- [Transactions](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data?select=transactions.csv)

Run the benchmarks or alternatively use the provided .pkl files for the results described in the paper.<br>
Caution: Running the benchmark with TimesNet will lead to extremely long runtimes. Edit *algorithms* in benchmark_backhorizon_horizon.py to change the algorithms to be included in the benchmark.<br>

```shell
python3 benchmark_ablation.py
python3 benchmark_backhorizon_horizon.py
python3 benchmark_by_sample.py
```

To visualize the results, generate the plots and run demos, use the provided jupyter notebook:

- ```visualization.ipynb```

## Usage of the model

Note: SHIFT does not require validation data. If you have validation instances, they can be merged with the training instances.<br>
Generate synthetic data with training and testing instances

```python
import numpy as np
# Define horizon and back horizon window lengths
h = 5
b = 10
# Sliding step size for generating train/test instances
stride_length = 1

# Generate synthetic sine wave data
sine_data = np.arange(500).astype(float)
sine_data *= np.pi / 16
sine_data = np.sin(sine_data)

# Split data into train and test
train_length = 0.8
split_point = int(train_length * len(sine_data))
timeseries_train = sine_data[:split_point]
timeseries_test = sine_data[split_point:]

# Split train/test part of the time series into instances
x_train = np.lib.stride_tricks.sliding_window_view(timeseries_train[:-h], b)[::stride_length, :]
y_train = np.lib.stride_tricks.sliding_window_view(timeseries_train[b:], h)[::stride_length, :]
x_test = np.lib.stride_tricks.sliding_window_view(timeseries_test[:-h], b)[::stride_length, :]
y_test = np.lib.stride_tricks.sliding_window_view(timeseries_test[b:], h)[::stride_length, :]
```

Run the model by optimizing hyperparameters, fitting on the training part of the time series and predicting for the test instances.<br>
For repeating patterns sampled at regular intervals, as in this synthetic use case, SHIFT produces perfect forecasts.

```python
from SHIFT import SHIFT
model = SHIFT()
# For synthetic sine data, the model option for chaining can be turned off to increase speed and performance
model.chain = False
# Optimize hyperparameters for the given time series
model.optimize_hyperparameters(timeseries_train, x_train, y_train)
# Make a forecast for each test instance
y_pred = model.fit_predict(timeseries_train, x_test, h)
```