import copy

import numpy as np
import pandas as pd

from metrics import mae, mse, rmse, mape, smape, mase, owa, corr
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer


class ForecastingDataset:
    TS_VARIABLES = [
        "ts_train",
        "ts_full",
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test",
        "y_test",
    ]
    INSTANCES = TS_VARIABLES[2:]
    _METRICS = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "mase": mase,
        "owa": owa,
        "corr": corr,
    }

    def __init__(
        self,
        dataframe,
        horizon=5,
        backhorizon=10,
        train_size=0.6,
        val_size=0.2,
        stride_length=1,
    ):
        self._horizon = horizon
        self._backhorizon = backhorizon
        self._train_size = train_size
        self._val_size = val_size
        self._stride_length = stride_length
        self._is_scaled = False

        self.is_multivariate = False
        self.is_ragged = False

        self.process(dataframe)

    def get_attr_flat(self, attr):
        assert attr in ForecastingDataset.TS_VARIABLES

        attribute = getattr(self, attr)
        if self.is_ragged:
            if self.is_multivariate:
                out = [[] for _ in range(len(attribute[0]))]
                for sample in attribute:
                    for i, channel in enumerate(sample):
                        out[i].extend(channel)
                return out
            else:
                return np.concatenate(attribute)
        else:
            if self.is_multivariate:
                transposed = attribute.transpose(0, 2, 1, 3)
                return transposed.reshape(
                    transposed.shape[0] * transposed.shape[1],
                    transposed.shape[2],
                    transposed.shape[3],
                ).transpose(1, 0, 2)
            else:
                return attribute.reshape(-1, attribute.shape[-1])

    def evaluate(
        self, y_pred, metrics=["mae", "mse"], invert_scaling=True, is_scaled=False
    ):
        assert is_scaled == self._is_scaled
        if is_scaled and invert_scaling:
            raise NotImplementedError("!")

        y_pred_flat = y_pred.copy()
        y_true_flat = self.y_test.copy()
        X_test_flat = self.X_test.copy()
        # Flatten everything to 2D to force an ndarray (otherwise ragged ts cause trouble)
        while type(y_pred_flat[0][0]) in [list, np.ndarray]:
            y_pred_flat = [val for list in y_pred_flat for val in list]
            y_true_flat = [val for list in y_true_flat for val in list]
            X_test_flat = [val for list in X_test_flat for val in list]
        y_pred_flat = np.array(y_pred_flat)
        y_true_flat = np.array(y_true_flat)
        X_test_flat = np.array(X_test_flat)

        errors = {}
        for metric in metrics:
            assert metric in ForecastingDataset._METRICS
            if metric in ["mase", "owa"]:
                error = ForecastingDataset._METRICS[metric](
                    X_test_flat, y_true_flat, y_pred_flat
                )
            else:
                error = ForecastingDataset._METRICS[metric](y_true_flat, y_pred_flat)
            errors[metric] = error
        return errors

    def _transform(self, data: np.ndarray, scaler, inverse: bool):
        if inverse:
            return scaler.inverse_transform(data)
        else:
            return scaler.transform(data)

    def _scale(
        self, scaling: str = "Standard", by_sample: bool = False, inverse: bool = False
    ):
        if scaling == "Standard":
            Scaler = StandardScaler
        elif scaling == "MinMax":
            Scaler = MinMaxScaler
        elif scaling == "Quantile":
            Scaler = QuantileTransformer
        else:
            raise NotImplementedError("Scaling type not implemented")

        nr_samples = len(self.ts_train)
        if self.is_multivariate:
            nr_channels = len(self.ts_train[0])
            if by_sample:
                if not inverse:
                    self._scalers = np.array([Scaler() for _ in range(nr_samples)])
                for i in range(nr_samples):
                    if not inverse:
                        transposed = self.ts_train[i].T
                        self._scalers[i].fit(transposed)
                    for attr in ForecastingDataset.INSTANCES:
                        values = getattr(self, attr)
                        transformed = self._transform(
                            values[i].transpose(1, 2, 0).reshape(-1, nr_channels),
                            self._scalers[i],
                            inverse,
                        )
                        transformed = transformed.reshape(
                            values.shape[2], values.shape[3], nr_channels
                        ).transpose(2, 0, 1)
                        values[i] = transformed
                        setattr(self, attr, values)
            else:
                if not inverse:
                    self._scalers = np.array([Scaler()])
                    transposed = self.ts_train.transpose(0, 2, 1).reshape(
                        -1, nr_channels
                    )
                    self._scalers[0].fit(transposed)
                for attr in ForecastingDataset.INSTANCES:
                    values = getattr(self, attr).transpose(0, 2, 3, 1)
                    transformed = self._transform(
                        values.reshape(-1, nr_channels), self._scalers[0], inverse
                    )
                    transformed = transformed.reshape(values.shape).transpose(
                        0, 3, 1, 2
                    )
                    setattr(self, attr, transformed)
        else:
            if not inverse:
                self._scalers = np.array([Scaler()])
            if by_sample:
                if not inverse:
                    self._scalers[0].fit(self.ts_train.T)
                for attr in ForecastingDataset.INSTANCES:
                    values = getattr(self, attr).transpose(1, 2, 0)
                    transformed = self._transform(
                        values.reshape(-1, nr_samples), self._scalers[0], inverse
                    )
                    transformed = transformed.reshape(values.shape).transpose(2, 0, 1)
                    setattr(self, attr, transformed)
            else:
                if not inverse:
                    self._scalers[0].fit(self.ts_train.flatten().reshape(-1, 1))
                for attr in ForecastingDataset.INSTANCES:
                    values = getattr(self, attr)
                    reshaped = values.flatten()
                    transformed = self._transform(
                        reshaped.reshape(-1, 1), self._scalers[0], inverse
                    ).reshape(-1)
                    transformed = transformed.reshape(values.shape)
                    setattr(self, attr, transformed)
        self._is_scaled = not inverse

    def _scale_ragged(self, scaling="Standard", by_sample=False, inverse=False):
        if scaling == "Standard":
            Scaler = StandardScaler
        elif scaling == "MinMax":
            Scaler = MinMaxScaler
        elif scaling == "Quantile":
            Scaler = QuantileTransformer
        else:
            raise NotImplementedError("Scaling type not implemented")

        nr_samples = len(self.ts_train)
        if self.is_multivariate:
            nr_channels = len(self.ts_train[0])
            if by_sample:
                if not inverse:
                    self._scalers = np.array(
                        [
                            [Scaler() for _ in range(nr_channels)]
                            for _ in range(nr_samples)
                        ]
                    )
                for i in range(nr_samples):
                    for j in range(nr_channels):
                        if not inverse:
                            reshaped = self.ts_train[i][j].reshape(-1, 1)
                            self._scalers[i][j].fit(reshaped)
                        for attr in ForecastingDataset.INSTANCES:
                            values = getattr(self, attr)
                            transformed = self._transform(
                                values[i][j].reshape(-1, 1),
                                self._scalers[i][j],
                                inverse,
                            )
                            values[i][j] = transformed.reshape(values[i][j].shape)
                            setattr(self, attr, values)
            else:
                if not inverse:
                    self._scalers = np.array([Scaler() for _ in range(nr_channels)])
                    ts_by_channel = [[] for _ in range(nr_channels)]
                    for mts in self.ts_train:
                        for i, channel in enumerate(mts):
                            ts_by_channel[i].append(channel)
                    for scaler, ts in zip(self._scalers, ts_by_channel):
                        scaler.fit(np.concatenate(ts).reshape(-1, 1))
                for attr in ForecastingDataset.INSTANCES:
                    values = getattr(self, attr)
                    all_scaled = []
                    for sample in values:
                        scaled_sample = []
                        for scaler, sample_values in zip(self._scalers, sample):
                            transformed = self._transform(
                                sample_values.reshape(-1, 1), scaler, inverse
                            )
                            transformed = transformed.reshape(sample_values.shape)
                            scaled_sample.append(transformed)
                        all_scaled.append(scaled_sample)
                    setattr(self, attr, all_scaled)
        else:
            if by_sample:
                if not inverse:
                    self._scalers = np.array([Scaler() for _ in range(nr_samples)])
                    for scaler, sample in zip(self._scalers, self.ts_train):
                        scaler.fit(np.array(sample).reshape(-1, 1))
                for attr in ForecastingDataset.INSTANCES:
                    values = getattr(self, attr)
                    scaled = []
                    for scaler, sample_values in zip(self._scalers, values):
                        sample_values = np.array(sample_values)
                        transformed = self._transform(
                            sample_values.reshape(-1, 1), scaler, inverse
                        )
                        transformed = transformed.reshape(sample_values.shape)
                        scaled.append(transformed)
                    setattr(self, attr, scaled)
            else:
                if not inverse:
                    self._scalers = np.array([Scaler()])
                    reshaped = np.array(
                        [[value] for sublist in self.ts_train for value in sublist]
                    )
                    self._scalers[0].fit(reshaped)
                for attr in ForecastingDataset.INSTANCES:
                    values = getattr(self, attr)
                    scaled = []
                    for sample_values in values:
                        sample_values = np.array(sample_values)
                        transformed = self._transform(
                            sample_values.reshape(-1, 1), self._scalers[0], inverse
                        )
                        transformed = transformed.reshape(sample_values.shape)
                        scaled.append(transformed)
                    setattr(self, attr, scaled)
        self._is_scaled = not inverse

    def scale(self, scaling="Standard", by_sample=False):
        assert not self._is_scaled
        self._scale_by_sample = by_sample
        if self.is_ragged:
            return self._scale_ragged(scaling, by_sample, False)
        else:
            return self._scale(scaling, by_sample, False)

    def inverse_scale(self):
        assert self._is_scaled
        if self.is_ragged:
            return self._scale_ragged(by_sample=self._scale_by_sample, inverse=True)
        else:
            return self._scale(by_sample=self._scale_by_sample, inverse=True)

    def reset_data(self):
        for var in ForecastingDataset.TS_VARIABLES:
            setattr(self, var, [])

    def process(self, dataframe):
        self.reset_data()

        self.is_multivariate = type(dataframe.index) == pd.MultiIndex

        if self.is_multivariate:
            common_length = dataframe.shape[1]
            ts_variables = [[] for _ in ForecastingDataset.TS_VARIABLES]
            for _, group in dataframe.groupby(level=0):
                # Reset self.TS_VARIABLES
                self.reset_data()
                for _, ts in group.iterrows():
                    ts = ts.to_numpy()
                    ts = ts[~np.isnan(ts)]  # Remove nan values (for ragged datasets)
                    if len(ts) != common_length:
                        self.is_ragged = True
                    self.train_test_split(ts)
                for i, var in enumerate(ForecastingDataset.TS_VARIABLES):
                    ts_variables[i].append(getattr(self, var))
            for var, ts_variable in zip(ForecastingDataset.TS_VARIABLES, ts_variables):
                setattr(self, var, ts_variable)
        else:
            common_length = dataframe.shape[1]
            for i in range(len(dataframe)):
                ts = dataframe.iloc[i, :].to_numpy()
                ts = ts[~np.isnan(ts)]  # Remove nan values (for ragged datasets)
                if len(ts) != common_length:
                    self.is_ragged = True
                self.train_test_split(ts)

        if not self.is_ragged:
            for attr in ForecastingDataset.TS_VARIABLES:
                setattr(self, attr, np.array(getattr(self, attr)))

    def train_test_split(self, ts):
        def generate_x_y(time_series):
            if self._backhorizon + self._horizon > len(time_series):
                return np.array([], dtype=float), np.array([], dtype=float)
            x_windows = np.lib.stride_tricks.sliding_window_view(
                time_series[: -self._horizon], self._backhorizon
            )[:: self._stride_length, :]
            y_windows = np.lib.stride_tricks.sliding_window_view(
                time_series[self._backhorizon :], self._horizon
            )[:: self._stride_length, :]
            return x_windows.astype(float), y_windows.astype(float)

        train_stop = int((self._train_size + self._val_size) * len(ts))

        X_train, y_train = generate_x_y(ts[:train_stop])
        # Go back only horizon steps to not overlap inputs of test instances and training instances
        X_test, y_test = generate_x_y(ts[train_stop - self._horizon :])
        if len(X_test) == 0 or len(X_train) == 0:
            return

        val_stop = int(
            (self._val_size / (self._val_size + self._train_size)) * len(X_train)
        )

        if val_stop > 0:
            self.X_val.append(X_train[-val_stop:])
            self.y_val.append(y_train[-val_stop:])
            self.X_train.append(X_train[:-val_stop])
            self.y_train.append(y_train[:-val_stop])
        else:
            self.X_val.append(np.array([]))
            self.y_val.append(np.array([]))
            self.X_train.append(X_train)
            self.y_train.append(y_train)
        # maybe need to cast to np array
        self.ts_train.append(ts[:train_stop])
        self.ts_full.append(ts)

        self.X_test.append(X_test)
        self.y_test.append(y_test)


def flatten(dataset):
    flattened = copy.deepcopy(dataset)
    original_shapes = {}
    for k in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        original_shapes[k] = [flattened[k][i].shape for i in range(len(flattened[k]))]
        flattened[k] = np.array([i for l in flattened[k] for i in l])
    return flattened, original_shapes


def inverse_flatten(dataset, original_shapes):
    inversed = copy.deepcopy(dataset)
    for k, shapes in original_shapes.items():
        inversed_values = []
        i = 0
        for shape in shapes:
            inversed_values.append(inversed[k][i : i + shape[0]])
            i += shape[0]
        inversed[k] = inversed_values
    return inversed


def inverse_flatten_predictions(predictions, original_shapes):
    y_test_shapes = original_shapes["y_test"]
    inversed_predictions = []
    i = 0
    for shape in y_test_shapes:
        inversed_predictions.append(predictions[i : i + shape[0]])
        i += shape[0]
    return np.array(inversed_predictions, dtype=object)


def scale(dataset):
    def scale_feature(feature, scaler):
        original_shape = feature.shape
        feature = np.array([i for l in feature for i in l])
        transformed = remove_dimension(scaler.transform(add_dimension(feature)))
        return transformed.reshape(original_shape).astype(float)

    scalers = []
    scaled = copy.deepcopy(dataset)
    for i in range(len(scaled["ts_raw"])):
        scaler = MinMaxScaler((0, 1))
        scaled["ts_raw"][i] = remove_dimension(
            scaler.fit_transform(add_dimension(scaled["ts_raw"][i]))
        )
        scaled["ts_full"][i] = remove_dimension(
            scaler.transform(add_dimension(scaled["ts_full"][i]))
        )
        scaled["X_train"][i] = scale_feature(scaled["X_train"][i], scaler)
        scaled["y_train"][i] = scale_feature(scaled["y_train"][i], scaler)
        scaled["X_test"][i] = scale_feature(scaled["X_test"][i], scaler)
        scaled["y_test"][i] = scale_feature(scaled["y_test"][i], scaler)
        if len(scaled["X_val"][i]) > 0:
            scaled["X_val"][i] = scale_feature(scaled["X_val"][i], scaler)
            scaled["y_val"][i] = scale_feature(scaled["y_val"][i], scaler)
        scalers.append(scaler)
    return scaled, scalers


def inverse_scale_feature(feature, scaler):
    feature = np.array(feature)
    original_shape = feature.shape
    while len(feature.shape) > 1:
        feature = np.array([i for l in feature for i in l])
    transformed = remove_dimension(scaler.inverse_transform(add_dimension(feature)))
    return transformed.reshape(original_shape)


def inverse_scale(dataset, scalers):
    inversed = copy.deepcopy(dataset)
    for i in range(len(inversed["ts_raw"])):
        scaler = scalers[i]

        inversed["ts_raw"][i] = remove_dimension(
            scaler.inverse_transform(add_dimension(inversed["ts_raw"][i]))
        )
        inversed["ts_full"][i] = remove_dimension(
            scaler.inverse_transform(add_dimension(inversed["ts_full"][i]))
        )

        inversed["X_train"][i] = inverse_scale_feature(inversed["X_train"][i], scaler)
        inversed["y_train"][i] = inverse_scale_feature(inversed["y_train"][i], scaler)
        inversed["X_test"][i] = inverse_scale_feature(inversed["X_test"][i], scaler)
        inversed["y_test"][i] = inverse_scale_feature(inversed["y_test"][i], scaler)
        if len(inversed["X_val"][i] > 0):
            inversed["X_val"][i] = inverse_scale_feature(inversed["X_val"][i], scaler)
            inversed["y_val"][i] = inverse_scale_feature(inversed["y_val"][i], scaler)

    return inversed


def inverse_scale_predictions(predictions, scalers):
    inversed = []
    for pred, scaler in zip(predictions, scalers):
        inversed.append(inverse_scale_feature(pred, scaler))
    return np.array(inversed, dtype=object)


def add_dimension(array):
    return array.reshape(array.shape + (1,))


def remove_dimension(array):
    return array.reshape(array.shape[: len(array.shape) - 1])


def train_test_split(ts, horizon, input_size, train_size, val_size, stride_length):
    def generate_x_y(data, horizon, input_size, stride_length):
        if input_size + horizon > len(data):
            return np.array([]), np.array([])
        x_windows = np.lib.stride_tricks.sliding_window_view(
            data[:-horizon], input_size
        )[::stride_length, :]
        y_windows = np.lib.stride_tricks.sliding_window_view(
            data[input_size:], horizon
        )[::stride_length, :]
        return x_windows, y_windows

    train_stop = int((train_size + val_size) * len(ts))

    train = ts[:train_stop]
    # Go back only horizon steps to not overlap inputs of test instances and training instances
    test = ts[train_stop - horizon :]

    X_train, y_train = generate_x_y(train, horizon, input_size, stride_length)
    X_test, y_test = generate_x_y(test, horizon, input_size, stride_length)
    if len(X_test) == 0 or len(X_train) == 0:
        return {}

    if val_size > 0:
        val_split = int((val_size / (val_size + train_size)) * len(X_train))
        if val_split == 0:
            X_val = np.array([])
            y_val = np.array([])
        else:
            X_val = X_train[-val_split:]
            y_val = y_train[-val_split:]
            X_train = X_train[:-val_split]
            y_train = y_train[:-val_split]
    else:
        X_val = np.array([])
        y_val = np.array([])
    return {
        "ts_raw": np.array(ts[:train_stop]),
        "ts_full": np.array(ts),
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


# ------------------------------------------------
# ------------------ UNIVARIATE ------------------
# ------------------------------------------------


def read_dataset(
    dataset, train_size=0.6, val_size=0.2, stride_length=1, backhorizon=10, horizon=10
):
    data_path = "data/UTS/"

    cif_path = data_path + "cif_2016_dataset.tsf"
    nn5_path = data_path + "nn5_daily_dataset_without_missing_values.tsf"
    tourism_path = data_path + "tourism_monthly_dataset.tsf"
    weather_path = data_path + "weather_prediction_dataset.csv"

    m4_hourly_path = data_path + "Hourly-train.csv"
    m4_weekly_path = data_path + "Weekly-train.csv"
    m4_yearly_path = data_path + "Yearly-train.csv"

    m3_monthly_path = data_path + "M3_monthly_TSTS.csv"
    m3_quarterly_path = data_path + "M3_quarterly_TSTS.csv"
    m3_yearly_path = data_path + "M3_yearly_TSTS.csv"
    m3_other_path = data_path + "M3_other_TSTS.csv"

    transactions_path = data_path + "transactions.csv"

    if dataset == "cif":
        df = read_cif(cif_path)
    elif dataset == "nn5":
        df = read_nn5(nn5_path)
    elif dataset == "tourism":
        df = read_tourism(tourism_path)
    elif dataset == "weather":
        df = read_weather(weather_path)
    elif dataset == "m4_h":
        df = read_m4(m4_hourly_path)
    elif dataset == "m4_w":
        df = read_m4(m4_weekly_path)
    elif dataset == "m4_y":
        df = read_m4(m4_yearly_path)
    elif dataset == "m3_m":
        df = read_m3(m3_monthly_path)
    elif dataset == "m3_q":
        df = read_m3(m3_quarterly_path)
    elif dataset == "m3_y":
        df = read_m3(m3_yearly_path)
    elif dataset == "m3_o":
        df = read_m3(m3_other_path)
    elif dataset == "transactions":
        df = read_transactions(transactions_path)
    else:
        print("Attempting to read unknown dataset")
        raise NotImplementedError()

    train_test = {
        "ts_raw": [],
        "ts_full": [],
        "X_train": [],
        "X_val": [],
        "X_test": [],
        "y_train": [],
        "y_val": [],
        "y_test": [],
    }

    for i in range(len(df)):
        ts = df.iloc[i, :].to_numpy()
        ts = ts[~np.isnan(ts)]

        split_dict = train_test_split(
            ts, horizon, backhorizon, train_size, val_size, stride_length
        )
        for k, v in split_dict.items():
            train_test[k].append(v)

    for k, v in train_test.items():
        numpy_converted = np.array(v, dtype=object)
        if len(numpy_converted.shape) == 1:
            train_test[k] = numpy_converted
        else:
            train_test[k] = np.array(v)
    train_test["is_multivariate"] = False
    return train_test


def read_cif(path):
    df = pd.read_csv(
        path,
        sep=":|,",
        encoding="cp1252",
        engine="python",
        header=None,
        index_col=0,
        skiprows=16,
    )
    # Filter for 12 months forecasting horizon
    df = df[df.iloc[:, 0] == 12]
    return df.iloc[:, 1:]


def read_nn5(path):
    df = pd.read_csv(
        path, sep=":|,", engine="python", header=None, index_col=0, skiprows=19
    )
    return df.iloc[:, 1:]


def read_tourism(path):
    df = pd.read_csv(
        path,
        sep=":",
        encoding="cp1252",
        engine="python",
        header=None,
        index_col=0,
        skiprows=15,
    )
    df = df.loc[:, 2].str.split(",", expand=True)
    df = df.astype("float")
    return df


def read_weather(path):
    df = pd.read_csv(path, sep=",")
    columns = df.columns
    temperature_columns = columns.str.endswith("temp_mean")
    df = df.loc[:, temperature_columns]
    df = df.T
    return df


def read_m4(path):
    df = pd.read_csv(path)
    df = df.iloc[:, 1:]

    return df


def read_m3(path):
    df = pd.read_csv(path)
    df = pd.DataFrame(df.groupby("series_id")["value"])
    df = df.iloc[:, 1]
    all_rows = []
    for row in df:
        all_rows.append(np.array(row))

    return pd.DataFrame(all_rows)


def read_transactions(path):
    df = pd.read_csv(path)
    df = pd.pivot(df, index=["store_nbr"], columns=["date"], values=["transactions"])
    df.columns = range(df.columns.size)
    all_rows = []
    for _, row in df.iterrows():
        row.dropna(inplace=True)
        all_rows.append(np.array(row))

    return pd.DataFrame(all_rows)
