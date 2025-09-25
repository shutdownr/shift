import copy
import time

# import keras
import numpy as np
import random
from sklearn.neighbors import KNeighborsRegressor
import torch
import torch.nn as nn

from statsforecast.models import AutoARIMA, AutoTheta

# from other_models.NBEATS import NBeatsNet
from other_models.TimesNet import Model as TimesNet
from other_models.DLinear import Model as DLinear
from other_models.FEDformer import Model as FEDformer
from other_models.Nonstationary_Transformer import Model as Nonstationary_Transformer
from other_models.train_model import train_model, predict, _acquire_device
from SHIFT import SHIFT
from utility import merge_train_val


def reshape_dataset(dataset, order="NLC"):
    if order == "NCL":
        if len(dataset["X_train"].shape) < 3:
            X_train = dataset["X_train"].reshape(
                dataset["X_train"].shape[0], 1, dataset["X_train"].shape[1]
            )
            y_train = dataset["y_train"].reshape(
                dataset["y_train"].shape[0], 1, dataset["y_train"].shape[1]
            )
            X_val = dataset["X_val"].reshape(
                dataset["X_val"].shape[0], 1, dataset["X_val"].shape[1]
            )
            y_val = dataset["y_val"].reshape(
                dataset["y_val"].shape[0], 1, dataset["y_val"].shape[1]
            )
            X_test = dataset["X_test"].reshape(
                dataset["X_test"].shape[0], 1, dataset["X_test"].shape[1]
            )
        else:
            X_train = dataset["X_train"].transpose(1, 0, 2)
            y_train = dataset["y_train"].transpose(1, 0, 2)
            X_val = dataset["X_val"].transpose(1, 0, 2)
            y_val = dataset["y_val"].transpose(1, 0, 2)
            X_test = dataset["X_test"].transpose(1, 0, 2)
    elif order == "NLC":
        if len(dataset["X_train"].shape) < 3:
            X_train = dataset["X_train"].reshape(
                dataset["X_train"].shape[0], dataset["X_train"].shape[1], 1
            )
            y_train = dataset["y_train"].reshape(
                dataset["y_train"].shape[0], dataset["y_train"].shape[1], 1
            )
            X_val = dataset["X_val"].reshape(
                dataset["X_val"].shape[0], dataset["X_val"].shape[1], 1
            )
            y_val = dataset["y_val"].reshape(
                dataset["y_val"].shape[0], dataset["y_val"].shape[1], 1
            )
            X_test = dataset["X_test"].reshape(
                dataset["X_test"].shape[0], dataset["X_test"].shape[1], 1
            )
        else:
            X_train = dataset["X_train"].transpose(1, 2, 0)
            y_train = dataset["y_train"].transpose(1, 2, 0)
            X_val = dataset["X_val"].transpose(1, 2, 0)
            y_val = dataset["y_val"].transpose(1, 2, 0)
            X_test = dataset["X_test"].transpose(1, 2, 0)
    return X_train, y_train, X_val, y_val, X_test


# Naive forecast
def predict_naive(dataset):
    # Note: dataset is flattened
    horizon = dataset["y_train"][0].shape[-1]
    if dataset["is_multivariate"]:
        all_pred = []
        total_inference_time = 0
        for channel in dataset["X_test"]:
            t0 = time.time()
            y_pred = np.array([np.full(horizon, X[-1]) for X in channel])
            total_inference_time += time.time() - t0
            all_pred.append(y_pred)
        return np.array(all_pred), total_inference_time
    else:
        t0 = time.time()
        y_pred = np.array([np.full(horizon, X[-1]) for X in dataset["X_test"]])
        total_inference_time = time.time() - t0
        return y_pred, total_inference_time


# =======================================================================
# ====================Statistical forecasting methods====================
# =======================================================================


# Automatically optimized model from the theta family; https://www.researchgate.net/publication/223049702_The_theta_model_A_decomposition_approach_to_forecasting
def predict_theta(dataset):
    def _fit_predict_theta(ts, horizon):
        model = AutoTheta()
        model.fit(ts)
        t0 = time.time()
        pred = model.predict(horizon)["mean"]
        time_taken = time.time() - t0
        return pred, time_taken

    # Note: dataset is flattened; For multivariate data, theta is applied per-channel
    horizon = dataset["y_train"][0].shape[-1]
    total_inference_time = 0
    y_pred = []
    for ts in dataset["X_test"]:
        if dataset["is_multivariate"] and len(ts.shape) > 1:
            pred_multivariate = []
            for channel in ts:
                pred, time_taken = _fit_predict_theta(channel, horizon)
                pred_multivariate.append(pred)
                total_inference_time += time_taken
            y_pred.append(pred_multivariate)
        else:
            pred, time_taken = _fit_predict_theta(ts, horizon)
            total_inference_time += time_taken
            y_pred.append(pred)

    return np.array(y_pred, dtype=object), total_inference_time


# Automatically optimized ARIMA; https://www.jstor.org/stable/pdf/2284333.pdf
def predict_arima(dataset):
    def _fit_predict_arima(ts, horizon):
        model = AutoARIMA()
        try:
            model.fit(ts)
            t0 = time.time()
            pred = model.predict(horizon)["mean"]
            time_taken = time.time() - t0
        except ZeroDivisionError:
            print("AutoARIMA bug")
            t0 = time.time()
            pred = np.full(horizon, np.median(ts))
            time_taken = time.time() - t0
        return pred, time_taken

    # Note: dataset is flattened; For multivariate data, arima is applied per-channel
    horizon = dataset["y_train"][0].shape[-1]
    if dataset["X_test"][0].shape[-1] <= 5:
        print("Fallback to naive prediction for short backhorizons")
        return predict_naive(dataset)
    total_inference_time = 0
    y_pred = []
    for ts in dataset["X_test"]:
        if dataset["is_multivariate"]:
            pred_multivariate = []
            for channel in ts:
                pred, time_taken = _fit_predict_arima(channel, horizon)
                pred_multivariate.append(pred)
                total_inference_time += time_taken
            y_pred.append(pred_multivariate)
        else:
            pred, time_taken = _fit_predict_arima(ts, horizon)
            total_inference_time += time_taken
            y_pred.append(pred)

    return np.array(y_pred, dtype=object), total_inference_time


def predict_knn(dataset):
    total_inference_time = 0
    samples, output_length = (
        dataset["X_test"].shape[0],
        dataset["y_train"].shape[1],
    )
    predictions = np.zeros((samples, output_length))
    for index in range(dataset["y_train"].shape[1]):
        k = np.sqrt(len(dataset["X_train"])).astype(int)
        model = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        model.fit(
            dataset["X_train"],
            dataset["y_train"][:, index],
        )
        t0 = time.time()
        predictions[:, index] = model.predict(dataset["X_test"])
        total_inference_time += time.time() - t0
    return predictions, total_inference_time


def predict_shift(dataset):
    # If validation data is present, merge this into the training data
    if len(dataset["X_val"]) > 0:
        data = merge_train_val(dataset)
    else:
        data = copy.deepcopy(dataset)

    total_inference_time = 0

    y_pred = []
    if dataset["is_multivariate"] and len(data["ts_raw"][0].shape) > 1:
        # Note: Multivariate data with multiple samples
        raise NotImplementedError("Not implemented for MTS with multiple samples")
    else:
        # Note: Unvariate data with multiple samples or
        #       Multivariate data with one sample (in this case each channel is a ts_raw)
        for index in range(len(data["ts_raw"])):
            h = len(data["y_train"][index][0])
            timeseries = data["ts_raw"][index]

            model = SHIFT()
            model.optimize_hyperparameters(
                timeseries, data["X_train"][index], data["y_train"][index]
            )

            t0 = time.time()
            pred = model.fit_predict(timeseries, data["X_test"][index], h)
            total_inference_time += time.time() - t0

            y_pred.append(pred)

    return np.array(y_pred, dtype=object), total_inference_time


def predict_shift_distances(dataset):
    # If validation data is present, merge this into the training data
    if len(dataset["X_val"]) > 0:
        data = merge_train_val(dataset)
    else:
        data = copy.deepcopy(dataset)

    total_inference_time = 0

    y_pred = []
    if dataset["is_multivariate"] and len(data["ts_raw"][0].shape) > 1:
        # Note: Multivariate data with multiple samples
        raise NotImplementedError("Not implemented for MTS with multiple samples")
    else:
        # Note: Unvariate data with multiple samples or
        #       Multivariate data with one sample (in this case each channel is a ts_raw)
        all_distances = []
        for index in range(len(data["ts_raw"])):
            h = len(data["y_train"][index][0])
            timeseries = data["ts_raw"][index]

            model = SHIFT()
            model.chain = False
            model.optimize_hyperparameters(
                timeseries, data["X_train"][index], data["y_train"][index]
            )

            t0 = time.time()
            pred = model.fit_predict(timeseries, data["X_test"][index], h)
            total_inference_time += time.time() - t0

            y_pred.append(pred)
            all_distances.append(np.sum(np.mean(model.distances, axis=(1)), axis=1))

    return np.array(y_pred, dtype=object), all_distances, total_inference_time


def predict_shift_ablation(
    dataset, k=False, fallback=None, chain=None, z_normalization=None, use_kd_tree=None
):
    # If validation data is present, merge this into the training data
    if len(dataset["X_val"]) > 0:
        data = merge_train_val(dataset)
    else:
        data = copy.deepcopy(dataset)

    model = SHIFT()
    if type(k) != bool:
        model.k = k
    if chain != None:
        model.chain = chain
    if z_normalization != None:
        model.z_normalization = z_normalization
    if fallback != None:
        model.fallback = fallback
    if use_kd_tree != None:
        model._use_kd_tree = use_kd_tree

    total_inference_time = 0
    y_pred = []
    if data["is_multivariate"] and len(data["ts_raw"][0].shape) > 1:
        # Note: Multivariate data with multiple samples
        raise NotImplementedError("Not implemented for MTS with multiple samples")
    else:
        # Note: Unvariate data with multiple samples or
        #       Multivariate data with one sample (in this case each sample is a ts_raw)
        for index in range(len(data["ts_raw"])):
            h = data["y_train"][index].shape[-1]
            timeseries = data["ts_raw"][index]

            model.optimize_hyperparameters(
                timeseries, data["X_train"][index], data["y_train"][index]
            )

            t0 = time.time()
            pred = model.fit_predict(timeseries, data["X_test"][index], h)
            total_inference_time += time.time() - t0

            y_pred.append(pred)

    return np.array(y_pred, dtype=object), total_inference_time


# =======================================================================
# =========================Deep learning methods=========================
# =======================================================================


def train_nbeats(X_train, y_train, X_val, y_val, random_state=42):
    keras.utils.set_random_seed(int(random_state))
    if len(X_train.shape) == 3:
        input_dim = X_train.shape[-1]
        output_dim = y_train.shape[-1]
    else:
        input_dim = 1
        output_dim = 1
    model = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=y_train[0].shape[0],
        backcast_length=X_train[0].shape[0],
        input_dim=input_dim,
        output_dim=output_dim,
    )
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss="mae")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    return model


# N-BEATS; https://arxiv.org/pdf/1905.10437
def predict_nbeats(dataset, random_states):
    all_pred = []
    all_inference_times = []
    for r in random_states:
        if dataset["is_multivariate"]:
            assert len(dataset["X_train"].shape) == 3
            # NOTE: This doesn't work, have to apply N-BEATS per channel, otherwise performance is really low
            # X_train, y_train, X_val, y_val, X_test = reshape_dataset(dataset, order="NLC")
            # print(X_train.shape)

            # nbeats = train_nbeats(X_train, y_train, X_val, y_val, random_state=r)

            # t0 = time.time()
            # y_pred = nbeats.predict(X_test, verbose=0)
            # all_inference_times.append(time.time() - t0)
            # y_pred = y_pred.transpose(2,0,1)
            # all_pred.append(y_pred)

            pred_by_channel = []
            total_inference_time = 0
            nr_channels = dataset["X_train"].shape[0]
            for c in range(nr_channels):
                nbeats = train_nbeats(
                    dataset["X_train"][c],
                    dataset["y_train"][c],
                    dataset["X_val"][c],
                    dataset["y_val"][c],
                    random_state=r,
                )

                t0 = time.time()
                y_pred = nbeats.predict(dataset["X_test"][c], verbose=0)
                total_inference_time += time.time() - t0
                y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
                pred_by_channel.append(y_pred)
            all_inference_times.append(total_inference_time)
            all_pred.append(pred_by_channel)
        else:
            nbeats = train_nbeats(
                dataset["X_train"],
                dataset["y_train"],
                dataset["X_val"],
                dataset["y_val"],
                random_state=r,
            )
            total_inference_time = 0

            t0 = time.time()
            y_pred = nbeats.predict(dataset["X_test"], verbose=0)
            total_inference_time += time.time() - t0
            all_inference_times.append(total_inference_time)

            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
            all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)


def train_nbeats_i(X_train, y_train, X_val, y_val, random_state=42):
    keras.utils.set_random_seed(int(random_state))
    model = NBeatsNet(
        stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
        forecast_length=y_train.shape[1],
        backcast_length=X_train.shape[1],
    )
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss="mae")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    return model


# GRU; https://arxiv.org/pdf/1412.3555)
def predict_gru(dataset, random_states):
    def train_gru(X_train, y_train, X_val, y_val, random_state=42):
        keras.utils.set_random_seed(int(random_state))
        if len(X_train.shape) < 3:
            inputs = keras.Input(shape=(X_train.shape[1], 1))
        else:
            inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[-1]))

        gru = keras.layers.GRU(100, activation="tanh", return_sequences=True)(inputs)
        if len(X_train.shape) < 3:
            gru2 = keras.layers.GRU(100, activation="tanh", return_sequences=False)(gru)
            dense = keras.layers.Dense(y_train.shape[1], activation="linear")(gru2)
            model = keras.Model(inputs=inputs, outputs=dense)
        else:
            gru2 = keras.layers.GRU(100, activation="tanh", return_sequences=False)(gru)
            dense = keras.layers.Dense(
                y_train.shape[1] * y_train.shape[-1], activation="linear"
            )(gru2)
            reshape = keras.layers.Reshape((y_train.shape[1], y_train.shape[-1]))(dense)
            model = keras.Model(inputs=inputs, outputs=reshape)

        optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
        model.compile(optimizer, loss="mae")

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0,
        )

        return model

    all_pred = []
    all_inference_times = []
    for r in random_states:
        X_train, y_train, X_val, y_val, X_test = reshape_dataset(dataset, order="NLC")

        gru = train_gru(X_train, y_train, X_val, y_val, random_state=r)

        t0 = time.time()
        y_pred = gru.predict(X_test, verbose=0)
        all_inference_times.append(time.time() - t0)
        if not dataset["is_multivariate"]:
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        else:
            y_pred = y_pred.transpose(2, 0, 1)
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)


# DLinear; https://arxiv.org/pdf/2205.13504
def predict_dlinear(dataset, random_states):
    class ModelConfig(object):
        pass

    config = ModelConfig()

    # config.use_gpu = True
    config.use_gpu = False
    config.devices = "0,1,2,3"
    config.gpu = 0
    # config.use_multi_gpu = True
    config.use_multi_gpu = False

    config.pred_len = dataset["y_train"].shape[-1]
    config.label_len = dataset["y_train"].shape[-1]
    config.seq_len = dataset["X_train"].shape[-1]
    if len(dataset["X_train"].shape) < 3:
        config.enc_in = 1
        config.dec_in = 1
    else:
        config.enc_in = dataset["X_train"].shape[0]
        config.dec_in = dataset["X_train"].shape[0]
    config.embed = "timeF"
    config.freq = "h"
    config.patience = 3
    config.learning_rate = 0.001
    config.train_epochs = 10
    config.features = "M"
    config.lradj = "type1"
    config.moving_avg = 25
    config.batch_size = 16

    all_pred = []
    all_inference_times = []
    for r in random_states:
        random.seed(r)
        torch.manual_seed(r)
        np.random.seed(r)

        # Create model with pytorch
        device = _acquire_device(
            config.use_gpu, config.gpu, config.use_multi_gpu, config.devices
        )

        model = DLinear(config).float()
        if config.use_gpu:
            device_ids = config.devices.split(",")
            device_ids = [int(id_) for id_ in device_ids]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        X_train, y_train, X_val, y_val, X_test = reshape_dataset(dataset)

        # Train the model with early stopping based on validation data
        model = train_model(model, config, X_train, y_train, X_val, y_val)

        total_inference_time = 0

        t0 = time.time()
        y_pred = predict(model, config, X_test)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)
        if len(dataset["X_train"].shape) < 3:
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        else:
            y_pred = y_pred.transpose(2, 0, 1)
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)


# FEDformer; https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf
def predict_fedformer(dataset, d_ff, d_model, random_states):
    class ModelConfig(object):
        pass

    config = ModelConfig()

    # config.use_gpu = True
    config.use_gpu = False
    config.devices = "0,1,2,3"
    config.gpu = 0
    # config.use_multi_gpu = True
    config.use_multi_gpu = False

    config.pred_len = dataset["y_train"].shape[-1]
    config.label_len = dataset["y_train"].shape[-1]
    config.seq_len = dataset["X_train"].shape[-1]
    config.e_layers = 2
    if len(dataset["X_train"].shape) < 3:
        config.enc_in = 1
        config.dec_in = 1
    else:
        config.enc_in = dataset["X_train"].shape[0]
        config.dec_in = dataset["X_train"].shape[0]
    config.embed = "timeF"
    config.dropout = 0.1
    config.c_out = 1
    config.d_model = d_model
    config.d_ff = d_ff
    config.freq = "h"
    config.patience = 3
    config.learning_rate = 0.001
    config.train_epochs = 5
    config.features = "M"
    config.lradj = "type1"
    config.top_k = 5
    config.num_kernels = 6
    config.batch_size = 16

    config.moving_avg = 25
    config.n_heads = 8
    config.activation = "gelu"
    config.d_layers = 1

    all_pred = []
    all_inference_times = []
    for r in random_states:
        random.seed(r)
        torch.manual_seed(r)
        np.random.seed(r)

        # Create model with pytorch
        device = _acquire_device(
            config.use_gpu, config.gpu, config.use_multi_gpu, config.devices
        )

        model = FEDformer(config).float()
        if config.use_gpu:
            device_ids = config.devices.split(",")
            device_ids = [int(id_) for id_ in device_ids]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        X_train, y_train, X_val, y_val, X_test = reshape_dataset(dataset)

        # Train the model with early stopping based on validation data
        model = train_model(model, config, X_train, y_train, X_val, y_val)

        total_inference_time = 0

        t0 = time.time()
        y_pred = predict(model, config, X_test)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        if len(dataset["X_train"].shape) < 3:
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        else:
            y_pred = y_pred.transpose(2, 0, 1)
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)


# Non-stationary transformer; https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf
def predict_nonstationary_transformer(dataset, d_ff, d_model, random_states):
    class ModelConfig(object):
        pass

    config = ModelConfig()

    # config.use_gpu = True
    config.use_gpu = False
    config.devices = "0,1,2,3"
    config.gpu = 0
    # config.use_multi_gpu = True
    config.use_multi_gpu = False

    config.pred_len = dataset["y_train"].shape[-1]
    config.label_len = dataset["y_train"].shape[-1]
    config.seq_len = dataset["X_train"].shape[-1]
    config.e_layers = 2
    if len(dataset["X_train"].shape) < 3:
        config.enc_in = 1
        config.dec_in = 1
    else:
        config.enc_in = dataset["X_train"].shape[0]
        config.dec_in = dataset["X_train"].shape[0]
    config.embed = "timeF"
    config.dropout = 0.1
    config.c_out = 1
    config.d_model = d_model
    config.d_ff = d_ff
    config.freq = "h"
    config.patience = 3
    config.learning_rate = 0.001
    config.train_epochs = 5
    config.features = "M"
    config.lradj = "type1"
    config.top_k = 5
    config.num_kernels = 6
    config.batch_size = 16

    config.n_heads = 8
    config.activation = "gelu"
    config.d_layers = 1
    config.output_attention = True
    config.factor = 1
    config.p_hidden_dims = [256, 256]
    config.p_hidden_layers = 2

    all_pred = []
    all_inference_times = []
    for r in random_states:
        random.seed(r)
        torch.manual_seed(r)
        np.random.seed(r)

        # Create model with pytorch
        device = _acquire_device(
            config.use_gpu, config.gpu, config.use_multi_gpu, config.devices
        )

        model = Nonstationary_Transformer(config).float()
        if config.use_gpu:
            device_ids = config.devices.split(",")
            device_ids = [int(id_) for id_ in device_ids]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        X_train, y_train, X_val, y_val, X_test = reshape_dataset(dataset)

        # Train the model with early stopping based on validation data
        model = train_model(model, config, X_train, y_train, X_val, y_val)

        total_inference_time = 0

        t0 = time.time()
        y_pred = predict(model, config, X_test)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        if len(dataset["X_train"].shape) < 3:
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        else:
            y_pred = y_pred.transpose(2, 0, 1)
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)


# TimesNet; https://openreview.net/pdf?id=ju_Uqw384Oq
def predict_timesnet(dataset, d_ff, d_model, random_states):
    class ModelConfig(object):
        pass

    config = ModelConfig()

    config.use_gpu = False
    config.devices = "0,1,2,3"
    config.gpu = 0
    config.use_multi_gpu = False

    config.pred_len = dataset["y_train"].shape[-1]
    config.label_len = dataset["y_train"].shape[-1]
    config.seq_len = dataset["X_train"].shape[-1]
    config.e_layers = 2
    if len(dataset["X_train"].shape) < 3:
        config.enc_in = 1
        config.dec_in = 1
    else:
        config.enc_in = dataset["X_train"].shape[0]
        config.dec_in = dataset["X_train"].shape[0]
    config.embed = "timeF"
    config.dropout = 0.1
    config.c_out = 1
    config.d_model = d_model
    config.d_ff = d_ff
    config.freq = "h"

    config.patience = 3
    config.learning_rate = 0.001
    config.train_epochs = 5
    config.features = "M"
    config.lradj = "type1"
    config.top_k = 5
    config.num_kernels = 6
    config.batch_size = 16

    all_pred = []
    all_inference_times = []
    for r in random_states:
        random.seed(r)
        torch.manual_seed(r)
        np.random.seed(r)

        # Create model with pytorch
        device = _acquire_device(
            config.use_gpu, config.gpu, config.use_multi_gpu, config.devices
        )

        model = TimesNet(config).float()
        if config.use_gpu:
            device_ids = config.devices.split(",")
            device_ids = [int(id_) for id_ in device_ids]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        X_train, y_train, X_val, y_val, X_test = reshape_dataset(dataset)

        # Train the model with early stopping based on validation data
        model = train_model(model, config, X_train, y_train, X_val, y_val)

        total_inference_time = 0

        t0 = time.time()
        y_pred = predict(model, config, X_test)
        total_inference_time += time.time() - t0
        all_inference_times.append(total_inference_time)

        if len(dataset["X_train"].shape) < 3:
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        else:
            y_pred = y_pred.transpose(2, 0, 1)
        all_pred.append(y_pred)

    return np.array(all_pred), np.mean(all_inference_times)
