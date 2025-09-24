import os
import time

import numpy as np
import torch
from torch import optim

from other_models.losses import mape_loss, mase_loss, smape_loss
from other_models.tools import EarlyStopping, adjust_learning_rate

def _acquire_device(use_gpu, gpu, use_multi_gpu, devices):
    if use_gpu:
        if use_multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")
    return device

def train_model(model, config, X_train, y_train, X_val, y_val):
    device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)
    # Split batches
    X_train = np.array_split(X_train, int(np.ceil(len(X_train) / config.batch_size)))
    y_train = np.array_split(y_train, int(np.ceil(len(y_train) / config.batch_size)))
    # Validation data is split in 1 batch, no need to make batches
    path = "other_models"

    train_steps = len(X_train)
    early_stopping = EarlyStopping(patience=config.patience, verbose=False)

    model_optim = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = smape_loss()

    for epoch in range(config.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        for batch_x, batch_y, in zip(X_train, y_train):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.astype(float)
            batch_y = batch_y.astype(float)
            batch_x = torch.tensor(batch_x, dtype=torch.float32).float().to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, None, dec_inp, None)

            f_dim = -1 if config.features == 'MS' else 0
            outputs = outputs[:, -config.pred_len:, f_dim:]
            batch_y = batch_y[:, -config.pred_len:, f_dim:].to(device)

            loss_value = criterion(batch_x, outputs, batch_y, torch.ones_like(batch_y))
            loss = loss_value
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()

        # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = validate(model, config, X_val, y_val, criterion)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            # epoch + 1, train_steps, train_loss, vali_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            # print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, config)

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model

def validate(model, config, X_val, y_val, criterion):
    device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)
    x = torch.tensor(X_val, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        # decoder input
        B, _, C = x.shape
        dec_inp = torch.zeros((B, config.pred_len, C)).float().to(device)
        dec_inp = torch.cat([x[:, -config.label_len:, :], dec_inp], dim=1).float()
        # encoder - decoder
        outputs = torch.zeros((B, config.pred_len, C)).float()  # .to(self.device)
        id_list = np.arange(0, B, 500)  # validation set size
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(x[id_list[i]:id_list[i + 1]], None,
                                                                    dec_inp[id_list[i]:id_list[i + 1]],
                                                                    None).detach().cpu()
        f_dim = -1 if config.features == 'MS' else 0
        outputs = outputs[:, -config.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y_val[:, :, 0]))
        batch_y_mark = torch.ones(true.shape)

        loss = criterion(x.detach().cpu()[:, :, 0], pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss

def predict(model, config, X_test):
    device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)
    x = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, config.pred_len, C)).float().to(device)
        dec_inp = torch.cat([x[:, -config.label_len:, :], dec_inp], dim=1).float()
        # encoder - decoder
        outputs = torch.zeros((B, config.pred_len, C)).float().to(device)
        id_list = np.arange(0, B, 1)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(x[id_list[i]:id_list[i + 1]], None,
                                                                    dec_inp[id_list[i]:id_list[i + 1]], None)

        f_dim = -1 if config.features == 'MS' else 0
        outputs = outputs[:, -config.pred_len:, f_dim:]
        outputs = outputs.detach().cpu().numpy()

        preds = outputs
        x = x.detach().cpu().numpy()

    return preds