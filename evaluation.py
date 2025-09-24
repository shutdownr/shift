import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))

def smape(y_true, y_pred):
    return np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + np.finfo(float).eps))

def mase(X_test, y_true, y_pred):
    # src: https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/common/metrics.py
    def mase_sample(actual, forecast, insample, m=1):
        # num = np.mean(np.abs(actual - forecast))
        denum = np.mean(np.abs(insample[:-m] - insample[m:]))

        # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway (TODO)
        if denum == 0.0:
            denum = 1.0
        return np.mean(np.abs(actual - forecast)) / denum

    return np.mean(
        [mase_sample(y_true[i], y_pred[i], X_test[i]) for i in range(len(y_pred))]
    )

def owa(smape, smape_naive, mase, mase_naive):
    return ((smape / smape_naive) + (mase / mase_naive)) / 2

# Naive forecast
def predict_naive(dataset):
    return np.array([np.full(dataset["y_train"].shape[-1], X[-1]) for X in dataset["X_test"]])

def evaluate_errors(x_test, y_test, y_pred, metrics, dataset=None):
    # Return a mean of multiple predictions for non-deterministic models
    if len(y_pred.shape) > len(y_test.shape):
        return np.mean([evaluate_errors(x_test, y_test, y_p, metrics, dataset) for y_p in y_pred], axis=0)

    # Flatten inputs if necessary
    if type(y_pred[0][0]) == list or type(y_pred[0][0]) == np.ndarray:
        x_test_flat = np.array([i for l in x_test for i in l])
        y_test_flat = np.array([i for l in y_test for i in l])
        y_pred_flat = np.array([i for l in y_pred for i in l])
        return evaluate_errors(x_test_flat, y_test_flat, y_pred_flat, metrics, dataset)

    results = []
    for metric in metrics:
        if metric == "mae":
            error = mae(y_test, y_pred)
        elif metric == "mape":
            error = mape(y_test, y_pred)
        elif metric == "rmse":
            error = rmse(y_test, y_pred)
        elif metric == "smape":
            error = smape(y_test, y_pred)
        elif metric == "mase":
            error = mase(x_test, y_test, y_pred)
        elif metric == "owa":
            if dataset == None:
                print("[ERROR] To calculate OWA, the dataset needs to be passed")
            pred_naive = predict_naive(dataset)
            smape_naive = smape(y_test, pred_naive)
            mase_naive = mase(x_test, y_test, pred_naive)
            error = owa(smape(y_test, y_pred), smape_naive, mase(x_test, y_test, y_pred), mase_naive)
        results.append(error)
    return results