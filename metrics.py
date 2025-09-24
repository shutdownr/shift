import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

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

def owa(X_test, y_true, y_pred):
    horizon = y_true.shape[-1]
    y_pred_naive = np.repeat(X_test[:,-1],horizon).reshape(y_true.shape)

    smape_ = smape(y_true, y_pred)
    mase_ = mase(X_test, y_true, y_pred)
    smape_naive = smape(y_true, y_pred_naive)
    mase_naive = mase(X_test, y_true, y_pred_naive)

    return ((smape_ / smape_naive) + (mase_ / mase_naive)) / 2

# Not sure if corr should be implemented like this
#  or per channel and then averaged
def corr(y_true, y_pred):
    raise NotImplementedError("Not sure about the implementation")
    means_true = np.mean(y_true)
    means_pred = np.mean(y_pred)
    a = y_true - means_true
    b = y_pred - means_pred

    return np.sum(a*b) / np.sqrt(np.sum((a**2))*np.sum((b**2)))