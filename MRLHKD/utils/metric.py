from pycox.evaluation import EvalSurv
import pandas as pd
import numpy as np


def calculate_c_index(time_index, probability_array, y):
    t = y[:, 0]
    e = y[:, 1]
    surv = pd.DataFrame(probability_array, index=time_index)
    ev = EvalSurv(
        surv,
        t,
        e,
        censor_surv="km"
    )
    return ev.concordance_td('antolini')



def calculate_mae(predict_time, y):
    t = y[:, 0]
    e = y[:, 1]
    uncensored_index = e != 0
    t_unc = t[uncensored_index]
    p_unc = predict_time[uncensored_index]
    total_mae = np.absolute(p_unc.reshape(-1,)-t_unc)
    return np.mean(total_mae)

