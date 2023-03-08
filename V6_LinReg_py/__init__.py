from sklearn.linear_model import SGDRegressor
import numpy as np







def master():
    pass

def RPC_fit_round(data, parameters, cols, lr):

    

    #TODO: implement train/test split(seed wise)
    X_train = data.loc[data['train/test']=='train'][cols].values.reshape(-1,1)
    y_train = data.loc[data['train/test']=='train']['brainAge_sim'].values
    X_test = data.loc[data['train/test']=='test'][cols].values.reshape(-1,1)
    y_test = data.loc[data['train/test']=='test']['brainAge_sim'].values
    

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=lr)
    
    model.coef_ = np.copy(parameters[0])
    model.intercept_ = np.copy(parameters[1])

    model.partial_fit(X_train, y_train)

    loss = np.mean((model.predict(X_test) - y_test) **2)

    return {
        "param": (model.coef_, model.intercept_),
        "loss": loss,
        "size": y_test.shape[0]
    }


