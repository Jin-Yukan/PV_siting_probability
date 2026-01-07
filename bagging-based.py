import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import recall_score
from hyperopt import hp, fmin,STATUS_OK, tpe, Trials

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
import xgboost as xgb
import lightgbm as lgb

model_n='MLP'

seed = 42 
metric = 'f1'
kFoldSplits = 5
n_iter_hopt = 50

space = {'MLP':{'layer_size':hp.quniform('layer_size', 25, 100, 1),
                'alpha':hp.lognormal('alpha', mu=np.log(1e-4), sigma=1),
                'solver':hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
                'activation':hp.choice('activation', ['identity','logistic', 'tanh', 'relu']),
                #'learning_rate':hp.uniform('learning_rate', low=0.001, high=0.999),
                'learning_rate':hp.loguniform('learning_rate', low=np.log(1e-4), high=np.log(1.)),
                },
         'RF':{'n_estimators' : hp.choice('n_estimators', range(1,200)),
               'max_depth' : hp.choice('max_depth', range(1,20)),
               'min_samples_leaf' : hp.choice('min_samples_leaf',range(1,20))
               },
         'xgb':{'n_estimators': hp.choice('n_estimators', range(50,501,2)),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                'max_depth': hp.choice('max_depth', range(2,8,1)),
                'subsample': hp.uniform('subsample', 0.5, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
                },
         'lgb':{'n_estimators': hp.choice('lgb_n_estimators', range(50,501,2)),
                'learning_rate': hp.uniform('lgb_learning_rate', 0.01, 0.3),
                'max_depth': hp.choice('lgb_max_depth', range(2,8,1)),
                'num_leaves': hp.choice('lgb_num_leaves', range(20, 50, 1)),
                'min_child_weight': hp.choice('lgb_min_child_weight', [0.001,0.005,0.01,0.05,0.1]),
                'min_child_samples': hp.choice('lgb_min_child_samples', range(5,51,5)),
                'subsample': hp.uniform('lgb_subsample', 0.5, 1.0),
                'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.6, 1.0),
                'reg_alpha': hp.uniform('lgb_reg_alpha', 0, 1.0)
                },
        }
space=space[model_n]

def bestObj(space, n_iter_hopt, kFoldSplits, data_bootstrap, train_label, seed = 42, metric = 'accuracy'):
    def objective(space):
        best_score=1.0 
        metric = 'f1'

        if model_n =='xgb':
            model = xgb.XGBClassifier(**space)
        elif model_n =='RF':
            model = RFC(**space)
        elif model_n =='lgb':
            model = lgb.LGBMClassifier(**space)
        elif model_n=='MLP':
            model = Pipeline([('scaler', StandardScaler()),
                              ('mlp',MLPClassifier(hidden_layer_sizes=(int(space['layer_size']),), max_iter=100,
                                                    alpha=space['alpha'], solver=space['solver'], tol=1e-4, 
                                                    random_state=1, activation=space['activation'], 
                                                    learning_rate_init=space['learning_rate'])
                                )])

        score = 1-cross_val_score(model, data_bootstrap,train_label,
                                  cv=kFoldSplits,
                                  scoring=metric,
                                  verbose=False).mean() 
        
        if (score < best_score):
            best_score=score

        return {'loss': score, 'status': STATUS_OK, 'model':model}
    
    trials = Trials()
    
    best = fmin(objective, 
                space = space, 
                algo = tpe.suggest, 
                max_evals = n_iter_hopt, 
                trials = trials, 
                rstate = np.random.RandomState(seed)
               )
    
    return trials.best_trial['result']['model']

def puData(P_dir,U_dir):
    '''
    P_dir: the path and name of positive data
    U_dir: the path and name of unlabeled data
    '''
    data_P = pd.read_csv(f'{P_dir}')
    data_U = pd.read_csv(f'{U_dir}')

    factors = [
        "GHI", "temp", "windspeed", "hunmidity", "prec", "aod",
        "elevation", "slope", "aspect", "waterdist", "pop",
        "EC", "GDP", "CO2", "roaddis", "plantdist", "settledist"
        ]
    data_P = data_P[factors].dropna()
    data_U = data_U[factors].dropna()

    data_P=np.array(data_P)
    data_U=np.array(data_U)
    
    NP = data_P.shape[0]
    NU = data_U.shape[0]
                                                               
    P_label= np.zeros(shape=(NP,))
    P_label[:]=1.0

    PX_train,PX_test,PY_train,PY_test=train_test_split(data_P,P_label,test_size=0.3,random_state=0)
    scaler = StandardScaler()
    PX_train = scaler.fit_transform(PX_train)
    PX_test = scaler.transform(PX_test)
    data_U = scaler.transform(data_U)

    N_P_train= PX_train.shape[0]
    K = N_P_train
    train_label = np.zeros(shape=(N_P_train+K,))
    train_label[:N_P_train] = 1.0
   
 
    N_P_test=PX_test.shape[0]  
    TS=N_P_test
    test_label = np.zeros(shape=(N_P_test+TS,))
    test_label[:N_P_test] = 1.0
    
    n_oob = np.zeros(shape=(NU, 1))
    f_oob = np.zeros(shape=(NU, 2))
    t_m   = np.zeros(shape=(2*TS, 1))
    t_test = np.zeros(shape=(2*TS, 2))
    
    return  PX_train, PX_test, data_U, n_oob, f_oob, K, TS, NU, train_label, test_label, t_m, t_test

PX_train, PX_test,  data_U, n_oob, f_oob, K, TS, NU, train_label, test_label, t_m, t_test = puData(P_dir = r'positive.csv'
                                                                                            ,U_dir = r'unlabeled.csv')

test_label=pd.Series(test_label)
#test_label.to_csv(r'test_label.csv',header=True)
begin_time = time()
feature_improtance = []

test_pro=[]

T = 10
for i in range(T):
    bootstrap_sample = np.random.choice( np.arange(NU)
                                       , replace=True
                                       , size = K
                                       )
    
    data_bootstrap = np.concatenate((  PX_train
                                     , data_U[bootstrap_sample, :]
                                     )
                                     , axis=0
                                    )    
   
    
    model = bestObj( space, 
                     n_iter_hopt, 
                     kFoldSplits, 
                     data_bootstrap,
                     train_label,
                     seed = 42,
                     metric = 'accuracy')
    
    model.fit(data_bootstrap, train_label)
   
    idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
    
  
    bootstrap_test_sample = np.random.choice( idx_oob
                                            , replace=True
                                            , size = TS
                                            )
    data_test_bootstrap = np.concatenate((  PX_test
                                          ,data_U[bootstrap_test_sample, :]
                                          )
                                          ,axis=0
                                         )   
    
    t_idx=np.arange(TS)
   
    t_test[t_idx] += model.predict_proba(data_test_bootstrap[t_idx])              

    f_oob[idx_oob] += model.predict_proba(data_U[idx_oob])
    #feature_improtance.append(model.feature_importances_)
    n_oob[idx_oob] += 1
    t_m[t_idx] += 1

predict_proba = f_oob[:, 1]/n_oob.squeeze() 
#fea_imp = sum(feature_improtance)/T 
predict_test=t_test[:, 1]/t_m.squeeze()
predict_proba=pd.DataFrame(predict_proba)
predict_proba.to_csv(r'bagging-based_'+ model_n +'.csv')
end_time = time()
run_time = end_time-begin_time
print ('paraming run time:',run_time)
predict_test=np.int64(predict_test>0.5)
recall= recall_score(test_label, predict_test)
Pr=(sum(predict_test))/(predict_test.shape[0])
score=(recall*recall)/Pr

print("Recall:", recall)
print("Pr:", Pr)
print("Score:", score)