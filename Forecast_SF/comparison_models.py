import requests
import pandas as pd 
import numpy as np 
import warnings
import pickle
warnings.filterwarnings("ignore")
np.random.seed(1)
from shape import Shape,finder
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from cycler import cycler
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import geopandas as gpd
from dtaidistance import ed
from scipy.stats import ttest_1samp
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

plot_params = {"text.usetex":True,"font.family":"serif","font.size":20,"xtick.labelsize":20,"ytick.labelsize":20,"axes.labelsize":20,"figure.titlesize":20,"figure.figsize":(8,5),"axes.prop_cycle":cycler(color=['black','rosybrown','gray','indianred','red','maroon','silver',])}
plt.rcParams.update(plot_params)

opti_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,2,4],
               'bootstrap': [True, False]}

################
### Get Data ###
################

# Input data, 1989-2022
df_list_input={f"predictors_fatalities002_0000_00/cm?page={i}":i for i in range(1,78)}

df_input_t=pd.DataFrame()
for i in range(len(df_list_input)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df_input_t = pd.concat([df_input_t, df])
    df_input_t=df_input_t.reset_index(drop=True)
df_input_t=df_input_t.sort_values(by=["country_id","month_id"]) 
df_input_t=df_input_t.loc[df_input_t["year"]<=2022]
df_input_t=df_input_t.reset_index(drop=True)

# Create lagged DV
df_input_t["ucdp_ged_sb_best_sum_lag1"]=df_input_t.groupby("country_id")["ucdp_ged_sb_best_sum"].shift(1).fillna(0)

##################    
### Zero model ###
##################    

df_null = pd.DataFrame(0, index=range(12), columns=range(191))
df_null.to_csv('df_null_preds.csv') 

#################
### t-1 model ###
#################

# Test data 2021 

x_train = df_input_t[['country_id','month_id','year','ucdp_ged_sb_best_sum_lag1']].loc[df_input_t["year"]<=2020]
x_test = df_input_t[['country_id','month_id','year','ucdp_ged_sb_best_sum_lag1']].loc[df_input_t["year"]>=2021]

y_train = df_input_t[['country_id','month_id','year','ucdp_ged_sb_best_sum']].loc[df_input_t["year"]<=2020]
y_test = df_input_t[['country_id','month_id','year','ucdp_ged_sb_best_sum']].loc[df_input_t["year"]>=2021]

# Validation data
val_train_index = x_train.loc[x_train["year"]<=2018].index
val_test_index = x_train.loc[x_train["year"]>=2019].index

splits=np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
ps = PredefinedSplit(test_fold=splits)

x_train_s=x_train.drop(['country_id','month_id','year'], axis=1)
x_test_s=x_test.drop(['country_id','month_id','year'], axis=1)
y_train_s=y_train.drop(['country_id','month_id','year'], axis=1)
y_test_s=y_test.drop(['country_id','month_id','year'], axis=1)

model=RandomForestRegressor(random_state=0)
opti_model = GridSearchCV(estimator=model, param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
opti_model.fit(x_train_s, y_train_s.values.ravel())
x_test["preds_t1"] = opti_model.predict(x_test_s)

x_test_t=x_test.drop(['year','ucdp_ged_sb_best_sum_lag1'], axis=1)
x_train_t=x_test_t.pivot(index="month_id",columns='country_id',values='preds_t1')
x_train_t.to_csv('df_t1_preds1.csv') 

# Test data 2022 

x_train = df_input_t[['country_id','month_id','year','ucdp_ged_sb_best_sum_lag1']].loc[df_input_t["year"]<=2021]
x_test = df_input_t[['country_id','month_id','year','ucdp_ged_sb_best_sum_lag1']].loc[df_input_t["year"]>=2022]

y_train = df_input_t[['country_id','month_id','year','ucdp_ged_sb_best_sum']].loc[df_input_t["year"]<=2021]
y_test = df_input_t[['country_id','month_id','year','ucdp_ged_sb_best_sum']].loc[df_input_t["year"]>=2022]

# Validation data
val_train_index = x_train.loc[x_train["year"]<=2019].index
val_test_index = x_train.loc[x_train["year"]>=2020].index

splits=np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
ps = PredefinedSplit(test_fold=splits)

x_train_s=x_train.drop(['country_id','month_id','year'], axis=1)
x_test_s=x_test.drop(['country_id','month_id','year'], axis=1)
y_train_s=y_train.drop(['country_id','month_id','year'], axis=1)
y_test_s=y_test.drop(['country_id','month_id','year'], axis=1)

model=RandomForestRegressor(random_state=0)
opti_model = GridSearchCV(estimator=model, param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
opti_model.fit(x_train_s, y_train_s.values.ravel())
x_test["preds_t2"] = opti_model.predict(x_test_s)

x_test=x_test.pivot(index="month_id",columns='country_id',values='preds_t2')
x_test.to_csv('df_t1_preds2.csv') 








