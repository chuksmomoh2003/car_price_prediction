from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,cross_validate
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer, mean_absolute_error,mean_squared_error,r2_score

X = df.drop(columns=['price'])
y = df['price']
    # create the train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)

from lightgbm import LGBMRegressor

# Add the provided hyperparameters
lgb_model = LGBMRegressor(
    objective='regression',
    metric='mse',
    n_jobs=-1,
    random_state=101,
    n_estimators=1932,
    num_leaves=18,
    min_child_samples=25,
    learning_rate=0.009270894888704539,
    max_bin=2**10,  # Since `log_max_bin` is 10, max_bin is 2^10 = 1024
    colsample_bytree=0.5274395821304206,
    reg_alpha=0.008493713626609325,
    reg_lambda=0.005910984041619941
)

lgb_model.fit(X_train,y_train)

pred = lgb_model.predict(X_test)

print('MAE:', mean_absolute_error(y_test, pred))
print('MSE:', mean_squared_error(y_test, pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test, pred)))
print('R-squared: ',r2_score(y_test,pred))


import joblib

# Save the model as a pickle in a file
filename = 'car_price_predictor.pkl'
joblib.dump(lgb_model, filename)
