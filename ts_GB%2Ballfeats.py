# coding: utf-8
# # Gradient Boosting  + 'features'
# ## Train set

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingClassifier
import cPickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from time import time
all_df = pd.read_csv('train_feats_max_desc.csv')

x_train, x_val, y_train, y_val = train_test_split(all_df.drop(['interest_level'], 1),all_df[['interest_level']], test_size=0.2, random_state=42)

cat_feats = cPickle.load(open('cat_feats.p', 'rb'))

for col in ['interest_level']:
    y_train[col] = y_train[col].astype('category')
    y_val[col] = y_val[col].astype('category')
    
for col in cat_feats:
    x_train[col] = x_train[col].astype('category')
    x_val[col] = x_val[col].astype('category')

drop_list = [u'listing_id', 'index']
x_train_small = x_train.drop(drop_list,1)

x_val_small = x_val.drop(drop_list,1)

x_train_best = x_train_small
x_val_best = x_val_small
x_train_best.shape


# model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=4, max_features=0.3, random_state=0)
# # n_estimator: 500; learning_rate: 0.2; max_features: 0.4; max_depth: 3
# model = model.fit(x_train_best, y_train)

# predicted_train = pd.DataFrame(model.predict_proba(x_train_best))
# # predicted = model.predict_proba(x)
# predicted_train.columns = ['high', 'low', 'medium']
# predicted_train.head()
# # predicted

# log_loss_train = log_loss(y_train, predicted_train.as_matrix())
# log_loss_train

# predicted_val = pd.DataFrame(model.predict_proba(x_val_best))
# # predicted = model.predict_proba(x)
# predicted_val.columns = ['high', 'low', 'medium']
# predicted_val.head()
# predicted

# log_loss_val = log_loss(y_val, predicted_val.as_matrix())
# log_loss_val

# a = zip(x_train_best.columns, list(model.feature_importances_))
# a.sort(key = lambda t: t[1], reverse=True)

# bad_feats_gb = [i[0] for i in a[-20:]]
# cPickle.dump(bad_feats_gb, open('bad_feats_gb.p', 'wb')) 
bad_feats_gb = cPickle.load(open('bad_feats_gb.p', 'rb'))
x_train_best.drop(bad_feats_gb, axis=1, inplace=True)
x_val_best.drop(bad_feats_gb, axis=1, inplace=True)

log_scoring=make_scorer(log_loss, greater_is_better=False, needs_proba=True)
log_scoring

clf = GradientBoostingClassifier()
# specify parameters and distributions to sample from
param_dist = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8],
              "max_features": ['sqrt',0.2, 0.3, 0.4, 0.5],
              "n_estimators": [300, 400, 500, 600, 700, 800],
              "learning_rate": [round(i,2) for i in np.arange(0.05,0.2,0.01)],
             "min_samples_split": [None, 100, 300, 400, 500, 600, 800, 1000],
             "min_samples_leaf": [None, 20, 40, 50, 70 ,100]}

# run randomized search
n_iter_search = 2
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5, n_jobs=-1, scoring=log_scoring, verbose=1)

start = time()
search = random_search.fit(x_train_best, y_train['interest_level'])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

print search.grid_scores_

print search.best_score_
print search.best_estimator_
print search.best_params_