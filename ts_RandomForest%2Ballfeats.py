import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import cPickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2
from sklearn.cross_validation import KFold
# all_df_original = pd.read_json('train.json')
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

x_train_best = x_train_small#[low_pvalues_cols]
x_val_best = x_val_small#[low_pvalues_cols]

# model = RandomForestClassifier(n_estimators=10, max_features=0.3, max_depth=18, min_samples_split=20, random_state=0, n_jobs=-1)
# model = model.fit(x_train_best, y_train)

# imp = zip(x_train_best.columns, list(model.feature_importances_))
# imp.sort(key = lambda t: t[1], reverse=True)

# bad_feats = [i[0] for i in imp[-20:]]
# cPickle.dump(bad_feats, open('bad_feats.p', 'wb')) 
bad_feats = cPickle.load(open('bad_feats.p', 'rb'))
print bad_feats
# print bad_feats
x_train_best.drop(bad_feats, axis=1, inplace=True)
x_val_best.drop(bad_feats, axis=1, inplace=True)

# predicted_train = pd.DataFrame(model.predict_proba(x_train_best))
# predicted_train.columns = ['high', 'low', 'medium']
# predicted_train.head()
# log_loss_train = log_loss(y_train, predicted_train.as_matrix())
# log_loss_train
# predicted_val = pd.DataFrame(model.predict_proba(x_val_best))
# predicted_val.columns = ['high', 'low', 'medium']
# predicted_val.head()
# log_loss_val = log_loss(y_val, predicted_val.as_matrix())
# log_loss_val
# 0.68867588674963398, 0.68 ok (n_estimators=20, max_features=0.5, max_depth=5, min_samples_split=2, random_state=0)
# 0.65246470021282676, 0.60 ok (n_estimators=20, max_features=0.5, max_depth=5, min_samples_split=2, random_state=0)
# 0.65992553205901083, 0.47953531208509781 OF (n_estimators=100, max_features='sqrt', max_depth=20, min_samples_split=2, random_state=0) 
# 0.64955018062340741, 0.5490683949104378 OF (n_estimators=100, max_features='sqrt', max_depth=20, min_samples_split=10, random_state=0)

# new
# best_loss: 0.627188125099
# max_features: 0.3; max_depth: 20; min_samples_split: 20


# ## Cross Validation

# In[30]:

def get_cv_loss(mf, md, mss):
    kf = KFold(x_train_best.shape[0], n_folds=5, random_state=2017)
    loss_tr = []
    loss_ts = []
    for train_index, test_index in kf:
        x_tr = x_train_best.reset_index().loc[train_index].set_index(['index'])
        y_tr =  y_train.reset_index().loc[train_index].set_index(['index'])
        x_ts = x_train_best.reset_index().loc[test_index].set_index(['index'])
        y_ts = y_train.reset_index().loc[test_index].set_index(['index'])

        model = RandomForestClassifier(n_estimators=1000, max_features=mf, max_depth=md, min_samples_split=mss, random_state=23, n_jobs=-1)
        model = model.fit(x_tr, y_tr)

        predicted_ts = pd.DataFrame(model.predict_proba(x_ts))
        predicted_ts.columns = ['high', 'low', 'medium']
        log_loss_ts = log_loss(y_ts, predicted_ts.as_matrix())
        loss_ts.append(log_loss_ts)
    
    print 'CV loss:', np.mean(loss_ts)
    predicted_val = pd.DataFrame(model.predict_proba(x_val_best))
    predicted_val.columns = ['high', 'low', 'medium']
    log_loss_val = log_loss(y_val, predicted_val.as_matrix())
    print 'Val loss:', log_loss_val
    return np.mean(loss_ts)


# In[ ]:

max_features = ['sqrt',0.2, 0.3, 0.4, 0.5]
max_depth = [10, 12, 14, 16, 18, 20]
min_samples_split = [2, 5, 10, 20]
best_loss = 100
best_params = ''
count = 1
for mf in max_features:
    for md in max_depth:
        for mss in min_samples_split:
            print count
            count += 1
            params = 'max_features: ' + str(mf) + '; max_depth: ' + str(md) + '; min_samples_split: ' + str(mss)
            print params
#             model = RandomForestClassifier(n_estimators=500, max_features=mf, max_depth=md, min_samples_split=mss, random_state=0)
#             model = model.fit(x_train_best, y_train)
#             predicted_train = pd.DataFrame(model.predict_proba(x_train_best))
#             predicted_train.columns = ['high', 'low', 'medium']
#             log_loss_train = log_loss(y_train, predicted_train.as_matrix())
#             print 'Train loss:', log_loss_train
#             predicted_val = pd.DataFrame(model.predict_proba(x_val_best))
#             predicted_val.columns = ['high', 'low', 'medium']
#             log_loss_val = log_loss(y_val, predicted_val.as_matrix())
#             print 'Val loss:', log_loss_val
#             if log_loss_val < best_loss:
#                 best_loss = log_loss_val
#                 best_params = params
            cv_loss = get_cv_loss(mf, md, mss)
            if cv_loss < best_loss:
                best_loss = cv_loss
                best_params = params   
                
print 'best_loss:', best_loss
print 'best_params:', best_params


# test_df = pd.read_csv('test_feats_max.csv')


# # In[79]:

# test_df.head(2)


# # In[80]:

# x_test = test_df[x_train_best.columns]
# x_test.shape


# # In[81]:

# pred_x = pd.DataFrame(model.predict_proba(x_test))
# pred_x.columns = ['high', 'low', 'medium']
# pred_x.head()


# # In[82]:

# subm = pd.merge(test_df[['listing_id']].reset_index(), pred_x.reset_index(), left_index=True, right_index=True)


# # In[83]:

# subm = subm[['listing_id', 'high', 'medium', 'low']]


# # In[84]:

# subm.shape
# # (74659, 4)


# # In[85]:

# subm.to_csv('Submission_RandomForest_auto120_tune+58feats_2.csv', index=None)


# # In[51]:

# model


# # In[ ]:



