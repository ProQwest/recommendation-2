from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score

EPOCHS = 10
ALPHA = 1e-3
NUM_THREADS = 4

# fetch data
movielens = fetch_movielens(data_home='.')

for key, value in movielens.items():
    print(key, type(value), value.shape)

train = movielens['train']
test = movielens['test']

# BPR model
model = LightFM(learning_rate=0.05, loss='bpr')
model = model.fit(train,
                  epochs=EPOCHS,
                  num_threads=NUM_THREADS)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()
train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print 'BPR model'
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

# WARP model
model = LightFM(learning_rate=0.05, loss='warp')
model = model.fit(train,
                  epochs=EPOCHS,
                  num_threads=NUM_THREADS)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()
train_recall = recall_at_k(model, train, k=10).mean()
test_recall = recall_at_k(model, test, k=10).mean()
train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print 'WARP model'
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

# use learning schedules
adagrad_model = LightFM(no_components=30,
                        loss='warp',
                        learning_schedule='adagrad',
                        user_alpha=ALPHA,
                        item_alpha=ALPHA)
adagrad_model = adagrad_model.fit(train,
                                  epochs=EPOCHS,
                                  num_threads=NUM_THREADS)

train_precision = precision_at_k(adagrad_model, train, k=10).mean()
test_precision = precision_at_k(adagrad_model, test, k=10).mean()
train_auc = auc_score(adagrad_model, train).mean()
test_auc = auc_score(adagrad_model, test).mean()

print 'adagrad'
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

adadelta_model = LightFM(no_components=30,
                        loss='warp',
                        learning_schedule='adadelta',
                        user_alpha=ALPHA,
                        item_alpha=ALPHA)
adadelta_model = adadelta_model.fit(train,
                                    epochs=EPOCHS,
                                    num_threads=NUM_THREADS)

train_precision = precision_at_k(adadelta_model, train, k=10).mean()
test_precision = precision_at_k(adadelta_model, test, k=10).mean()
train_auc = auc_score(adadelta_model, train).mean()
test_auc = auc_score(adadelta_model, test).mean()

print 'adadelta'
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

# use item features
item_features = movielens['item_features']
tag_labels = movielens['item_feature_labels']
model = LightFM(no_components=30,
                loss='warp',
                learning_schedule='adagrad',
                user_alpha=ALPHA,
                item_alpha=ALPHA)
model = model.fit(train,
                  epochs=EPOCHS,
                  item_features=item_features,
                  num_threads=NUM_THREADS)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print 'use item features (adagrad)'
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
