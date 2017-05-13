import numpy as np

from lightfm.datasets import fetch_stackexchange
from lightfm.evaluation import auc_score
from lightfm import LightFM

data = fetch_stackexchange('crossvalidated',
                           test_set_fraction=0.1,
                           data_home='.',
                           indicator_features=False,
                           tag_features=True)

train = data['train']
test = data['test']

print('The dataset has %s users and %s items, '
      'with %s interactions in the test and %s interactions in the training set.'
      % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))



# Set the number of threads; you can increase this
# ify you have more physical cores available.
NUM_THREADS = 8
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

item_features = data['item_features']
tag_labels = data['item_feature_labels']

print('There are %s distinct tags, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))
print item_features.shape, type(item_features)
print item_features.toarray()

# # Define a new model instance
# model = LightFM(loss='warp',
#                 item_alpha=ITEM_ALPHA,
#                 no_components=NUM_COMPONENTS)
#
# # Fit the hybrid model. Note that this time, we pass
# # in the item features matrix.
# model = model.fit(train,
#                 item_features=item_features,
#                 epochs=NUM_EPOCHS,
#                 num_threads=NUM_THREADS)
#
# # Don't forget the pass in the item features again!
# train_auc = auc_score(model,
#                       train,
#                       item_features=item_features,
#                       num_threads=NUM_THREADS).mean()
# print('Hybrid training set AUC: %s' % train_auc)
#
# test_auc = auc_score(model,
#                     test,
#                     train_interactions=train,
#                     item_features=item_features,
#                     num_threads=NUM_THREADS).mean()
# print('Hybrid test set AUC: %s' % test_auc)