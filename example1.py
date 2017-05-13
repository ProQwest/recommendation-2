import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(data_home='.', min_rating=4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("   --Known Movies:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("   --Recommended Movies:")

        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])