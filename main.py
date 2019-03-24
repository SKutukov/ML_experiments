# Used to read the Parquet data
import pyarrow.parquet as parquet
# Used to train the baseline model
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

from sklearn.utils import shuffle

import sys

# redirect output 
sys.stdout = open("out.txt", "w")

# Where the downloaded data are
input_path = './'
# Where to store results
output_path = './'

# Read a single day to train model on as Pandas dataframe
data = parquet.read_table(input_path + 'date=2018-02-01').to_pandas()

def feedback_to_float(x):
    res = []
    feeddict = {
        "Commented": 0,
        "ReShared": 0,
        "Liked": 1,
        "Clicked": 0,
        "Ignored": 0,
        "Unliked": 0,
        "Complaint": 0,
        "Disliked": 0,
        "Viewed": 0
    }
    for feed in x:
        res.append(feeddict[feed])

    return np.array(res).mean().astype(int)



# Construct the label (liked objects)
data['liked'] = data['feedback'].apply(feedback_to_float)

def transform_data(data):
    users_data = data.groupby('instanceId_userId')
    resdata = None
    for user_data in users_data:
        user_data_liked = user_data[user_data['liked'] == 1]
        user_data_disliked = user_data[user_data['liked'] == 0]

        user_data_liked = shuffle(user_data_liked)
        user_data_disliked = shuffle(user_data_disliked)

        if user_data_liked.shape[0] == 0 or user_data_disliked.shape[0] == 0:
            continue

        liked_mask  = np.random.randint(2, size=user_data_liked.shape[0]).astype(bool)
        disliked_mask = np.random.randint(2, size=user_data_disliked.shape[0]).astype(bool)

        liked_mask[0] = True
        disliked_mask[0] = True

        user_data_liked_masked = user_data_liked[liked_mask].values
        user_data_disliked_masked = user_data_disliked[disliked_mask].values

        data_disliked = user_data_disliked_masked[[
            'instanceId_userId',
            'auditweights_svd_prelaunch',
            'auditweights_ctr_gender',
            'auditweights_friendLikes'
            ]].fillna(0.0).values

        r = np.array((None, data.shape[1]))
        for data_liked_masked in user_data_liked_masked[[
            'instanceId_userId',
            'auditweights_svd_prelaunch',
            'auditweights_ctr_gender',
            'auditweights_friendLikes'
            ]].fillna(0.0).values:
        
            a = np.repeat(data_liked_masked, data_disliked.shape[0])
            np.contaginate(r, np.concatenate(a, data_disliked, axis=1), axis=0)

        return r , np.repeat(1, r.shape[0])

print(transform_data(data))
exec(0)         

# Extract the most interesting features
X = data[[
        'instanceId_userId',
        'auditweights_svd_prelaunch',
        'auditweights_ctr_gender',
        'auditweights_friendLikes'
        ]].fillna(0.0).values

y = data['liked'].values


def split_data(X, Y, p, k):
    N = X.shape[0]
    assert(p <= k)
    # assert(X.shape[0], y.shape[0])

    b1, b2 = N * p, N * (p + 1) 
    X_train = np.concatenate(X[0:b1], X[b2:N])
    y_train = np.concatenate(Y[0:b1], Y[b2:N])

    X_test = X[b1:b2]
    y_test = Y[b1:b2]

    return X_train, y_train, X_test, y_test

k = 5
for p in range(0, 5):
    X_train, y_train, X_test, y_test = split_data(X, y, p, k)
    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    print(mean_squared_error(y_test, y_score))
