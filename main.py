# Used to read the Parquet data
import pyarrow.parquet as parquet
# Used to train the baseline model
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

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
# Extract the most interesting features
X = data[[
        'auditweights_svd_prelaunch',
        'auditweights_ctr_gender',
        'auditweights_friendLikes'
        ]].fillna(0.0).values



def split_data(X, Y, p, k):
    N = X.shape[0]
    assert(p <= k)
    # assert(X.shape[0], y.shape[0])

    b1, b2 = N * p, N * (p + 1) 
    X_train = np.concat(X[0:b1], X[b2:N])
    y_train = np.concat(Y[0:b1], Y[b2:N])

    X_test = X[b1:b2]
    y_test = Y[b1:b2]

    return X_train, y_train, X_test, y_test

k = 5
for p in range(0, 5):
    X_train, y_train, X_test, y_test = split_data(X, y, p, k)
    model = DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

    y_score = model.predict(X_test, y_test)
    print(mean_squared_error(y_test, y_score))
