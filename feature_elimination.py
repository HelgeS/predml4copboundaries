import config
import features
import numpy as np
import pandas as pd
import random

from sklearn.svm import LinearSVR
from sklearn.feature_selection import RFECV

records = []

for p in config.PROBLEMS[2:]:
    print(p.name)

    all_dzns = p.get_dzns()
    #dzns = random.sample(all_dzns, min(100, len(all_dzns)))
    #dzns = [name for name, _ in dzns]
    #feat = features.cached_feature_matrix(p, dzns=dzns, include_opt=True, include_mzn2feat=True)

    X, results = features.cached_feature_matrix(p, include_opt=True, include_mzn2feat=True, include_labels=False)
    #y = results['optimum'].values
    feat_labels = features.feature_names(p)

    print(X.shape)

    varmat = np.var(X, axis=0, dtype=np.float64)

    mzn_feat = {feat: std for feat, std in zip(feat_labels, varmat) if '/' not in feat}
    mzn_feat['problem'] = p.name
    records.append(mzn_feat)

    cust_feat = [{'problem': p.name, 'i_' + feat.split('/')[0]: std} for feat, std in zip(feat_labels, varmat) if '/' in feat]
    records.extend(cust_feat)

    for feat, std in zip(feat_labels, varmat):
        print(feat, '\t', std)

    #svm = LinearSVR()
    #rfecv = RFECV(estimator=svm, step=1, scoring='neg_mean_squared_error')
    #rfecv.fit(X, y)

    #print("Optimal number of features : %d" % rfecv.n_features_)
    #print("Support", rfecv.support_)
    #print("Ranking", rfecv.ranking_)

    print('-------')

df = pd.DataFrame.from_records(records).groupby('problem').mean().round(3).transpose()
print(df)
df.to_clipboard()
