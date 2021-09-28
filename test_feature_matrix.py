import config
import features


def test_generator():
    for p in config.PROBLEMS: # + config.MZNC1617_PROBLEMS:
        yield get_features, p


def get_features(problem):
    features.cached_feature_matrix(problem, include_opt=True, include_mzn2feat=True)
    features.cached_feature_matrix(problem, include_opt=True, include_mzn2feat=False)

    #tts = data_loader.get_train_test_split(problem)

    #for ds_id in range(5):
    #    X_train, y_train = features.cached_feature_matrix(problem, dzns=tts[ds_id]['train'], include_opt=True)
    #    X_test, y_test = features.cached_feature_matrix(problem, dzns=tts[ds_id]['test'], include_opt=True)
    #    X_train, y_train = features.cached_feature_matrix(problem, dzns=tts[ds_id]['train'], include_opt=True, include_mzn2feat=True)
    #    X_test, y_test = features.cached_feature_matrix(problem, dzns=tts[ds_id]['test'], include_opt=True, include_mzn2feat=True)


if __name__ == '__main__':
    for f, p in test_generator():
        f(p)
