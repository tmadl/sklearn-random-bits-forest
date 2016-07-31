from uci_comparison import compare_estimators
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from randombitsforest import RandomBitsForest

estimators = {
              'RandomForest': RandomForestClassifier(n_estimators=200),
              'ExtraTrees': ExtraTreesClassifier(n_estimators=200),
              'RandomBitsForest': RandomBitsForest(number_of_trees=200)
            }

# optionally, pass a list of UCI dataset identifiers as the datasets parameter, e.g. datasets=['iris', 'diabetes']
# optionally, pass a dict of scoring functions as the metric parameter, e.g. metrics={'F1-score': f1_score}
compare_estimators(estimators)