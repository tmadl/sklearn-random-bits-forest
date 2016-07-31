from sklearn.base import BaseEstimator
import sklearn.metrics
import numpy as np
import pandas as pd
import subprocess

class RandomBitsForest(BaseEstimator):
    """
    This is a scikit-learn compatible wrapper for the Random Bits Forest
    classifier and regressor for big data, developed by Yi Wang 
    (http://www.nature.com/articles/srep30086)
    Parameters
    ----------
    random_bits : int, optional (default=8192)
        Number of random bits used. The default setting (8192 bits) balances well on
        various datasets and the parameter is usually not tuned.
    boost_chains : int, optional (default=256)
        Number of independent boosting chain. Larger value is equivalent to larger
        regularization. For small sample size dataset, you can set a big value (but should be
        less than -b/32).
    candidates : int, optional (default=256)
        Number of candidates for each random bit. We generate this many random neural
        network candidates, and pick the best one as one random bit. Larger value take more
        time but produces better result.
    first_layer_features : int, optional (default=3)
        The number of features involved in first layer neural network. This parameter
        should be tuned with small integers (2-9). Usually 2 or 3 are choice.
    second_layer_features : int, optional (default=3)
        The number of node in second layer neural network. This parameter is usually 2.
    number_of_trees : int, optional (default=128)
        Number of trees in the forest. Larger value is slightly better but take more time.
    sample_bootstrap_fold : float, optional (default=1)
        Sample bootstrap fold. Smaller value (<1) is corresponding to more
        regularization. Larger value takes more time
    feature_bootstrap_fold : float, optional (default=1)
        Feature bootstrap fold. Smaller value (<1) is corresponding to more
        regularization. Larger value takes more time
    verbose : bool, optional (default: False)
        verbose output
    """
    
    def __init__(self, random_bits=8192, boost_chains=256, candidates=256, first_layer_features=3, second_layer_features=3, number_of_trees=128, sample_bootstrap_fold=1, feature_bootstrap_fold=1, verbose=False):
        self.params = {
            "-b": random_bits,
            "-i": boost_chains,
            "-c": candidates,
            "-1": first_layer_features,
            "-2": second_layer_features,
            "-n": number_of_trees,
            "-s": sample_bootstrap_fold,
            "-f": feature_bootstrap_fold
        }
        self.verbose = verbose
        
        
    def fit(self, X, y, temp_file_name="tmp.csv"):
        """Fit the random bit forest to data
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data 
        y : array_like, shape = [n_samples]
            Labels
        temp_file_name : string, optional (default="tmp.csv")
            Temporary file for writing the data ()
            
        Returns
        -------
        self : returns an instance of self.
        """
        extension = ""
        if len(temp_file_name.split(".")[-1]) <= 5:
            extension = "."+temp_file_name.split(".")[-1]
            temp_file_name = ".".join(temp_file_name.split(".")[:-1])
        self.temp_file_name = temp_file_name
        self.temp_extension = extension
        pd.DataFrame(X).to_csv(self.temp_file_name+"_trainX"+self.temp_extension, sep='\t', header=False, index=False)
        pd.DataFrame(y.reshape(-1,1)).to_csv(temp_file_name+"_trainy"+extension, sep='\t', header=False, index=False)
        return self
                
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        
        pd.DataFrame(X).to_csv(self.temp_file_name+"_testX"+self.temp_extension, sep='\t', header=False, index=False)

        args = []
        for p in self.params.keys():
            args.append(p)
            args.append(str(self.params[p]))

        p = subprocess.Popen(
            ["./rbf"]+args+[self.temp_file_name+"_trainX"+self.temp_extension, self.temp_file_name+"_trainy"+self.temp_extension, self.temp_file_name+"_testX"+self.temp_extension, self.temp_file_name+"_testy"+self.temp_extension],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        out, err = p.communicate()
        
        if "invalid" in err or "usage" in err:
            raise Exception(err)
        elif self.verbose:
            print out
            print err
        
        P = pd.read_csv(self.temp_file_name+"_testy"+self.temp_extension, header=None).as_matrix().flatten()
        
        return np.vstack((1-P, P)).T
        
    def predict(self, X):
        """Perform classification on samples in X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        return 1*(self.predict_proba(X)[:,1]>=0.5)
    
    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)
