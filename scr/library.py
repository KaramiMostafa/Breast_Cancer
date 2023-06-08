import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, RFE, RFECV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier, cv
import xgboost as xgb
import seaborn as sns

class FeatureSelector:
    def __init__(self):
        pass

    @staticmethod
    def top_features_XGB_nestedcv(df, number_feat, target_var, param_grid, inner_cv=KFold(n_splits=5), outer_cv=KFold(n_splits=5)):
        """
        This function finds common top i features (by XGBoost classifier) and
        returns them as a list of strings that are the features' names.
    
        Parameters
        ----------
            df: DataFrame
                dataframe as an input
            number_feat: int
                number of features needed to be ranked
            target_var: str
                target variable
            param_grid: dict
                dictionary of hyperparameters for XGBoostClassifier
            inner_cv: cross-validation method
                cross-validation method for inner loop (default: 5-fold CV)
            outer_cv: cross-validation method
                cross-validation method for outer loop (default: 5-fold CV)
    
        Returns
        -------
            im_feat: list
                list of i top-ranked features
        """
        # defining the target value and separate it
        y = df[target_var]
        X = df.drop([target_var], axis=1)
    
        # instantiate the classifier
        xgb_clf = XGBClassifier()
    
        # set up the nested cross-validation
        nested_cv = GridSearchCV(xgb_clf, param_grid=param_grid, cv=inner_cv)
    
        # perform the nested cross-validation
        nested_scores = cross_val_score(nested_cv, X=X, y=y, cv=outer_cv, scoring='accuracy')
        print("Nested CV accuracy: %.3f (%.3f)" % (np.mean(nested_scores), np.std(nested_scores)))
    
        # fit the classifier to the full dataset
        nested_cv.fit(X, y)
        xgb_clf = nested_cv.best_estimator_
    
        # list of features name
        feat_names = list(X.columns)
    
        feats = {}  # a dict to hold feature_name: feature_importance
        for feature, importance in zip(feat_names, xgb_clf.feature_importances_):
            feats[feature] = importance  # add the name/value pair
        # appending the dictionary of features with their scores by each k subset
        feats.update({x: y for x, y in feats.items() if y != 0})
    
        # sort the features based on their importance
        im_feat = sorted(feats.items(), key=lambda feats: feats[1], reverse=True)[:number_feat]
        im_feat = [item for sublist in im_feat for item in sublist]
        im_feat = [elm for elm in im_feat if isinstance(elm, str)]
    
        # the list of most i-th top ranked features
        return im_feat
    
    @staticmethod
    def top_features_XGB(df, number_feat, target_var):
        """
        This function finds common top i features (by XGBoost classifier) and
        returns them as a list of strings that are the features' names.
    
        Parameters
        ----------
            df: DataFrame
                dataframe as an input
            number_feat: int
                number of features needed to be ranked
            target_var: str
                target variable
    
        Returns
        -------
            im_feat: list
                list of i top-ranked features
        """
        # defining the target value and separate it
        y = df[target_var]
        X = df.drop([target_var], axis=1)
    
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
            # declare parameters
            params = {
                "objective": "binary:logistic",  # "objective": "binary:logistic"
                "max_depth": 4,
                "alpha": 10,
                "learning_rate": 1.0,
                "n_estimators": 100,
            }
    
            # instantiate the classifier
            xgb_clf = XGBClassifier(**params)
    
            # fit the classifier to the training data
            xgb_clf.fit(X_train, y_train)
    
            # list of features name
            feat_names = list(X_train.columns)
    
            feats = {}  # a dict to hold feature_name: feature_importance
            for feature, importance in zip(feat_names, xgb_clf.feature_importances_):
                feats[feature] = importance  # add the name/value pair
            # appending the dictionary of features with their scores by each k subset
            feats.update({x: y for x, y in feats.items() if y != 0})
    
        # sort the features based on their importance
        im_feat = sorted(feats.items(), key=lambda feats: feats[1], reverse=True)[:number_feat]
        # im_feat.sort(key = lambda x: x[1], reverse=True)
        im_feat = [item for sublist in im_feat for item in sublist]
        im_feat = [elm for elm in im_feat if isinstance(elm, str)]
    
        # the list of most i-th top ranked features
        return im_feat
    
    @staticmethod
    def mean_conf(confusion_matrix):
        ''' <<mean_conf>> simply gets the mean of each element in confiusion matrixs 
        which are the outcome in each k-subset cross validaion.
    
        return: mean of all confision matrics
        '''
        # empty lists to fill up every elements of the different confusion matrics
        e1, e2, e3, e4 = [], [], [], []
        for i in range(0, len(confusion_matrix)):
            e1.append(confusion_matrix[i][0][0])
            e2.append(confusion_matrix[i][0][1])
            e3.append(confusion_matrix[i][1][0])
            e4.append(confusion_matrix[i][1][1])
        # getting mean of each element
        mean_matrix = [[round(np.mean(e1), 2), round(np.mean(e2), 2)],
                       [round(np.mean(e3), 2), round(np.mean(e4), 2)]]
        return mean_matrix
    
    @staticmethod
    def top_features_XGB_nestedcv(df, n_features, target_col, param_grid, inner_cv, outer_cv):
        """
        Given a dataframe df, returns the top n_features features ranked by importance
        using XGBoost model optimized by nested cross-validation with grid search.
        """
        # Split data into features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    
        # Create XGBoost model
        xgb = XGBClassifier()
    
        # Use nested cross-validation with grid search to optimize model and select top features
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
        nested_cv_results = cross_validate(grid_search, X=X, y=y, cv=outer_cv, scoring='roc_auc', n_jobs=-1)
        mean_nested_cv_scores = nested_cv_results['test_score']
        mean_nested_cv_score = np.mean(mean_nested_cv_scores)
        std_nested_cv_score = np.std(mean_nested_cv_scores)
    
        grid_search.fit(X, y)
        top_n_features = list(grid_search.best_estimator_.feature_importances_.argsort()[::-1][:n_features])
    
        return df.columns[top_n_features].tolist()
    
    @staticmethod
    def top_features_XGB(df, n_features, target_col):
        """
        Given a dataframe df, returns the top n_features features ranked by importance
        using XGBoost model.
        """
        # Split data into features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    
        # Create XGBoost model
        xgb = XGBClassifier()
    
        # Fit the XGBoost model
        xgb.fit(X, y)
    
        # Get feature importances
        feature_importances = xgb.feature_importances_
    
        # Sort features based on importance
        top_n_features = np.argsort(feature_importances)[::-1][:n_features]
    
        return df.columns[top_n_features].tolist()
