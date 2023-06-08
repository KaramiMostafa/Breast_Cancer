import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
from xgboost import XGBClassifier



class LogisticRegressionGridSearch:
    def __init__(self, df_file):
        self.df = pd.read_csv(df_file)
        self.param_grid = {
            'objective': ['binary:logistic', 'binary:hinge'],
            'max_depth': [3, 4, 5],
            'alpha': [1, 10, 100],
            'learning_rate': [0.01, 0.1, 1.0],
            'n_estimators': [50, 100, 200]
        }
        self.test_metrics = pd.DataFrame(columns=['features_number', 'mean_accuracy_LR', 'std_accuracy_LR',
                                                  'confusion_LR', 'mean_log_loss', 'mean_auc_roc'])
        self.train_metrics = pd.DataFrame(columns=['features_number', 'mean_accuracy_LR', 'std_accuracy_LR',
                                                   'mean_training_time', 'confusion_LR', 'mean_log_loss',
                                                   'mean_auc_roc'])

    def run_grid_search(self):
        # Define the target value and separate it
        y = self.df['diagnosis']
        X = self.df.drop(['diagnosis'], axis=1)

        # Split the whole dataset into train (80%) and test (20%)
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=0)

        # Apply k-fold cross-validation (K=10) --> outer loop
        cv_outer = KFold(n_splits=10, shuffle=True)
        scaler = MinMaxScaler()
        X_ts = scaler.fit_transform(X_ts)

        # Convert the test set to a dataframe in order to use it in the while loop
        X_ts = pd.DataFrame(X_ts, columns=X_tr.columns)

        # Iteration over the number of features i
        i = 30
        while i != 0:
            list_im_feat = self.top_features_XGB_nestedcv(self.df, i, 'diagnosis', self.param_grid,
                                                          inner_cv=KFold(n_splits=3), outer_cv=KFold(n_splits=5))

            # Split the data into features and target
            X = self.df[list_im_feat]
            y = self.df['diagnosis']

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Fit the logistic regression model on the training data
            clf = LogisticRegression(max_iter=1000, random_state=42)

            # Perform cross-validation on the training set
            cv_results = cross_validate(clf, X_train, y_train, cv=10, scoring=['accuracy', 'roc_auc', 'neg_log_loss'],
                                        return_train_score=True)
            train_time = cv_results['fit_time']
            acc_tr = cv_results['train_accuracy']
            AUC_ROC_tr = cv_results['train_roc_auc']
            log_loss_list_tr = -cv_results['train_neg_log_loss']

            # Fit model on the full training set
            clf.fit(X_train, y_train)
            y_train_pred = clf.predict(X_train)
            conf_tr = confusion_matrix(y_train, y_train_pred, normalize='true')

            # Calculate performance metrics on the test set
            y_test_pred = clf.predict(X_test)
            y_test_proba = clf.predict_proba(X_test)[:, 1]
            acc_ts = accuracy_score(y_test, y_test_pred)
            AUC_ROC_ts = roc_auc_score(y_test, y_test_proba)
            log_loss_list_ts = log_loss(y_test, y_test_proba)
            conf_ts = confusion_matrix(y_test, y_test_pred, normalize='true')

            # Check for overfitting
            flag = "Green" if np.mean(acc_tr) - acc_ts <= 0.005 else "Red"

            # Store the results in dictionaries
            # Test results
            test_data_to_store = {'features_number': f'{i}', 'mean_accuracy_LR': np.mean(acc_ts),
                                  'std_accuracy_LR': np.std(acc_ts), 'confusion_LR': conf_ts.tolist(),
                                  'mean_auc_roc': np.mean(AUC_ROC_ts), 'mean_log_loss': np.mean(log_loss_list_ts),
                                  'Flag': flag}

            # Train results
            train_data_to_store = {'features_number': f'{i}', 'mean_accuracy_LR': np.mean(acc_tr),
                                   'std_accuracy_LR': np.std(acc_tr), 'mean_training_time': np.mean(train_time),
                                   'confusion_LR': conf_tr.tolist(), 'mean_auc_roc': np.mean(AUC_ROC_tr),
                                   'mean_log_loss': np.mean(log_loss_list_tr)}

            self.test_metrics = self.test_metrics.append(test_data_to_store, ignore_index=True)
            self.train_metrics = self.train_metrics.append(train_data_to_store, ignore_index=True)

            # Reduce the number of features for the next iteration
            i -= 1

        self.test_metrics.to_csv('grid_LR_ts.csv')
        self.train_metrics.to_csv('grid_LR_tr.csv')

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


class LogisticRegressionGridSearch:
    def __init__(self, df_file):
        self.df = pd.read_csv(df_file)
        self.param_grid = {
            'objective': ['binary:logistic', 'binary:hinge'],
            'max_depth': [3, 4, 5],
            'alpha': [1, 10, 100],
            'learning_rate': [0.01, 0.1, 1.0],
            'n_estimators': [50, 100, 200]
        }
        self.test_metrics = pd.DataFrame(columns=['features_number', 'mean_accuracy_LR', 'std_accuracy_LR',
                                                  'confusion_LR', 'mean_log_loss', 'mean_auc_roc'])
        self.train_metrics = pd.DataFrame(columns=['features_number', 'mean_accuracy_LR', 'std_accuracy_LR',
                                                   'mean_training_time', 'confusion_LR', 'mean_log_loss',
                                                   'mean_auc_roc'])

    def run_grid_search(self):
        # Define the target value and separate it
        y = self.df['diagnosis']
        X = self.df.drop(['diagnosis'], axis=1)

        # Split the whole dataset into train (80%) and test (20%)
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=0)

        # Apply k-fold cross-validation (K=10) --> outer loop
        cv_outer = KFold(n_splits=10, shuffle=True)
        scaler = MinMaxScaler()
        X_ts = scaler.fit_transform(X_ts)

        # Convert the test set to a dataframe in order to use it in the while loop
        X_ts = pd.DataFrame(X_ts, columns=X_tr.columns)

        # Iteration over the number of features i
        i = 30
        while i != 0:
            list_im_feat = self.top_features_XGB_nestedcv(self.df, i, 'diagnosis', self.param_grid,
                                                          inner_cv=KFold(n_splits=3), outer_cv=KFold(n_splits=5))

            # Split the data into features and target
            X = self.df[list_im_feat]
            y = self.df['diagnosis']

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Fit the logistic regression model on the training data
            clf = LogisticRegression(max_iter=1000, random_state=42)

            # Perform cross-validation on the training set
            cv_results = cross_validate(clf, X_train, y_train, cv=10, scoring=['accuracy', 'roc_auc', 'neg_log_loss'],
                                        return_train_score=True)
            train_time = cv_results['fit_time']
            acc_tr = cv_results['train_accuracy']
            AUC_ROC_tr = cv_results['train_roc_auc']
            log_loss_list_tr = -cv_results['train_neg_log_loss']

            # Fit model on the full training set
            clf.fit(X_train, y_train)
            y_train_pred = clf.predict(X_train)
            conf_tr = confusion_matrix(y_train, y_train_pred, normalize='true')

            # Calculate performance metrics on the test set
            y_test_pred = clf.predict(X_test)
            y_test_proba = clf.predict_proba(X_test)[:, 1]
            acc_ts = accuracy_score(y_test, y_test_pred)
            AUC_ROC_ts = roc_auc_score(y_test, y_test_proba)
            log_loss_list_ts = log_loss(y_test, y_test_proba)
            conf_ts = confusion_matrix(y_test, y_test_pred, normalize='true')

            # Check for overfitting
            flag = "Green" if np.mean(acc_tr) - acc_ts <= 0.005 else "Red"

            # Store the results in dictionaries
            # Test results
            test_data_to_store = {'features_number': f'{i}', 'mean_accuracy_LR': np.mean(acc_ts),
                                  'std_accuracy_LR': np.std(acc_ts), 'confusion_LR': conf_ts.tolist(),
                                  'mean_auc_roc': np.mean(AUC_ROC_ts), 'mean_log_loss': np.mean(log_loss_list_ts),
                                  'Flag': flag}

            # Train results
            train_data_to_store = {'features_number': f'{i}', 'mean_accuracy_LR': np.mean(acc_tr),
                                   'std_accuracy_LR': np.std(acc_tr), 'mean_training_time': np.mean(train_time),
                                   'confusion_LR': conf_tr.tolist(), 'mean_auc_roc': np.mean(AUC_ROC_tr),
                                   'mean_log_loss': np.mean(log_loss_list_tr)}

            self.test_metrics = self.test_metrics.append(test_data_to_store, ignore_index=True)
            self.train_metrics = self.train_metrics.append(train_data_to_store, ignore_index=True)

            # Reduce the number of features for the next iteration
            i -= 1

        self.test_metrics.to_csv('grid_LR_ts.csv')
        self.train_metrics.to_csv('grid_LR_tr.csv')

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


