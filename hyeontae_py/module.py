import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

import gc

warnings.filterwarnings('ignore')


# Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])

    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize=(12, 6))

    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label='TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label='TARGET == 1')

    # label the plot
    plt.xlabel(var_name);
    plt.ylabel('Density');
    plt.title('%s Distribution' % var_name)
    plt.legend();

    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)


def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.

    Parameters
    --------
        df (dataframe):
            the dataframe to calculate the statistics on
        group_var (string):
            the variable by which to group df
        df_name (string):
            the variable used to rename the columns

    Return
    --------
        agg (dataframe):
            a dataframe with the statistics aggregated for
            all numeric columns. Each instance of the grouping variable will have
            the statistics (mean, min, max, sum; currently supported) calculated.
            The columns are also renamed to keep track of features created.

    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns=col)

    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg


# Function to calculate correlations with the target for a dataframe
def target_corrs(df):
    # List of correlations
    corrs = []

    # Iterate through the columns
    for col in df.columns:
        print(col)
        # Skip the target column
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df[col])

            # Append the list as a tuple
            corrs.append((col, corr))

    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)

    return corrs


def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable

    Parameters
    --------
    df : dataframe
        The dataframe to calculate the value counts for.

    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row

    df_name : string
        Variable added to the front of column names to keep track of columns


    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.

    """
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    return categorical


# Function to calculate missing values by column# Funct
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def merge_df(df1_train, df1_test, df2, standard, df2_col, df2_name):
    agg_new = agg_numeric(df2.drop(columns=[df2_col]), group_var=standard, df_name=df2_name)
    print(agg_new.head())

    # Merge with the training data
    train = df1_train.merge(agg_new, on=standard, how='left')
    print(train.head())
    print(train.info())

    counts = count_categorical(df2, group_var=standard, df_name=df2_name)
    print(counts.head())

    train = train.merge(counts, left_on=standard, right_index=True, how='left')
    print(train.head())

    test = df1_test.merge(agg_new, on=standard, how='left')
    test = test.merge(counts, left_on=standard, right_index=True, how='left')

    train_labels = train['TARGET']

    # Align the dataframes, this will remove the 'TARGET' column
    train, test = train.align(test, join='inner', axis=1)

    train['TARGET'] = train_labels
    print('Training Data Shape: ', train.shape)
    print('Testing Data Shape: ', test.shape)

    return train, test


def handle_missing_values(train, test, percentage):
    # Handling missing values
    # Train dataset missing values
    missing_train = missing_values_table(train)
    print(missing_train.head(10))
    missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > percentage])
    print(len(missing_train_vars))

    # Test dataset missing values
    missing_test = missing_values_table(test)
    print(missing_test.head(10))
    missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > percentage])
    print(len(missing_test_vars))

    missing_columns = list(set(missing_test_vars + missing_train_vars))
    print("There are %d columns with more than %d%% missing in either the training or testing data." % (len(
        missing_columns), percentage))

    # Drop the missing columns
    train = train.drop(columns=missing_columns)
    test = test.drop(columns=missing_columns)

    return train, test


def save_csv(train, test, train_name, test_name):
    train.to_csv(train_name, index=False)
    test.to_csv(test_name, index=False)


def corr_selection(train, test, threshold):
    # Calculate all correlations in dataframe
    corrs = train.corr()

    corrs = corrs.sort_values('TARGET', ascending=False)

    # Ten most positive correlations
    print(pd.DataFrame(corrs['TARGET'].head(10)))

    # Ten most negative correlations
    print(pd.DataFrame(corrs['TARGET'].dropna().tail(10)))

    # Set the threshold
    threshold = threshold

    # Empty dictionary to hold correlated variables
    above_threshold_vars = {}

    # For each column, record the variables that are above the threshold
    for col in corrs:
        above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])

    # Track columns to remove and columns already examined
    cols_to_remove = []
    cols_seen = []
    cols_to_remove_pair = []

    # Iterate through columns and correlated columns
    for key, value in above_threshold_vars.items():
        # Keep track of columns already examined
        cols_seen.append(key)
        for x in value:
            if x == key:
                next
            else:
                # Only want to remove one in a pair
                if x not in cols_seen:
                    cols_to_remove.append(x)
                    cols_to_remove_pair.append(key)

    cols_to_remove = list(set(cols_to_remove))
    print('Number of columns to remove: ', len(cols_to_remove))

    train_corrs_removed = train.drop(columns=cols_to_remove)
    test_corrs_removed = test.drop(columns=cols_to_remove)

    print('Training Corrs Removed Shape: ', train_corrs_removed.shape)
    print('Testing Corrs Removed Shape: ', test_corrs_removed.shape)

    return train_corrs_removed, test_corrs_removed


def feature_engineering(train, test, encoder, scaler):
    # Function for encoding categorical features with Label Encoder
    # Parameter : df -> Missing value handling, feature selection completed dataset
    def labelEnc(df):
        label = LabelEncoder()
        # select only columns with values are in object type and make new dataframe
        categorical_df = df.select_dtypes(include='object')
        numerical_df = df.select_dtypes(exclude='object')
        # encode all categorical columns with Label Encoder
        for i in range(0, len(categorical_df.columns)):
            df[categorical_df.columns[i]] = label.fit_transform(categorical_df.iloc[:, [i]].values)

        return df

    def OneHotEnc(df):
        df = pd.get_dummies(df)
        return df

    # Function for showing heatmap of correlation between each attributes
    # Parameter : df -> Missing value handling, feature selection completed dataset
    def showHeatmap(df):
        df = labelEnc(df)
        heatmap_data = df
        colormap = plt.cm.PuBu
        plt.figure(figsize=(15, 15))
        plt.title("Correlation of Features", y=1.05, size=15)
        sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
                    annot=True, annot_kws={"size": 8})
        plt.show()

    def scaling(scaler, X_train, X_test):

        feature_names = X_train.columns.values.tolist()
        # Make scaler with the parameter inputted with below condition
        Scaler = scaler
        if scaler == 'MinMaxScaler':
            Scaler = MinMaxScaler()
        elif scaler == 'MaxAbsScaler':
            Scaler = MaxAbsScaler()
        elif scaler == 'StandardScaler':
            Scaler = StandardScaler()
        elif scaler == 'RobustScaler':
            Scaler = RobustScaler()

        # Scale the datasets with each scaler chosen with condition above
        X_train_scale = Scaler.fit_transform(X_train)
        X_test_scale = Scaler.transform(X_test)
        train = pd.DataFrame(X_train_scale, columns=feature_names)
        test = pd.DataFrame(X_test_scale, columns=feature_names)

        return train, test

    # Remove the ids and target
    train = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    test = test.drop(columns=['SK_ID_CURR'])

    # Align the dataframes by the columns
    train, test = train.align(test, join='inner', axis=1)

    train['training_set'] = True
    test['training_set'] = False

    df_full = pd.concat([train, test])

    if encoder == 'LabelEncoder':
        # Do label encoding with all categorical features
        df_full = labelEnc(df_full)
        train = df_full[df_full['training_set'] == True]
        train = train.drop('training_set', axis=1)
        test = df_full[df_full['training_set'] == False]
        test = test.drop('training_set', axis=1)

        # When it meets MinMaxScaler
        if scaler == 'MinMaxScaler':
            # MinMax Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)
            # print('========================= Label Encoder & MinMax Scaler =========================')
            # # store values while rotating all algorithms used in this module
            # for alg in Alg_list:
            #     algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
            #     # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
            #     if F1 > best_F1:
            #         best_encoder = 'Label Encoder'
            #         best_scaler = 'MinMax Scaler'
            #         best_algorithm = algorithm
            #         best_estimator = estimator
            #         best_score = score
            #         best_precision = precision
            #         best_recall = recall
            #         best_F1 = F1
            #         best_pred_proba = pred_proba
            #         best_AUC = AUC
            #         best_y_test = y_test
        # When it meets MaxAbsScaler
        elif scaler == 'MaxAbsScaler':
            # MaxAbs Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)
            # print('========================= Label Encoder & MaxAbs Scaler =========================')
            # # store values while rotating all algorithms used in this module
            # for alg in Alg_list:
            #     algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
            #     # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
            #     if F1 > best_F1:
            #         best_encoder = 'Label Encoder'
            #         best_scaler = 'MaxAbs Scaler'
            #         best_algorithm = algorithm
            #         best_estimator = estimator
            #         best_score = score
            #         best_precision = precision
            #         best_recall = recall
            #         best_F1 = F1
            #         best_pred_proba = pred_proba
            #         best_AUC = AUC
            #         best_y_test = y_test
        # When it meets StandardScaler
        elif scaler == 'StandardScaler':
            # Standard Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)
            # print('========================= Label Encoder & Standard Scaler =========================')
            # # store values while rotating all algorithms used in this module
            # for alg in Alg_list:
            #     algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
            #     # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
            #     if F1 > best_F1:
            #         best_encoder = 'Label Encoder'
            #         best_scaler = 'Standard Scaler'
            #         best_algorithm = algorithm
            #         best_estimator = estimator
            #         best_score = score
            #         best_precision = precision
            #         best_recall = recall
            #         best_F1 = F1
            #         best_pred_proba = pred_proba
            #         best_AUC =AUC
            #         best_y_test = y_test
        # When it meets RobustScaler
        elif scaler == 'RobustScaler':
            # Robust Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)
            # print('========================= Label Encoder & Robust Scaler =========================')
            # # store values while rotating all algorithms used in this module
            # for alg in Alg_list:
            #     algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
            #     # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
            #     if F1 > best_F1:
            #         best_encoder = 'Label Encoder'
            #         best_scaler = 'Robust Scaler'
            #         best_algorithm = algorithm
            #         best_estimator = estimator
            #         best_score = score
            #         best_precision = precision
            #         best_recall = recall
            #         best_F1 = F1
            #         best_pred_proba = pred_proba
            #         best_AUC =AUC
            #         best_y_test = y_test
    elif encoder == 'OneHotEncoder':
        # Set the feature attributes and target attribute
        df_full = OneHotEnc(df_full)
        train = df_full[df_full['training_set'] == True]
        train = train.drop('training_set', axis=1)
        test = df_full[df_full['training_set'] == False]
        test = test.drop('training_set', axis=1)

        print(train.head())

        # When it meets MinMaxScaler
        if scaler == 'MinMaxScaler':
            # MinMax Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)
            # print('========================= OneHot Encoder & MinMax Scaler =========================')
            # # store values while rotating all algorithms used in this module
            # for alg in Alg_list:
            #     algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
            #     # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
            #     if F1 > best_F1:
            #         best_encoder = 'OneHot Encoder'
            #         best_scaler = 'MinMax Scaler'
            #         best_algorithm = algorithm
            #         best_estimator = estimator
            #         best_score = score
            #         best_precision = precision
            #         best_recall = recall
            #         best_F1 = F1
            #         best_pred_proba = pred_proba
            #         best_AUC = AUC
            #         best_y_test = y_test
        # When it meets MaxAbsScaler
        elif scaler == 'MaxAbsScaler':
            # MaxAbs Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)
            # print('========================= OneHot Encoder & MaxAbs Scaler =========================')
            # # store values while rotating all algorithms used in this module
            # for alg in Alg_list:
            #     algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
            #     # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
            #     if F1 > best_F1:
            #         best_encoder = 'OneHot Encoder'
            #         best_scaler = 'MaxAbs Scaler'
            #         best_algorithm = algorithm
            #         best_estimator = estimator
            #         best_score = score
            #         best_precision = precision
            #         best_recall = recall
            #         best_F1 = F1
            #         best_pred_proba = pred_proba
            #         best_AUC = AUC
            #         best_y_test = y_test
        # When it meets StandardScaler
        elif scaler == 'StandardScaler':
            # Standard Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)
            # print('========================= OneHot Encoder & Standard Scaler =========================')
            # # store values while rotating all algorithms used in this module
            # for alg in Alg_list:
            #     algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
            #     # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
            #     if F1 > best_F1:
            #         best_encoder = 'OneHot Encoder'
            #         best_scaler = 'Standard Scaler'
            #         best_algorithm = algorithm
            #         best_estimator = estimator
            #         best_score = score
            #         best_precision = precision
            #         best_recall = recall
            #         best_F1 = F1
            #         best_pred_proba = pred_proba
            #         best_AUC =AUC
            #         best_y_test = y_test
        # When it meets RobustScaler
        elif scaler == 'RobustScaler':
            # Robust Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)
            # print('========================= OneHot Encoder & Robust Scaler =========================')
            # # store values while rotating all algorithms used in this module
            # for alg in Alg_list:
            #     algorithm, estimator, score, parameter, precision, recall, F1, pred_proba, AUC = alg(X_train, y_train, X_test, y_test)
            #     # F1 score is the standard and set condition to store best scores, estimator, algorithm, etc
            #     if F1 > best_F1:
            #         best_encoder = 'OneHot Encoder'
            #         best_scaler = 'Robust Scaler'
            #         best_algorithm = algorithm
            #         best_estimator = estimator
            #         best_score = score
            #         best_precision = precision
            #         best_recall = recall
            #         best_F1 = F1
            #         best_pred_proba = pred_proba
            #         best_AUC =AUC
            #         best_y_test = y_test

    return train, test


def model(train, test, id, label, n_folds=5):
    # Extract feature names
    feature_names = train.columns.values.tolist()

    # Convert to np arrays
    train = np.array(train)
    test = np.array(test)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=False, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(train.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(train):
        # Training data for the fold
        train_features, train_labels = train[train_indices], label[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], label[valid_indices]

        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)

        # Train the model
        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], categorical_feature=cat_indices,
                  early_stopping_rounds=100, verbose=200)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits
