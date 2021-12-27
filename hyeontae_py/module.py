import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

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
    print('There are %d columns with more than 90%% missing in either the training or testing data.' % len(
        missing_columns))

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
