import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import missingno as msno
from impyute.imputation.cs import mice
import module
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

import gc
import re



warnings.filterwarnings('ignore')

# print("pandas version: ", pd.__version__)
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 150)

# read dataset
application_train = pd.read_csv('train_remove_outlier.csv', encoding='utf-8')
application_test = pd.read_csv('application_test.csv', encoding='utf-8')
bureau = pd.read_csv('bureau_remove_outlier.csv', encoding='utf-8')
# bureau_balance = pd.read_csv('bureau_balance.csv', encoding='utf-8')
previous_application = pd.read_csv('previous_application.csv', encoding='utf-8')
# credit_card_balance = pd.read_csv('credit_card_balance.csv', encoding='utf-8')
# POS_CASH_balance = pd.read_csv('POS_CASH_balance.csv', encoding='utf-8')
# installments_payments = pd.read_csv('installments_payments.csv', encoding='utf-8')


#Glimpse the data
print("application_train -  rows:",application_train.shape[0]," columns:", application_train.shape[1])
print("application_test -  rows:",application_test.shape[0]," columns:", application_test.shape[1])
print("bureau -  rows:",bureau.shape[0]," columns:", bureau.shape[1])
# print("bureau_balance -  rows:",bureau_balance.shape[0]," columns:", bureau_balance.shape[1])
# print("credit_card_balance -  rows:",credit_card_balance.shape[0]," columns:", credit_card_balance.shape[1])
# print("installments_payments -  rows:",installments_payments.shape[0]," columns:", installments_payments.shape[1])
print("previous_application -  rows:",previous_application.shape[0]," columns:", previous_application.shape[1])
# print("POS_CASH_balance -  rows:",POS_CASH_balance.shape[0]," columns:", POS_CASH_balance.shape[1])

#application_train
print(application_train.head())
print(application_train.columns.values)


#application_test
print(application_test.head())
print(application_test.columns.values)


#bureau
print(bureau.head())
print(bureau.columns.values)

#bureau_balance
# print(bureau_balance.head())

#previous_application
print(previous_application.head())
print(previous_application.columns.values)


#credit_card_balance
# print(credit_card_balance.head())
# print(credit_card_balance.columns.values)


#installments_payments
# print(installments_payments.head())
# print(installments_payments.columns.values)


#POS_CASH_balance
# print(POS_CASH_balance.head())
# print(POS_CASH_balance.columns.values)

label = 'TARGET'
id = 'SK_ID_CURR'

# Merging into one dataframe
train, test = module.merge_df(application_train, application_test, bureau, id, 'SK_ID_BUREAU', 'bureau')
train, test = module.merge_df(application_train, application_test, previous_application, id, 'SK_ID_PREV', 'prev_app')

# Handling missing values on train, test dataset
train, test = module.handle_missing_values(train, test, 50)

# Save file into CSV
# module.save_csv(train, test, 'train_new.csv', 'test_new.csv')

# Remove low correlated features in each dataset
train, test = module.corr_selection(train, test, 0.7)

# Save file into CSV
# module.save_csv(train, test, 'train_corr_removed.csv', 'test_corr_removed.csv')

fe_train, fe_test = module.feature_engineering(train, test, 'OneHotEncoder', 'StandardScaler')
# print(fe_train.head())
# print(fe_test.head())
# module.save_csv(fe_train, fe_test, 'fffinal_train.csv', 'fffinal_test.csv')

fe_train = fe_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
fe_test = fe_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

module.model(fe_train,fe_test, id, label, 5)



