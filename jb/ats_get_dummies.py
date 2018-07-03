#%%
import pandas as pd

#%%
data_file_path = 'C:/Development/kaggle--home-credit-default-risk/data/'
data_file_name = 'application_train.csv'

df = pd.read_table(data_file_path + data_file_name, sep=',')

#%%
def ats_get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None):
    
    ats_dummies_metadata = {}
    
    if not column_names:
        if df_categorical_column_names:
            column_names = df_categorical_column_names
        else:
            list(df.select_dtypes(include=['category']).columns)
    
    for column_name in column_names:
        df_new_dummies = pd.get_dummies(df[column_name], prefix=column_name, prefix_sep='__')
        #df_model_data = df_model_data.join(df_new_dummies)
        df_model_data = pd.concat([df_model_data, df_new_dummies], axis=1)
        
        # drop most frequent variable for ref category
        #dropped_column_name = df.groupby([column_name]).size().idxmax()
        dropped_column_name = column_name + '__' + df.groupby([column_name]).size().idxmax()
        df_model_data.drop(dropped_column_name, axis=1, inplace=True)
        
        print(column_name + " dropping " + dropped_column_name)
        print(df.groupby([column_name]).size())

#%%
df_nonmodel_column_names

#%%
columns = set(df_dtype_object_column_names) - set(df_nonmodel_column_names)
columns


#%%
df['combined'] = df['bar'].astype(str) + '_' + df['foo'] + '_' + df['new']

df["combined"] = df["foo"].str.cat(df[["bar", "new"]].astype(str), sep="_")

