# ats_get_missing_flags_test.py

import pandas

import ats_get_missing_flags

dictionary_of_data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': [0, 1, 0, 1, 0],
        'col3': ['lvl1', 'lvl2', 'lvl1', 'lvl2', 'lvl2'],
        'col4': [1, 2, 3, 4, pandas.np.nan,],
        'col5': [0, 1, 0, pandas.np.nan, pandas.np.nan,],
        'col6': ['lvl1', 'lvl2', pandas.np.nan, pandas.np.nan, pandas.np.nan],
        'col7': [pandas.np.nan, pandas.np.nan, pandas.np.nan, pandas.np.nan, pandas.np.nan]
        }

dataframe_of_data = pandas.DataFrame(
        data=dictionary_of_data
        )

for column_name in dataframe_of_data.columns:
    print(dataframe_of_data[column_name].isnull().sum())

df_missing_values_flags = pandas.DataFrame()

for column_name in dataframe_of_data.columns:
        
        # skip column with no missing values
        if dataframe_of_data[column_name].isnull().sum() == 0:
            print('INFO: Skipping [' + column_name + '] ' + 'No Missing Values')
            continue
        
        #df_missing_values_flags[column_name + '__missing_flag'] = dataframe_of_data[column_name].isnull().astype(int)
        #df_missing_values_flags[column_name + '__missing_flag'] = dataframe_of_data[column_name].isnull().astype(float)
        #df_missing_values_flags[column_name + '__missing_flag'] = dataframe_of_data[column_name].isnull().astype(str)
        ##df_missing_values_flags[column_name + '__missing_flag'] = dataframe_of_data[column_name].isnull().map(dict(True=1, False=0))
        ##df_missing_values_flags[column_name + '__missing_flag'] = dataframe_of_data[column_name].isnull().map(dict(True=1, False=0)).astype(str)
        df_missing_values_flags[column_name + '__missing_flag'] = dataframe_of_data[column_name].isnull().astype(int).astype(str)
        #df_missing_values_flags[column_name + '__missing_flag'] = dataframe_of_data[column_name].isnull().replace(False, 0, inplace=True)
        #df_missing_values_flags[column_name + '__missing_flag'] = dataframe_of_data[column_name].isnull().replace(False, 0)

print(df_missing_values_flags)

print(df_missing_values_flags.dtypes)

df_ats_get_missing_values_flags = ats_get_missing_flags.ats_get_missing_values_flags(dataframe_of_data)

print(df_ats_get_missing_values_flags)

#%%

print(df_ats_get_missing_values_flags)

print(df_ats_get_missing_values_flags.dtypes)

