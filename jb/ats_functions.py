# ats_functions.py

import numpy
import pandas

from pandas.core.frame import DataFrame
from pandas.core.series import Series

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
# https://datascience.stackexchange.com/questions/29671/how-to-count-occurrences-of-values-within-specific-range-by-row

def count_values_in_range(series, range_min, range_max):

    # "between" returns a boolean Series equivalent to left <= series <= right.
    # NA values will be treated as False.
    return series.between(left=range_min, right=range_max).sum()

    # Alternative approach:
    # return ((range_min <= series) & (series <= range_max)).sum()


def test__count_values_in_range():
    import pandas as pd

    df = pd.DataFrame({
        'id0': [1.71, 1.72, 1.72, 1.23, 1.71],
        'id1': [6.99, 6.78, 6.01, 8.78, 6.43],
        'id2': [3.11, 3.11, 4.99, 0.11, 2.88]})

    range_min, range_max = 1.72, 6.43

    df["n_values_in_range"] = df.apply(
            func=lambda row: count_values_in_range(row, range_min, range_max), axis=1)

    print(df)
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
# https://datascience.stackexchange.com/questions/12645/how-to-count-the-number-of-missing-values-in-each-row-in-pandas-dataframe

def count_columns_with_missing_values(data_frame_name):
    #You can apply a count over the rows like this:
    data_frame_name.apply(lambda x: x.count(), axis=1)

    # You can add the result as a column like this:
    data_frame_name['full_count'] = data_frame_name.apply(lambda x: x.count(), axis=1)
    
    # When using pandas, try to avoid performing operations in a loop, including apply, map, applymap etc. That's slow!
    # If you want to count the missing values in each column, try:
    data_frame_name.isnull().sum()
    # or
    data_frame_name.isnull().sum(axis=0)
    # On the other hand, you can count in each row (which is your question) by:
    data_frame_name.isnull().sum(axis=1)
    # It's roughly 10 times faster than Jan van der Vegt's solution(BTW he counts valid values, rather than missing values):
    # In [18]: %timeit -n 1000 df.apply(lambda x: x.count(), axis=1)
    # 1000 loops, best of 3: 3.31 ms per loop
    # In [19]: %timeit -n 1000 df.isnull().sum(axis=1)
    # 1000 loops, best of 3: 329 µs per loop
    
    numpy.logical_not(data_frame_name.isnull()).sum()


def test__count_columns_with_missing_values():
#    test_df:
#    A   B   C
#    0:  1   1   3
#    1:  2   nan nan
#    2:  nan nan nan
#    output:
#    0:  3
#    1:  1
#    2:  0
    pass

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
def is_numeric_scalar(scalar_value):
    # Iterating the string and checking for numeric characters
    # Incrementing the counter if a numeric character is found
    # And adding the character to new string if not numeric
    # NOTE: iteration over a string is actually iteration over the individual characters
    #TODO(JamesBalcomb): add try-catch for categorical that are actually nan/float
    
    scalar_value_is_numeric_scalar = None
    
    for single_character in scalar_value:
        #print(scalar_value)
        #print(single_character)
        if (single_character.isnumeric()) == True:
            scalar_value_is_numeric_scalar = True
            #print(True)
        else:
            scalar_value_is_numeric_scalar = False
            #print(False)
            #continue
            break
        
        #if scalar_value_is_numeric_scalar == False:
        #    continue
    
    return scalar_value_is_numeric_scalar
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
def is_numeric_series(pandas_series):
    # Iterating the string and checking for numeric characters
    # Incrementing the counter if a numeric character is found
    # And adding the character to new string if not numeric
    # NOTE: iteration over a string is actually iteration over the individual characters
    #TODO(JamesBalcomb): add try-catch for categorical that are actually nan/float
    #TODO(JamesBalcomb): figure out how to handle categorical that are all numbers (convert to words, just skip categorical, etc.)
    
    pandas_series_is_numeric_series = None
    
    #TODO(JamesBalcomb): decide on early-exit when dtype is bool, float, int, etc. (.:. TypeError: 'numpy.float64' object is not iterable)
    #if pandas_series.dtype == numpy.number:
    #if pandas.api.types.is_numeric_dtype(pandas_series):
    
    if pandas.api.types.is_numeric_dtype(pandas_series):
        pandas_series_is_numeric_series = True
        #print('is_numeric_dtype = True')
    else:
        #print('is_numeric_dtype = False (' + str(type(pandas_series)) + ')')
        #print('is_numeric_dtype = False')
        #print(type(pandas_series))
        #print(pandas_series.dtype)
        for index_number in pandas_series.index:
            pandas_series_value = pandas_series.loc[index_number]
            #print(pandas_series_value)
            if is_numeric_scalar(pandas_series_value):
                pandas_series_is_numeric_series = True
                #print(True)
            else:
                pandas_series_is_numeric_series = False
                #print(False)
                break
    
    return pandas_series_is_numeric_series
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
def is_boolean_series(pandas_series):
    
    pandas_series_is_boolean_series = None
    
    if pandas_series.nunique(dropna=False) == 2:
        # if set(pandas_series.unique().tolist()) == {0,1}:
        #if set(pandas_series.unique().astype(int).tolist()) == {0,1}:
            #TODO(JamesBalcomb): ValueError: cannot convert float NaN to integer
            #TODO(JamesBalcomb): ValueError: invalid literal for int() with base 10: 'M'
        pandas_series_set = set(pandas_series.unique().astype(str).tolist())
        if (
                pandas_series_set == {False,True} or
                pandas_series_set == {'0','1'}
            ):
            pandas_series_is_boolean_series = True
        else:
            pandas_series_is_boolean_series = False
    else:
        pandas_series_is_boolean_series = False
    
    return pandas_series_is_boolean_series
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
def is_binary_series(pandas_series):
    
    pandas_series_is_binary_series = None
    
    if pandas_series.nunique(dropna=False) == 2:
        pandas_series_is_binary_series = True
    else:
        pandas_series_is_binary_series = False
    
    return pandas_series_is_binary_series
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
def make_capped_columns(pandas_data_frame):
    
    pandas_data_frame_column_names = sorted(pandas_data_frame.columns.tolist())
    
    new_data_frame = pandas.DataFrame()
    
    column_name_suffix = '__capped_iqr'
    
    for column_name in pandas_data_frame_column_names:
        first_quartile = pandas_data_frame[column_name].quantile(0.25)
        third_quartile = pandas_data_frame[column_name].quantile(0.75)
        inner_quartile_range = third_quartile - first_quartile
        IQR150 = inner_quartile_range * 1.50
        Q1IQR150 = first_quartile - IQR150
        Q3IQR150 = third_quartile + IQR150
        new_data_frame[column_name + column_name_suffix] = pandas_data_frame[column_name]
        new_data_frame.loc[new_data_frame[column_name + column_name_suffix] < Q1IQR150, column_name + column_name_suffix] = Q1IQR150
        new_data_frame.loc[new_data_frame[column_name + column_name_suffix] > Q3IQR150, column_name + column_name_suffix] = Q3IQR150
    
    return new_data_frame
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
def make_transformed_columns(pandas_data_frame):
    
    pandas_data_frame_column_names = sorted(pandas_data_frame.columns.tolist())
    
    new_data_frame = pandas.DataFrame()
    
    for column_name in pandas_data_frame_column_names:
        
        column_name_suffix = '__loge'
        new_data_frame[column_name + column_name_suffix] = numpy.log(pandas_data_frame[column_name])
        
        column_name_suffix = '__log1p'
        new_data_frame[column_name + column_name_suffix] = numpy.log1p(pandas_data_frame[column_name])
        
        column_name_suffix = '__log2'
        new_data_frame[column_name + column_name_suffix] = numpy.log2(pandas_data_frame[column_name])
        
        column_name_suffix = '__log10'
        new_data_frame[column_name + column_name_suffix] = numpy.log10(pandas_data_frame[column_name])
        
        column_name_suffix = '__pwr2'
        new_data_frame[column_name + column_name_suffix] = numpy.power(pandas_data_frame[column_name], 2)
        
        column_name_suffix = '__pwr3'
        new_data_frame[column_name + column_name_suffix] = numpy.power(pandas_data_frame[column_name], 3)
        
        column_name_suffix = '__pwr4'
        new_data_frame[column_name + column_name_suffix] = numpy.power(pandas_data_frame[column_name], 4)
        
        column_name_suffix = '__pwr5'
        new_data_frame[column_name + column_name_suffix] = numpy.power(pandas_data_frame[column_name], 5)
        
        column_name_suffix = '__pwr6'
        new_data_frame[column_name + column_name_suffix] = numpy.power(pandas_data_frame[column_name], 6)
        
        column_name_suffix = '__pwr7'
        new_data_frame[column_name + column_name_suffix] = numpy.power(pandas_data_frame[column_name], 7)
        
        column_name_suffix = '__pwr8'
        new_data_frame[column_name + column_name_suffix] = numpy.power(pandas_data_frame[column_name], 8)
        
        column_name_suffix = '__pwr9'
        new_data_frame[column_name + column_name_suffix] = numpy.power(pandas_data_frame[column_name], 9)
        
        column_name_suffix = '__exp'
        new_data_frame[column_name + column_name_suffix] = numpy.exp(pandas_data_frame[column_name])
        
        column_name_suffix = '__sqrt'
        new_data_frame[column_name + column_name_suffix] = numpy.sqrt(pandas_data_frame[column_name])
    
    return new_data_frame
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
def make_imputed_columns(pandas_data_frame):
    
    pandas_data_frame_column_names = sorted(pandas_data_frame.columns.tolist())
    
    new_data_frame = pandas.DataFrame()
    
    column_name_suffix = '__imputed_mean'
    
    for column_name in pandas_data_frame_column_names:
        new_data_frame[column_name + column_name_suffix] = pandas_data_frame[column_name].fillna(pandas_data_frame[column_name].mean())
    
    column_name_suffix = '__imputed_median'
    
    for column_name in pandas_data_frame_column_names:
        new_data_frame[column_name + column_name_suffix] = pandas_data_frame[column_name].fillna(pandas_data_frame[column_name].mean())
    
    return new_data_frame
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
def make_zscored_columns(pandas_data_frame):
    
    pandas_data_frame_column_names = sorted(pandas_data_frame.columns.tolist())
    
    new_data_frame = pandas.DataFrame()
    
    column_name_suffix = '__zscore'
        
    for column_name in pandas_data_frame_column_names:
        new_data_frame[column_name + column_name_suffix] = (pandas_data_frame[column_name] - pandas_data_frame[column_name].mean()) / pandas_data_frame[column_name].std(ddof=0)
    
    return new_data_frame
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
import lightgbm as lgb

def get_score_lightgbm(df, usecols, params, dropcols=[]):
    
    dtrain = lgb.Dataset(df[usecols].drop(dropcols, axis=1), df['TARGET'])
    
    eval = lgb.cv(
            params,
            dtrain,
            nfold=5,
            stratified=True,
            num_boost_round=20000,
            early_stopping_rounds=200,
            verbose_eval=100,
            seed = 5,
            show_stdv=True
            )
    
    return max(eval['auc-mean'])
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #

def ats_get_dummies(data,
                    prefix=None,
                    prefix_sep='_',
                    dummy_na=False,
                    sparse=False,
                    included_column_names=None,
                    excluded_column_names=None):
    """
    Generates columns with flags indicating missing values in the existing columns
    
    Parameters
    ----------
    data : DataFrame or Series
    
    prefix : string, list of strings, or dict of strings, default None
    prefix_sep : string, default '_'
    dummy_na : bool, default False
    sparse : bool, default False
    included_column_names : list-like, default None
    excluded_column_names : list-like, default None
    
    Returns
    -------
    ats_df_dummies : DataFrame
    """
    
    ats_df_dummies = pandas.DataFrame()
    
    if isinstance(data, DataFrame):
        
        if included_column_names is None:
            if excluded_column_names is not None:
                columns_to_encode = list(set(data.select_dtypes(include=['object', 'category']).columns) - set(excluded_column_names))
            else:
                columns_to_encode = list(set(data.select_dtypes(include=['object', 'category']).columns))
        else:
            if excluded_column_names is not None:
                columns_to_encode = list(set(included_column_names) - set(excluded_column_names))
            else:
                columns_to_encode = list(set(included_column_names))
        
        for column_name in columns_to_encode:
            if(data[column_name].dtype == pandas.np.number):
                columns_to_encode.remove(column_name)
            elif data[column_name].dtype.name == 'category':
                if data[column_name].cat.ordered:
                    columns_to_encode.remove(column_name)
        
        columns_to_encode = sorted(columns_to_encode)
        
        #TODO(JamesBalcomb): sanitize strings for column names
        # # Remove all the spaces in python
        # data[column_name].replace(" ", "")
        # my_string = re.sub('[^0-9a-zA-Z]+', '*', my_string)
        # this will perform slightly quicker if you pre-compile the regex, e.g.,
        # regex = re.compile('[^0-9a-zA-Z]+')
        # regex.sub('*', my_string)        
        
        for column_name in columns_to_encode:
            #print('column_name: ' + column_name)
            #print('dtype: ')
            #print(data[column_name].dtype)
            #print(data.groupby([column_name]))
            #print(data.groupby([column_name]).size())
            #print(data.groupby([column_name]).size().idxmax())
            #print('pandas.get_dummies: ' + column_name)
            dummy = pandas.get_dummies(data[column_name], prefix=column_name, prefix_sep='__', dummy_na=False, columns=None, sparse=False, drop_first=False)
            
            ats_df_dummies = pandas.concat([ats_df_dummies, dummy], axis=1)
            
            # drop the most frequent value for the reference level
            dropped_column_name = column_name + '__' + str(data.groupby([column_name]).size().idxmax())
            #TODO(JamesBalcomb): TypeError: must be str, not numpy.int64
            ats_df_dummies.drop(dropped_column_name, axis=1, inplace=True)
        
        return ats_df_dummies
    
    elif isinstance(data, Series):
        if(data.dtype == pandas.np.number):
            return ats_df_dummies
        elif data.dtype.name == 'category':
            if data.cat.ordered:
                return ats_df_dummies
        else:
            dummy = pandas.get_dummies(data,
                                   prefix=data.name,
                                   prefix_sep='__',
                                   dummy_na=False,
                                   columns=None,
                                   sparse=False,
                                   drop_first=False)
            
            ats_df_dummies = pandas.concat([ats_df_dummies, dummy], axis=1)
            
            # drop the most frequent value for the reference level
            print(data.name)
            print(data.value_counts())
            print(data.value_counts().index[0])
            dropped_column_name = data.name + '__' + data.value_counts().index[0]
            ats_df_dummies.drop(dropped_column_name, axis=1, inplace=True)
            
    else:
        return ats_df_dummies
    
    return ats_df_dummies
# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
