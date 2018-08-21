# ats_functions.py

import numpy
import pandas

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
        print('is_numeric_dtype = True')
    else:
        #print('is_numeric_dtype = False (' + str(type(pandas_series)) + ')')
        print('is_numeric_dtype = False')
        print(type(pandas_series))
        print(pandas_series.dtype)
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
    
    if pandas_series.nunique() == 2:
        # if set(pandas_series.unique().tolist()) == {0,1}:
        if set(pandas_series.unique().astype(int).tolist()) == {0,1}:
            #TODO(JamesBalcomb): ValueError: cannot convert float NaN to integer
            pandas_series_is_boolean_series = True
        else:
            pandas_series_is_boolean_series = False
    else:
        pandas_series_is_boolean_series = False
    
    return pandas_series_is_boolean_series

# ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### # ### #
