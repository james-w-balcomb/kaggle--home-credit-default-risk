# Author: Luke Schoen 2017
import pandas as pd
import numpy as np
import functools

# Create DataFrame
# df = pd.DataFrame(np.random.randn(10,2))

# Populate with NaN values
df = pd.DataFrame({'col1': ['1.111', '2.111', '3.111', '4.111'], 'col2': ['4.111', '5.111', np.NaN, '7.111'], 'col3': ['8', '9', np.NaN, np.NaN], 'col4': ['12', '13', '14', '15']})

# Round all values to 2 decimal places
df.apply(functools.partial(np.round, decimals=2))

# Populate DataFrame column 0 and indexed rows 2 to 6 with NaN values
df.iloc[3:6,0] = np.nan

def get_percentage_missing(series):
    """ Calculates percentage of NaN values in DataFrame
    :param series: Pandas DataFrame object
    :return: float
    """
    num = series.isnull().sum()
    den = len(series)
    return round(num/den, 2)

# Only include columns that contain any NaN values
df_with_any_null_values = df[df.columns[df.isnull().any()].tolist()]

get_percentage_missing(df_with_any_null_values)

# Show qty of each value in a Column 
# df.astype(str).groupby(['col1']).sum()

# Show DataFrame
df
# df.head()

# Show DataFrame info
print(df.describe())
print(df.info())

# Iterate over columns in DataFrame and delete those with where >30% of the values are null/NaN
for name, values in df_with_any_null_values.iteritems():
    print("%r: %r" % (name, values) )
    if get_percentage_missing(df_with_any_null_values[name]) > 0.3:
        print("Deleting Column %r: " % (name) )
        df_with_any_null_values.drop(name, axis=1, inplace=True)
        
# Iterate over columns in DataFrame and delete rows of columns where any values are null/NaN
for name, values in df_with_any_null_values.iteritems():
    if name != "id":
        if get_percentage_missing(df_with_any_null_values[name]) < 0.01:
            print("Retained Column: %r, but removed its null and NaN valued rows" % (name) )
            print("BEFORE %r: %r" % (name, values) )
            df_with_any_null_values.dropna(axis=0, how="any", subset=[name], inplace=True)
            print("AFTER %r: %r" % (name, values) )
            
# Select only Columns of certain types
# http://pandas.pydata.org/pandas-docs/version/0.19.2/generated/pandas.DataFrame.select_dtypes.html
df.select_dtypes(include=['int', 'float64', 'floating', 'number'], exclude=['O'])

# Iterate over Columns and perform modifications depending on the type
# IMPORTANT NOTE: ENSURE ONLY USE AFTER REMOVE NAN VALUES
for col in df.columns:
    for name, values in df[col].iteritems():
        # print("%r, %r" % (name, values))
        if(values.dtype == np.float64 or values.dtype == np.int64):
            print("float or int type %r" % (values.dtype))
            # treat_numeric(df[name])
        elif(df[name].dtype == np.str):
            print("string type %r" % (df[name].dtype))
            #treat_str(df[y])
        elif(df[name].dtype == np.object):
            print("object type %r" % (df[name].dtype))
            #treat_object(df[name])
        else:
            print("other type %r" % (values.dtype))
    