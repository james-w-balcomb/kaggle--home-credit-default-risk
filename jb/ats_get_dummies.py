import pandas

from pandas.core.frame import DataFrame
from pandas.core.series import Series


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
    data : array-like, Series, or DataFrame
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
        
        for column_name in columns_to_encode:
            dummy = pandas.get_dummies(data[column_name],
                                   prefix=column_name,
                                   prefix_sep='__',
                                   dummy_na=False,
                                   columns=None,
                                   sparse=False,
                                   drop_first=False)
        
            ats_df_dummies = pandas.concat([ats_df_dummies, dummy], axis=1)
            
            # drop the most frequent value for the reference level
            dropped_column_name = column_name + '__' + data.groupby([column_name]).size().idxmax()
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
            dropped_column_name = data.name + '__' + data.value_counts().index[0]
            ats_df_dummies.drop(dropped_column_name, axis=1, inplace=True)
            
    else:
        return ats_df_dummies
    
    return ats_df_dummies
