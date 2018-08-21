import pandas


def ats_get_missing_values_flags(data):
    """
    Generates columns with flags indicating missing values in the existing columns
    
    Parameters
    ----------
    data : array-like, Series, or DataFrame
    
    Returns
    -------
    ats_df_missing_values_flags : DataFrame of {1,0} as object
    """
    
    ats_df_missing_values_flags = pandas.DataFrame()
    
    for column_name in data.columns:
        
        # skip column that do no have any missing values
        if data[column_name].isnull().sum() == 0:
            continue
        
        ats_df_missing_values_flags[column_name + '__missing_value_flag'] = data[column_name].isnull().astype(int).astype(str)
    
    return ats_df_missing_values_flags
