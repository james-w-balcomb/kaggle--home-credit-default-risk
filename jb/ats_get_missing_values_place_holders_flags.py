
def ats_get_missing_values_flags(data):
    """
    Generates columns with flags indicating missing value place holders in the existing columns
    
    Parameters
    ----------
    data : array-like, Series, or DataFrame
    
    Returns
    -------
    missing_value_place_holders_flags : DataFrame or SparseDataFrame
    """
    
    import pandas
    
    df_missing_value_place_holders_flags = pandas.DataFrame()
    
    for column_name in data.columns:
        
        # skip column that do no have any missing values
        if data[column_name].isnull().sum() == 0:
            continue
        
        df_missing_value_place_holders_flags['is_missing__' + column_name] = data[column_name].isnull()
        df_missing_value_place_holders_flags['is_missing__' + column_name] = data[column_name].map(lambda v: 1 if v else 0)
        
    return df_missing_value_place_holders_flags
