import pandas


def ats_get_capped_iqr(arraylike):
    """
    Generates columns with values that have been capped/truncated to 1.5 times the Inner-Quartile Range (IQR)
    
    Parameters
    ----------
    data : array-like, Series, or DataFrame
    
    Returns
    -------
    ats_df_capped_iqr : DataFrame of {1,0} as object
    """
    
    ats_df_capped_iqr = pandas.DataFrame()
    
    for column_name in arraylike.columns:
        
        # skip column that do no have any missing values
        if arraylike[column_name].isnull().sum() == 0:
            continue
        
        # Computing the Inner-Quartile Range (IQR)
        Q1 = arraylike[column_name].quantile(0.25)
        Q3 = arraylike[column_name].quantile(0.75)
        IQR = Q3 - Q1
        IQR150 = IQR * 1.50
        Q1IQR150 = Q1 - IQR150
        Q3IQR150 = Q3 + IQR150
        
        arraylike[column_name + '__capped_iqr'] = arraylike[column_name]
        arraylike[column_name + '__capped_iqr'][arraylike[column_name + '__capped_iqr'] < Q1IQR150] = Q1IQR150
        arraylike[column_name + '__capped_iqr'][arraylike[column_name + '__capped_iqr'] > Q3IQR150] = Q3IQR150
        
    return ats_df_capped_iqr
