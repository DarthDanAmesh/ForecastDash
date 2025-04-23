

def enhance_feature_engineering(df):
    """Create more advanced features to improve forecast accuracy"""
    # Create date-based features
    df['month'] = df['Created On'].dt.month
    df['quarter'] = df['Created On'].dt.quarter
    df['year'] = df['Created On'].dt.year
    df['day_of_week'] = df['Created On'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_month_start'] = df['Created On'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Created On'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Created On'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Created On'].dt.is_quarter_end.astype(int)
    
    # Create regional indicators
    df['region'] = df['Country Key Ship-to'].map(lambda x: 'Italy' if x == 'IT' else 'Other')
    
    # Create product category features
    df['product_category'] = df['Material Group'].astype(str)
    
    # Order vs delivery gap analysis
    if 'Customer Ref. Date' in df.columns and 'Act. Gds Issue Date' in df.columns:
        df['order_to_delivery_days'] = (df['Act. Gds Issue Date'] - df['Customer Ref. Date']).dt.days
    
    # Sales organization performance
    df['sales_org'] = df['Sales Organization'].astype(str)
    
    return df