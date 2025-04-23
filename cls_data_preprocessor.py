import pandas as pd


# Data Processing Module
class DataProcessor:
    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the input data"""
        if df is None:
            return None
            
        # Convert date columns
        date_cols = ['Created On', 'Pland Gds Mvmnt Date', 'Act. Gds Issue Date', 'Customer Ref. Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle missing values
        df['Customer Reference'].fillna('Unknown', inplace=True)
        df['Customer Material'].fillna('Unknown', inplace=True)
        
        # Add derived features
        if 'Pland Gds Mvmnt Date' in df.columns and 'Act. Gds Issue Date' in df.columns:
            df['Delivery_Delay'] = (df['Act. Gds Issue Date'] - df['Pland Gds Mvmnt Date']).dt.days
        
        return df
    

    @staticmethod
    def prepare_time_series(df: pd.DataFrame, target_col: str = 'Delivery Quantity', 
                          date_col: str = 'Act. Gds Issue Date', freq: str = 'M') -> pd.Series:
        """Convert dataframe to time series format"""
        if df is None or date_col not in df.columns or target_col not in df.columns:
            return None
            
        ts = df.set_index(date_col)[target_col]
        ts = ts.resample(freq).sum()
        return ts