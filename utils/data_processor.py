import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    
    def __init__(self, file_path=None, df=None):
     
        self.file_path = file_path
        self.df = df
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.eda_results = {}
        self.price_column = None
        self.date_column = None
        
    def load_data(self):
  
        if self.df is not None:
            df = self.df.copy()
        elif self.file_path:
            df = pd.read_csv(self.file_path)
        else:
            raise ValueError("File path or DataFrame must be provided")
        
        print(f"\n File loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f" Columns: {list(df.columns)}")
        
        date_candidates = [
            'Date', 'date', 'DATE', 'timestamp', 'Timestamp', 'DateTime',
            'datetime', 'time', 'Time', 'index', 'dt', 'day', 'trade_date',
            'Tarih', 'تاریخ', 'fecha', 'data', 'datum'
        ]
        
        self.date_column = None
        for col in date_candidates:
            if col in df.columns:
                self.date_column = col
                break
        
        if self.date_column is None:
            for col in df.columns[:3]:
                try:
                    pd.to_datetime(df[col])
                    self.date_column = col
                    print(f" Auto-detected date column: '{col}'")
                    break
                except:
                    continue
        
        if self.date_column:
            df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
            df = df.dropna(subset=[self.date_column])
            df.set_index(self.date_column, inplace=True)
            df.sort_index(inplace=True)
            print(f" Date column: '{self.date_column}'")
        else:
            print(" No date column found, using default index")
        
        price_candidates = [
            'Price', 'price', 'PRICE', 'Close', 'close', 'CLOSE',
            'Adj Close', 'adj_close', 'Adj_Close',
            'Value', 'value', 'VALUE',
            'Last', 'last', 'LAST',
            'Gold_Price', 'gold_price', 'GOLD_PRICE',
            'XAU/USD', 'XAUUSD', 'xauusd', 'GOLD',
            'Spot', 'spot', 'SPOT',
            'Fiyat', 'prix', 'precio', 'prezzo',
            
            'قیمت', 'قیمت طلا', 'طلا', 'ارزش',
            'آخرین قیمت', 'قیمت پایانی',
            
            'Fiyat', 'Altın', 'altin_fiyat',
            
            'Gold', 'gold', 'GOLD'
        ]
        
        self.price_column = None
        for col in price_candidates:
            if col in df.columns:
                self.price_column = col
                break
        
        if self.price_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['price', 'gold', 'close', 'value', 'قیمت', 'طلا']):
                    self.price_column = col
                    print(f" Auto-detected price column (keyword match): '{col}'")
                    break
            
            if self.price_column is None and len(numeric_cols) > 0:
                nan_counts = df[numeric_cols].isnull().sum()
                self.price_column = nan_counts.idxmin()
                print(f"Auto-detected price column (lowest NaN): '{self.price_column}'")
            
            if self.price_column is None and len(numeric_cols) > 0:
                self.price_column = numeric_cols[0]
                print(f" Using first numeric column as price: '{self.price_column}'")
        
        if self.price_column and self.price_column != 'Price':
            df.rename(columns={self.price_column: 'Price'}, inplace=True)
            self.price_column = 'Price'
            print(f" Renamed price column to 'Price'")
        
        if self.price_column is None:
            raise ValueError("""
             Could not find a price column!
            
            Please ensure your CSV has a column named one of:
            - 'Price', 'Close', 'Value', 'Gold_Price', 'XAUUSD'
            - 'قیمت', 'طلا', 'قیمت طلا'
            
            Your columns: """ + str(list(df.columns)))
        
        self.df = df
        
        print(f" Data loaded successfully")
        print(f"    Records: {len(self.df):,}")
        print(f"    Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"    Price column: '{self.price_column}'")
        print(f"    Total features: {len(self.df.columns)}")
        
        return self.df
    
    def show_data_info(self):
        if self.df is None:
            print("No data loaded!")
            return
        
        print("\n" + "="*60)
        print(" DATASET INFORMATION")
        print("="*60)
        
        print(f"\n Shape: {self.df.shape}")
        print(f" Date Range: {self.df.index.min()} to {self.df.index.max()}")
        print(f" Price Column: {self.price_column}")
        print(f" Date Column: {self.date_column}")
        
        print(f"\n Columns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            missing = self.df[col].isnull().sum()
            missing_pct = (missing / len(self.df)) * 100
            print(f"   {i:2d}. {col:<30s} | {str(dtype):<10s} | Missing: {missing:>6d} ({missing_pct:>5.1f}%)")
        
        print("\n Numeric Columns:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:10]:
            stats = self.df[col].describe()
            print(f"   {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        
        print("="*60 + "\n")
    
    def handle_missing_values(self, method='interpolate'):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_missing = self.df.isnull().sum().sum()
        print(f"\n Initial missing values: {initial_missing}")
        
        if initial_missing == 0:
            print(" No missing values found!")
            return self.df
        
        threshold = len(self.df) * 0.5
        cols_to_drop = self.df.columns[self.df.isnull().sum() > threshold].tolist()
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print(f" Dropped columns with >50% missing: {cols_to_drop}")
        
        if method == 'interpolate':
            self.df.interpolate(method='linear', limit_direction='both', inplace=True)
        elif method == 'ffill':
            self.df.fillna(method='ffill', inplace=True)
            self.df.fillna(method='bfill', inplace=True)
        elif method == 'bfill':
            self.df.fillna(method='bfill', inplace=True)
            self.df.fillna(method='ffill', inplace=True)
        elif method == 'mean':
            for col in self.df.select_dtypes(include=[np.number]).columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif method == 'median':
            for col in self.df.select_dtypes(include=[np.number]).columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(method='ffill', inplace=True)
                self.df[col].fillna(method='bfill', inplace=True)
                self.df[col].fillna(0, inplace=True)
        
        final_missing = self.df.isnull().sum().sum()
        print(f" Missing values after handling: {final_missing}")
        
        return self.df
    
    def add_technical_indicators(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if 'Price' not in self.df.columns:
            print(" 'Price' column not found. Available columns:", list(self.df.columns))
            return self.df
        
        df = self.df.copy()
        price = df['Price']
        
        print("\n Adding technical indicators...")
        added_count = 0
        
        for window in [5, 7, 10, 14, 20, 30, 50, 60]:
            df[f'MA_{window}'] = price.rolling(window=window, min_periods=1).mean()
            added_count += 1
        
        for span in [12, 26]:
            df[f'EMA_{span}'] = price.ewm(span=span, adjust=False).mean()
            added_count += 1
        
        for period in [7, 14]:
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            df[f'RSI_{period}'].fillna(50, inplace=True)
            added_count += 1
        
        df['BB_middle'] = price.rolling(window=20, min_periods=1).mean()
        bb_std = price.rolling(window=20, min_periods=1).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        added_count += 4
        
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        added_count += 3
        
        for window in [7, 30]:
            df[f'Volatility_{window}'] = price.pct_change().rolling(window=window).std()
            added_count += 1
        
        for period in [1, 3, 5, 10]:
            df[f'Returns_{period}d'] = price.pct_change(periods=period) * 100
            added_count += 1
        
        for period in [5, 20]:
            df[f'Momentum_{period}'] = price.diff(period)
            added_count += 1
        
        for lag in [1, 7, 30]:
            df[f'Price_Lag_{lag}'] = price.shift(lag)
            added_count += 1
        
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        added_count += 3
        
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        self.df = df
        print(f"Added {added_count} technical indicators")
        print(f" Total features: {len(self.df.columns)}")
        
        return self.df
    
    def perform_eda(self):
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        print("\n Performing EDA...")
        eda = {}
        
        eda['basic_info'] = {
            'total_records': len(self.df),
            'start_date': str(self.df.index.min()),
            'end_date': str(self.df.index.max()),
            'total_features': len(self.df.columns),
            'numeric_features': len(self.df.select_dtypes(include=[np.number]).columns)
        }
        
        eda['descriptive_stats'] = self.df.describe()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        important_cols = []
        for col in numeric_cols:
            col_upper = col.upper()
            if any(kw in col_upper for kw in ['PRICE', 'MA_', 'RSI', 'MACD', 'BB_', 'RETURNS', 'MOMENTUM', 'VOLATILITY']):
                important_cols.append(col)
        
        if len(important_cols) > 25:
            if 'Price' in self.df.columns:
                correlations = self.df[important_cols].corr()['Price'].abs().sort_values(ascending=False)
                important_cols = correlations.index[:25].tolist()
            else:
                important_cols = important_cols[:25]
        
        if len(important_cols) > 1:
            eda['correlation_matrix'] = self.df[important_cols].corr()
        
       
        if 'Price' in self.df.columns:
            price_corr = self.df[numeric_cols].corr()['Price'].sort_values(ascending=False)
            eda['price_correlation'] = price_corr.head(20)
        
       
        if 'Price' in self.df.columns:
            price = self.df['Price']
            eda['rolling_stats'] = {
                'rolling_30_mean': price.rolling(window=30).mean(),
                'rolling_30_std': price.rolling(window=30).std(),
            }
            
            
            if len(price) > 30:
                start_price = price.iloc[:30].mean()
                end_price = price.iloc[-30:].mean()
                eda['overall_trend'] = {
                    'start_avg_price': start_price,
                    'end_avg_price': end_price,
                    'change_pct': ((end_price - start_price) / start_price) * 100
                }
        
        
        if 'Price' in self.df.columns:
            monthly = self.df['Price'].groupby(self.df.index.month).mean()
            eda['monthly_seasonality'] = monthly
        
        self.eda_results = eda
        print(" EDA completed")
        
        return eda
    
    def prepare_lstm_data(self, sequence_length=60, test_size=0.2):
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        if 'Price' not in self.df.columns:
            print("\n ERROR: 'Price' column not found!")
            print("Available columns:", list(self.df.columns))
            print("\n Tip: Make sure your CSV has a price column named:")
            print("   'Price', 'Close', 'Value', 'Gold_Price', 'XAUUSD', 'قیمت', 'طلا'")
            raise ValueError("Price column not found in data. Available columns: " + 
                           str(list(self.df.columns)))
        
        
        feature_columns = ['Price']
        
        
        indicator_priority = [
            'RSI_14', 'RSI_7',
            'MACD', 'MACD_signal', 'MACD_histogram',
            'MA_7', 'MA_30', 'MA_60',
            'BB_upper', 'BB_lower', 'BB_width',
            'Volatility_30', 'Volatility_7',
            'Returns_1d', 'Returns_5d',
            'Momentum_20', 'Momentum_5',
            'Price_Lag_1', 'Price_Lag_7',
            'Day_of_Week', 'Month'
        ]
        
        for col in indicator_priority:
            if col in self.df.columns and col not in feature_columns:
                feature_columns.append(col)
        
        
        if len(feature_columns) < 5:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col not in feature_columns and len(feature_columns) < 20:
                    feature_columns.append(col)
        
        self.feature_columns = feature_columns
        
        print(f"\n Preparing LSTM data:")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Features: {feature_columns[:10]}...")
        
        
        data = self.df[feature_columns].copy()
        data = data.dropna()
        
        if len(data) < sequence_length + 10:
            raise ValueError(f"Not enough data after cleaning! Only {len(data)} rows available. Need at least {sequence_length + 10}")
        
        
        scaled_data = self.scaler.fit_transform(data)
        
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
       
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f" Training data: {X_train.shape}")
        print(f" Testing data: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, self.scaler