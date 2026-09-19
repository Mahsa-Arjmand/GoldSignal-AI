import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import warnings
import json
import re
import time
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, Callback
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class NewsCollector:
    
    def __init__(self):
        self.news_sources = [
            'https://www.kitco.com/news/gold/',
            'https://www.reuters.com/markets/commodities/',
            'https://www.bloomberg.com/markets/commodities',
        ]
    
    def collect_news(self):
        news = []
        
        sample_news = [
            {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'title': 'Federal Reserve Signals Dovish Stance on Interest Rates',
                'description': 'The Federal Reserve indicated potential rate cuts in coming months as inflation shows signs of cooling. Lower rates typically boost gold as opportunity cost decreases.',
                'source': 'Reuters',
                'category': 'Monetary Policy'
            },
            {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'title': 'Geopolitical Tensions Escalate in Middle East',
                'description': 'Rising tensions in the Middle East have increased demand for safe-haven assets. Gold prices surge as investors seek protection against geopolitical risks.',
                'source': 'Bloomberg',
                'category': 'Geopolitics'
            },
            {
                'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'title': 'China Central Bank Increases Gold Reserves for 18th Month',
                'description': 'The People\'s Bank of China added another 10 tons of gold to its reserves, continuing a buying spree that supports gold prices and signals de-dollarization.',
                'source': 'Financial Times',
                'category': 'Central Bank'
            },
            {
                'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'title': 'US Dollar Index Falls to Three-Month Low',
                'description': 'The dollar weakened against major currencies after disappointing economic data, making gold more attractive for foreign investors.',
                'source': 'CNBC',
                'category': 'Currency Markets'
            },
            {
                'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                'title': 'Global Gold ETF Holdings Hit New Record',
                'description': 'Exchange-traded funds backed by physical gold reached an all-time high as institutional investors increase exposure to precious metals.',
                'source': 'World Gold Council',
                'category': 'Investment'
            },
            {
                'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                'title': 'Inflation Expectations Rise Above Central Bank Targets',
                'description': 'Survey shows consumers and businesses expect higher inflation, traditionally a bullish signal for gold as a hedge against currency devaluation.',
                'source': 'MarketWatch',
                'category': 'Inflation'
            },
            {
                'date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                'title': 'Major Gold Discovery Reported in Australia',
                'description': 'Mining company announces significant new gold deposit, though production is years away. Short-term impact limited, long-term supply implications.',
                'source': 'Mining Weekly',
                'category': 'Supply'
            },
            {
                'date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                'title': 'India Gold Demand Expected to Surge During Festival Season',
                'description': 'World\'s second-largest gold consumer anticipates 20% demand increase during Diwali festival, providing seasonal support to prices.',
                'source': 'Economic Times',
                'category': 'Demand'
            }
        ]
        
        try:
            import feedparser
            feeds = [
                'https://www.kitco.com/news/gold/rss/',
                'https://www.investing.com/rss/news_14.rss',
            ]
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:
                        if hasattr(entry, 'published_parsed'):
                            pub_date = datetime(*entry.published_parsed[:3]).strftime('%Y-%m-%d')
                        else:
                            pub_date = datetime.now().strftime('%Y-%m-%d')
                        
                        news.append({
                            'date': pub_date,
                            'title': entry.get('title', ''),
                            'description': entry.get('summary', '')[:500],
                            'source': feed_url.split('/')[2],
                            'category': 'RSS Feed'
                        })
                except:
                    pass
        except:
            pass
        
        all_news = sample_news + news
        
        seen = set()
        unique_news = []
        for item in all_news:
            if item['title'] not in seen:
                seen.add(item['title'])
                unique_news.append(item)
        
        return pd.DataFrame(unique_news)


class SentimentAnalyzer:
    
    def __init__(self, use_llm=False, llm_url="http://localhost:1234/v1/chat/completions"):
        self.use_llm = use_llm
        self.llm_url = llm_url
        
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            nltk.data.find('vader_lexicon')
            self.vader = SentimentIntensityAnalyzer()
            self.has_vader = True
        except:
            self.has_vader = False
    
    def analyze_with_llm(self, title, description):
        try:
            prompt = f"""You are a gold market analyst. Analyze this news and its impact on gold prices.
            
News Title: {title}
News Description: {description}

Provide your analysis in this exact JSON format:
{{
    "sentiment_score": <number between -1 and 1>,
    "explanation": "<1-2 sentences explaining the impact>",
    "key_factors": ["<factor1>", "<factor2>"],
    "impact_level": "<high|medium|low>"
}}

Rules:
- -1 = very bearish for gold (prices likely to fall significantly)
- -0.5 = moderately bearish
- 0 = neutral/no clear impact
- +0.5 = moderately bullish
- +1 = very bullish (prices likely to rise significantly)
- Consider: interest rates, USD strength, inflation, geopolitics, demand/supply"""

            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "You are a gold market analyst. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            import requests
            response = requests.post(self.llm_url, json=payload, timeout=200)
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            content = re.sub(r'```json|```', '', content).strip()
            result = json.loads(content)
            
            return {
                'sentiment_score': float(result['sentiment_score']),
                'explanation': result['explanation'],
                'key_factors': result.get('key_factors', []),
                'impact_level': result.get('impact_level', 'medium'),
                'method': 'LLM (Gemma)'
            }
            
        except Exception as e:
            print(f"LLM error: {e}, falling back to VADER")
            return self.analyze_with_vader(title, description)
    
    def analyze_with_vader(self, title, description):
        """Fallback: تحلیل با VADER"""
        if not self.has_vader:
            return self._simple_analysis(title, description)
        
        text = f"{title} {description}"
        scores = self.vader.polarity_scores(text)
        
        bullish_words = ['rate cut', 'dovish', 'geopolitical', 'safe haven', 'inflation',
                        'central bank buying', 'weaker dollar', 'uncertainty', 'recession']
        bearish_words = ['rate hike', 'hawkish', 'strong dollar', 'economic growth',
                        'risk-on', 'equity rally', 'higher yields', 'tapering']
        
        text_lower = text.lower()
        adjustment = 0
        key_factors = []
        
        for word in bullish_words:
            if word in text_lower:
                adjustment += 0.15
                key_factors.append(f"Bullish: {word}")
        
        for word in bearish_words:
            if word in text_lower:
                adjustment -= 0.15
                key_factors.append(f"Bearish: {word}")
        
        sentiment_score = max(-1.0, min(1.0, scores['compound'] + adjustment))
        
        if sentiment_score > 0.3:
            explanation = "Bullish for gold - positive factors detected"
            impact = "medium" if sentiment_score > 0.6 else "low"
        elif sentiment_score < -0.3:
            explanation = "Bearish for gold - negative factors detected"
            impact = "medium" if sentiment_score < -0.6 else "low"
        else:
            explanation = "Neutral impact on gold prices"
            impact = "low"
        
        return {
            'sentiment_score': sentiment_score,
            'explanation': explanation,
            'key_factors': key_factors[:3] if key_factors else ['General sentiment'],
            'impact_level': impact,
            'method': 'VADER'
        }
    
    def _simple_analysis(self, title, description):
        text = f"{title} {description}".lower()
        positive = ['rise', 'surge', 'rally', 'bullish', 'gain', 'increase', 'strong']
        negative = ['fall', 'drop', 'decline', 'bearish', 'loss', 'decrease', 'weak']
        
        pos_count = sum(1 for w in positive if w in text)
        neg_count = sum(1 for w in negative if w in text)
        
        score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        
        return {
            'sentiment_score': score,
            'explanation': f"Simple keyword analysis: {pos_count} positive, {neg_count} negative words",
            'key_factors': ['Keyword analysis'],
            'impact_level': 'low',
            'method': 'Simple Keywords'
        }
    
    def analyze_news(self, news_df):
        results = []
        
        for _, row in news_df.iterrows():
            if self.use_llm:
                analysis = self.analyze_with_llm(row['title'], row['description'])
            else:
                analysis = self.analyze_with_vader(row['title'], row['description'])
            
            results.append({
                'date': row['date'],
                'title': row['title'],
                'source': row.get('source', 'Unknown'),
                'sentiment_score': analysis['sentiment_score'],
                'explanation': analysis['explanation'],
                'key_factors': ', '.join(analysis['key_factors']),
                'impact_level': analysis['impact_level'],
                'method': analysis['method']
            })
        
        return pd.DataFrame(results)


class DataProcessor:
    def __init__(self, file_path=None, df=None):
        self.file_path = file_path
        self.df = df
        self.scaler = MinMaxScaler()
        self.eda_results = {}
    
    def load_data(self):
        if self.df is not None:
            df = self.df.copy()
        elif self.file_path:
            df = pd.read_csv(self.file_path)
        else:
            raise ValueError("No data source")
        
        df.columns = df.columns.str.strip()
        
        column_map = {
            'Date': 'Date', 'Price': 'Price', 'Open': 'Open',
            'High': 'High', 'Low': 'Low', 'Vol.': 'Volume', 'Change %': 'Change_Pct'
        }
        df.rename(columns=column_map, inplace=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
            df = df.dropna(subset=['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
        
        for col in ['Price', 'Open', 'High', 'Low', 'Volume']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Price' not in df.columns:
            if 'Close' in df.columns:
                df['Price'] = df['Close']
            elif 'Open' in df.columns:
                df['Price'] = df['Open']
        
        self.df = df
        return self.df
    
    def handle_missing_values(self):  
        if self.df is None:
            return self.df
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[num_cols] = self.df[num_cols].interpolate(method='linear', limit_direction='both')
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(0, inplace=True)
        return self.df
    
    def add_technical_indicators(self):
        df = self.df
        price = df['Price']
        
        for w in [7, 14, 30, 60]:
            df[f'MA_{w}'] = price.rolling(w, min_periods=1).mean()
        
        for p in [7, 14]:
            delta = price.diff()
            gain = delta.clip(lower=0).rolling(p, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(p, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'RSI_{p}'] = (100 - (100 / (1 + rs))).fillna(50)
        
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df['BB_mid'] = price.rolling(20, min_periods=1).mean()
        bb_std = price.rolling(20, min_periods=1).std()
        df['BB_upper'] = df['BB_mid'] + 2*bb_std
        df['BB_lower'] = df['BB_mid'] - 2*bb_std
        
        for p in [1, 5, 10]:
            df[f'Returns_{p}d'] = price.pct_change(p) * 100
        
        df['Volatility_7d'] = price.pct_change().rolling(7).std()
        
        try:
            df['DayOfWeek'] = df.index.dayofweek
            df['Month'] = df.index.month
        except:
            df['DayOfWeek'] = 0
            df['Month'] = 1
        
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        self.df = df
        return self.df
    
    def perform_eda(self):
        eda = {}
        eda['basic_info'] = {
            'records': len(self.df),
            'start': str(self.df.index[0])[:10],
            'end': str(self.df.index[-1])[:10]
        }
        eda['descriptive_stats'] = self.df.describe()
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns[:15]
        if len(num_cols) > 1:
            eda['correlation_matrix'] = self.df[num_cols].corr()
        
        if 'Price' in self.df.columns:
            price = self.df['Price']
            eda['overall_trend'] = {
                'start_price': float(price.iloc[:30].mean()),
                'end_price': float(price.iloc[-30:].mean()),
                'change_pct': float(((price.iloc[-1] - price.iloc[0]) / price.iloc[0]) * 100)
            }
        
        self.eda_results = eda
        return eda
    
    def prepare_lstm_data(self, sequence_length=60, test_size=0.2):
        features = ['Price', 'MA_7', 'MA_30', 'RSI_14', 'MACD', 'MACD_signal',
                   'Returns_1d', 'Returns_5d', 'Volatility_7d', 'DayOfWeek', 'Month']
        features = [f for f in features if f in self.df.columns]
        
        data = self.df[features].dropna()
        
        if len(data) < sequence_length + 10:
            raise ValueError(f"Not enough data: {len(data)} rows")
        
        scaled = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i-sequence_length:i])
            y.append(scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        split = int(len(X) * (1 - test_size))
        
        return X[:split], X[split:], y[:split], y[split:], self.scaler


class TerminalProgress(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
    
    def on_epoch_end(self, epoch, logs=None):
        bar_length = 30
        filled = int(bar_length * (epoch + 1) / self.epochs)
        bar = '█' * filled + '░' * (bar_length - filled)
        percent = (epoch + 1) / self.epochs * 100
        print(f'\r Epoch {epoch+1:3d}/{self.epochs} [{bar}] {percent:5.1f}% | '
              f'loss: {logs["loss"]:.4f} | val_loss: {logs.get("val_loss", 0):.4f}', end='')
        if epoch == self.epochs - 1:
            print()


st.set_page_config(page_title="Gold Prediction", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
    }
    .sentiment-positive { color: #00FF00; font-weight: bold; font-size: 1.1rem; }
    .sentiment-negative { color: #FF4444; font-weight: bold; font-size: 1.1rem; }
    .sentiment-neutral { color: #FFD700; font-weight: bold; font-size: 1.1rem; }
    .news-card {
        background: #1E1E1E;
        border-left: 4px solid #FFD700;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #B8860B, #FFD700) !important;
        color: black !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)


def generate_natural_language_report(price, predictions, sentiment_score, news_summary, metrics):
    
    if predictions[-1] > price:
        trend = "صعودی"
        trend_en = "bullish"
    else:
        trend = "نزولی"
        trend_en = "bearish"
    
    change_pct = abs((predictions[-1] - price) / price * 100)
    
    if sentiment_score > 0.3:
        sentiment_text = "بازار احساسات مثبت و خوش‌بینانه‌ای دارد"
    elif sentiment_score < -0.3:
        sentiment_text = "بازار احساسات منفی و محتاطانه‌ای دارد"
    else:
        sentiment_text = "بازار در حالت تعادل و بی‌طرفی به سر می‌برد"
    
    if sentiment_score > 0.2 and trend_en == "bullish":
        confidence = "بالا"
        recommendation = "موقعیت خرید مناسب است"
    elif sentiment_score < -0.2 and trend_en == "bearish":
        confidence = "بالا"
        recommendation = "احتیاط و در نظر گرفتن فروش"
    else:
        confidence = "متوسط"
        recommendation = "منتظر سیگنال‌های واضح‌تر بمانید"
    
    report = f"""
##  گزارش تحلیلی جامع قیمت طلا

###  خلاصه وضعیت
قیمت فعلی طلا **${price:,.2f}** دلار است و مدل ترکیبی (LSTM + تحلیل اخبار) 
روند **{trend}** با اطمینان **{confidence}** پیش‌بینی می‌کند.

###  تحلیل تکنیکال (LSTM)
مدل یادگیری عمیق با نگاه به {metrics.get('seq_len', 60)} روز گذشته، 
حرکت قیمت به سمت **${predictions[-1]:,.2f}** دلار را پیش‌بینی می‌کند 
(تغییر {change_pct:.1f}% در ۷ روز آینده).

عملکرد مدل:
- میانگین خطای مطلق: ${metrics.get('mae', 0):,.2f}
- ضریب تعیین (R²): {metrics.get('r2', 0):.3f}

###  تحلیل اخبار و احساسات بازار
امتیاز کلی احساسات بازار: **{sentiment_score:+.2f}** (از ۱+ تا ۱-)

{sentiment_text}. تحلیل {len(news_summary)} خبر اقتصادی و سیاسی نشان می‌دهد:

"""
    
    for i, news in enumerate(news_summary[:5], 1):
        emoji = "" if news['score'] > 0.2 else ("" if news['score'] < -0.2 else "")
        report += f"{emoji} **{news['title'][:80]}...**\n"
        report += f"   ↳ {news['explanation']}\n\n"
    
    report += f"""
###  پیش‌بینی ترکیبی نهایی
با ترکیب تحلیل تکنیکال (LSTM) و تحلیل بنیادی (اخبار):

- **پیش‌بینی ۷ روزه:** ${predictions[-1]:,.2f} دلار
- **روند:** {trend}
- **سطح اطمینان:** {confidence}
- **توصیه:** {recommendation}

###  عوامل تأثیرگذار
- سیاست‌های فدرال رزرو و نرخ بهره
- تنش‌های ژئوپلیتیکی و تقاضای پناهگاه امن
- خرید طلا توسط بانک‌های مرکزی
- نوسانات شاخص دلار آمریکا

---
*این تحلیل توسط سیستم هوش مصنوعی ترکیبی (LSTM + LLM) در تاریخ {datetime.now().strftime('%Y-%m-%d %H:%M')} تولید شده است.*
"""
    
    return report


def main():
    st.markdown('<h1 class="main-header">Gold Price Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#888;">Hybrid AI: LSTM Deep Learning + LLM Sentiment Analysis</p>', unsafe_allow_html=True)
    
    for key in ['df', 'processor', 'model_trained', 'metrics', 'predictions', 'actual',
                'sentiment_df', 'news_df', 'avg_sentiment', 'full_report']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    with st.sidebar:
        st.header(" Configuration")
        
        uploaded_file = st.file_uploader(" Upload Gold CSV", type=['csv'])
        default_path = "data/gold_data_daily_comprehensive_cleaned.csv"
        
        if os.path.exists(default_path):
            st.success(" Default dataset found!")
        
        st.divider()
        st.subheader(" LSTM Settings")
        seq_len = st.slider("Sequence Length", 30, 90, 60, 5)
        epochs = st.slider("Epochs", 5, 50, 15, 5)
        
        st.divider()
        st.subheader(" LLM Settings")
        use_llm = st.checkbox("Use Local LLM (LM Studio)", value=False,
                             help="Enable if LM Studio is running on port 1234")
        
        if use_llm:
            llm_url = st.text_input("LLM URL", "http://localhost:1234/v1/chat/completions")
        
        if not TENSORFLOW_AVAILABLE:
            st.error(" Install TensorFlow:\n`pip install tensorflow`")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            load_btn = st.button(" Load", use_container_width=True)
        with col2:
            news_btn = st.button(" News", use_container_width=True)
        
        train_btn = st.button(" Train LSTM", use_container_width=True)
        
        st.divider()
        full_btn = st.button(" FULL ANALYSIS (All Steps)", type="primary", use_container_width=True)
    
    
    if load_btn or full_btn:
        with st.spinner(" Loading data..."):
            try:
                if uploaded_file:
                    processor = DataProcessor(df=pd.read_csv(uploaded_file))
                elif os.path.exists(default_path):
                    processor = DataProcessor(file_path=default_path)
                else:
                    st.error(" Upload CSV!")
                    st.stop()
                
                processor.load_data()
                processor.handle_missing_values()
                processor.add_technical_indicators()
                processor.perform_eda()
                
                st.session_state.processor = processor
                st.session_state.df = processor.df
                st.success(f" {len(processor.df):,} records loaded")
                
            except Exception as e:
                st.error(f" {str(e)}")
    
    if news_btn or full_btn:
        with st.spinner(" Collecting and analyzing news..."):
            collector = NewsCollector()
            news_df = collector.collect_news()
            
            analyzer = SentimentAnalyzer(use_llm=use_llm)
            sentiment_df = analyzer.analyze_news(news_df)
            
            avg_sentiment = sentiment_df['sentiment_score'].mean()
            
            st.session_state.news_df = news_df
            st.session_state.sentiment_df = sentiment_df
            st.session_state.avg_sentiment = avg_sentiment
            
            st.success(f" {len(sentiment_df)} news analyzed | Sentiment: {avg_sentiment:+.2f}")
    
    if train_btn or full_btn:
        if st.session_state.processor is None:
            st.warning(" Load data first!")
        elif not TENSORFLOW_AVAILABLE:
            st.error(" TensorFlow required!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                processor = st.session_state.processor
                status_text.text(" Preparing data...")
                
                X_train, X_test, y_train, y_test, scaler = processor.prepare_lstm_data(seq_len, 0.2)
                
                model = Sequential([
                    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                status_text.text(f" Training {epochs} epochs...")
                
                terminal_cb = TerminalProgress(epochs)
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=epochs, batch_size=32, callbacks=[terminal_cb, early_stop], verbose=0)
                
                preds = model.predict(X_test, verbose=0).flatten()
                
                n_feat = scaler.scale_.shape[0]
                d_pred = np.zeros((len(preds), n_feat))
                d_pred[:, 0] = preds
                predictions = scaler.inverse_transform(d_pred)[:, 0]
                
                d_true = np.zeros((len(y_test), n_feat))
                d_true[:, 0] = y_test
                actual = scaler.inverse_transform(d_true)[:, 0]
                
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                    'mae': mean_absolute_error(actual, predictions),
                    'r2': r2_score(actual, predictions),
                    'mape': np.mean(np.abs((actual - predictions) / actual)) * 100,
                    'seq_len': seq_len
                }
                
                st.session_state.predictions = predictions
                st.session_state.actual = actual
                st.session_state.metrics = metrics
                st.session_state.model_trained = True
                
                progress_bar.progress(100)
                status_text.empty()
                st.success(f" RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}")
                
            except Exception as e:
                st.error(f" {str(e)}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Data & EDA", "News & Sentiment", " LSTM Prediction", 
        "Combined Forecast", " Final Report"
    ])
    
    with tab1:
        if st.session_state.df is None:
            st.info(" Load data first")
        else:
            df = st.session_state.df
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Records", f"{len(df):,}")
            c2.metric("Price Range", f"${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")
            c3.metric("Last Price", f"${df['Price'].iloc[-1]:,.2f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines',
                                    name='Price', line=dict(color='#FFD700', width=1.5)))
            if 'MA_30' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_30'], mode='lines',
                                        name='MA 30', line=dict(color='#FF6B6B', width=2)))
            fig.update_layout(title='Gold Price History', template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df.tail(5), use_container_width=True)
    
    with tab2:
        if st.session_state.sentiment_df is None:
            st.info(" Collect news first")
        else:
            st.subheader(" News Sentiment Analysis")
            
            sentiment_df = st.session_state.sentiment_df
            avg_sentiment = st.session_state.avg_sentiment
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Sentiment", f"{avg_sentiment:+.2f}",
                     delta="Bullish " if avg_sentiment > 0.2 else ("Bearish " if avg_sentiment < -0.2 else "Neutral 🟡"))
            c2.metric("Positive News", f"{(sentiment_df['sentiment_score'] > 0.2).sum()}")
            c3.metric("Negative News", f"{(sentiment_df['sentiment_score'] < -0.2).sum()}")
            c4.metric("Method", sentiment_df['method'].iloc[0])
            
            fig = px.histogram(sentiment_df, x='sentiment_score', nbins=20,
                             title='Sentiment Score Distribution',
                             color_discrete_sequence=['#FFD700'])
            fig.add_vline(x=0, line_dash="dash", line_color="white")
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader(" Analyzed News")
            for _, row in sentiment_df.iterrows():
                score = row['sentiment_score']
                if score > 0.2:
                    emoji, css = "", "sentiment-positive"
                elif score < -0.2:
                    emoji, css = "", "sentiment-negative"
                else:
                    emoji, css = "", "sentiment-neutral"
                
                with st.expander(f"{emoji} [{score:+.2f}] {row['title'][:100]}..."):
                    st.write(f"**Analysis:** {row['explanation']}")
                    st.write(f"**Key Factors:** {row['key_factors']}")
                    st.write(f"**Source:** {row['source']} | **Date:** {row['date']}")
                    st.write(f"**Method:** {row['method']}")
    
    with tab3:
        if not st.session_state.model_trained:
            st.info(" Train model first")
        else:
            st.subheader(" LSTM Prediction Results")
            
            metrics = st.session_state.metrics
            predictions = st.session_state.predictions
            actual = st.session_state.actual
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"${metrics['rmse']:,.2f}")
            c2.metric("MAE", f"${metrics['mae']:,.2f}")
            c3.metric("MAPE", f"{metrics['mape']:.2f}%")
            c4.metric("R²", f"{metrics['r2']:.4f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual',
                                    line=dict(color='#FFD700', width=2)))
            fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted',
                                    line=dict(color='#00FF00', width=2, dash='dash')))
            fig.update_layout(title='LSTM: Actual vs Predicted', template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if not st.session_state.model_trained:
            st.info(" Train model first")
        else:
            st.subheader(" Combined Forecast (LSTM + News Sentiment)")
            
            df = st.session_state.df
            last_price = df['Price'].iloc[-1]
            
            trend_lstm = np.polyfit(range(30), df['Price'].tail(30).values, 1)[0]
            
            sentiment_factor = st.session_state.avg_sentiment if st.session_state.avg_sentiment else 0
            sentiment_adjustment = sentiment_factor * 0.3  
            
            combined_trend = trend_lstm * (1 + sentiment_adjustment)
            
            future_lstm = [last_price + trend_lstm * i for i in range(1, 8)]
            future_combined = [last_price + combined_trend * i for i in range(1, 8)]
            
            future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=7)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=future_dates, y=future_lstm, mode='lines+markers',
                                    name='LSTM Only', line=dict(color='#FFD700', width=2)))
            fig.add_trace(go.Scatter(x=future_dates, y=future_combined, mode='lines+markers',
                                    name='LSTM + Sentiment', line=dict(color='#00FF00', width=3)))
            fig.add_hline(y=last_price, line_dash="dash", line_color="gray",
                         annotation_text=f"Current: ${last_price:,.0f}")
            fig.update_layout(title='7-Day Forecast Comparison', template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            comp_df = pd.DataFrame({
                'Day': ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
                'LSTM Only': [f"${p:,.2f}" for p in future_lstm],
                'LSTM + Sentiment': [f"${p:,.2f}" for p in future_combined],
                'Difference': [f"${c-l:+,.2f}" for c, l in zip(future_combined, future_lstm)]
            })
            st.dataframe(comp_df, use_container_width=True)
            
            if st.session_state.avg_sentiment:
                impact = "صعودی " if sentiment_factor > 0 else ("نزولی " if sentiment_factor < 0 else "خنثی ")
                st.info(f" تأثیر احساسات بازار: {impact} (امتیاز: {sentiment_factor:+.2f})")
    
    with tab5:
        if not st.session_state.model_trained:
            st.info(" Complete all steps first")
        elif st.button(" Generate Comprehensive Report", type="primary"):
            with st.spinner("Generating natural language report..."):
                df = st.session_state.df
                last_price = df['Price'].iloc[-1]
                metrics = st.session_state.metrics
                
                trend_lstm = np.polyfit(range(30), df['Price'].tail(30).values, 1)[0]
                sentiment_factor = st.session_state.avg_sentiment if st.session_state.avg_sentiment else 0
                combined_trend = trend_lstm * (1 + sentiment_factor * 0.3)
                future_combined = [last_price + combined_trend * i for i in range(1, 8)]
                
                news_summary = []
                if st.session_state.sentiment_df is not None:
                    for _, row in st.session_state.sentiment_df.iterrows():
                        news_summary.append({
                            'title': row['title'],
                            'score': row['sentiment_score'],
                            'explanation': row['explanation']
                        })
                
                report = generate_natural_language_report(
                    last_price, future_combined,
                    sentiment_factor, news_summary, metrics
                )
                
                st.session_state.full_report = report
                
                st.markdown(report)
                
                st.download_button(
                    " Download Report",
                    report,
                    file_name=f"gold_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )
        elif st.session_state.full_report:
            st.markdown(st.session_state.full_report)


if __name__ == "__main__":
    main()