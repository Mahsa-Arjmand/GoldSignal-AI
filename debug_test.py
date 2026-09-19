import sys
sys.path.insert(0, '.')

from src.data_loader import DataLoader
from src.news_sentiment import NewsSentimentAnalyzer
from src.feature_engineer import FeatureEngineer
from src.lstm_model import GoldLSTMPredictor
import numpy as np

print("1. بارگذاری داده...")
loader = DataLoader()
data = loader.load_all_data()
data = loader.clean_data()
data = loader.add_technical_indicators()

price_col = loader._find_price_column(data)
print(f"price_col: {price_col}")
print(f"data.shape: {data.shape}")
print(f"data columns: {data.columns.tolist()}")

print("\n2. تحلیل اخبار...")
news_analyzer = NewsSentimentAnalyzer()
news_df = news_analyzer.generate_news(data.index, len(data))
daily_sentiment = news_analyzer.aggregate_daily(news_df)
sentiment_aligned = news_analyzer.align_with_prices(daily_sentiment, data.index)

print("\n3. مهندسی ویژگی...")
feature_eng = FeatureEngineer(data, sentiment_aligned)
all_features = feature_eng.create_features(price_col)
selected_features = feature_eng.select_features(all_features, price_col, k=25)
scaled_features, scaler = feature_eng.scale_data(selected_features)

print(f"scaled_features.shape: {scaled_features.shape}")
print(f"n_features: {scaled_features.shape[1]}")

print("\n4. آموزش مدل...")
predictor = GoldLSTMPredictor(sequence_length=20)
model, results = predictor.train(scaled_features, price_col=price_col)
predictor.save_model()

print(f"model n_features: {predictor.n_features}")

print("\n5. پیش‌بینی...")
future = predictor.predict_future(scaled_features, days=7)
print(f"predictions: {future}")
