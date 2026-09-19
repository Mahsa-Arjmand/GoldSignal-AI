# GoldSignal-AI

A hybrid AI-powered gold price prediction system combining LSTM deep learning with news sentiment analysis for accurate gold market forecasting.

## Features

- **LSTM Deep Learning**: Advanced neural network model for technical analysis using historical price data
- **News Sentiment Analysis**: Analyzes financial news and market sentiment to complement technical predictions
- **LLM Integration**: Optional integration with local LLMs (e.g., LM Studio) for enhanced sentiment analysis
- **Technical Indicators**: Comprehensive set of indicators including:
  - Moving Averages (MA 5, 7, 10, 14, 20, 30, 50, 60)
  - Exponential Moving Averages (EMA 12, 26)
  - Relative Strength Index (RSI 7, 14)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Volatility measures
  - Momentum indicators
- **Interactive Dashboard**: Streamlit-based web interface for easy data visualization and analysis
- **Combined Forecasting**: Merges technical predictions with sentiment analysis for more accurate forecasts
- **Natural Language Reports**: Generates comprehensive market analysis reports

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Mahsa-Arjmand/GoldSignal-AI.git
cd GoldSignal-AI
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For enhanced sentiment analysis, set up a local LLM (e.g., LM Studio) running on port 1234

## Usage

### Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Load Data**: Upload a CSV file with gold price data or use the default dataset
2. **Configure Settings**: Adjust LSTM parameters (sequence length, epochs) and LLM settings
3. **Load Data**: Click "Load" to process the data and generate technical indicators
4. **Collect News**: Click "News" to fetch and analyze market news
5. **Train Model**: Click "Train LSTM" to train the neural network
6. **Full Analysis**: Click "FULL ANALYSIS" to run all steps automatically

### Dashboard Tabs

- **Data & EDA**: View data statistics, price history, and technical indicators
- **News & Sentiment**: Analyze news sentiment and market factors
- **LSTM Prediction**: View model performance and predictions
- **Combined Forecast**: Compare LSTM-only vs LSTM+Sentiment forecasts
- **Final Report**: Generate and download comprehensive market analysis reports

## Data Format

The system expects CSV files with the following columns:
- `Date`: Date of the price data
- `Price`: Gold price (or alternative names: Close, Value, Gold_Price, XAUUSD)
- `Open`: Opening price (optional)
- `High`: Highest price (optional)
- `Low`: Lowest price (optional)
- `Volume`: Trading volume (optional)

## Project Structure

```
GoldSignal-AI/
├── app.py                      # Main Streamlit application
├── main.py                     # Data inspection script
├── requirements.txt            # Python dependencies
├── reset.py                    # Reset utility
├── debug_test.py              # Debugging utilities
├── data/                       # Data directory
│   ├── gold_data_daily_comprehensive_cleaned.csv
│   ├── Gold Futures Historical Data (6).csv
│   └── Gold Futures Historical Data 1.csv
└── utils/                      # Utility modules
    ├── __init__.py
    ├── data_processor.py       # Data processing and feature engineering
    ├── model_trainer.py        # LSTM model training and evaluation
    ├── news_collector.py       # News collection from RSS feeds
    └── sentiment_analyzer.py   # Sentiment analysis (VADER/LLM)
```

## Model Architecture

The LSTM model uses the following architecture:
- Input layer: Accepts sequences of historical price data and technical indicators
- LSTM layers: Multiple LSTM layers with dropout for regularization
- Dense layers: Fully connected layers for final prediction
- Output: Single value representing predicted gold price

## Sentiment Analysis

The system supports multiple sentiment analysis methods:
1. **VADER**: Rule-based sentiment analysis (default)
2. **LLM Integration**: Enhanced analysis using local LLMs (Gemma, etc.)
3. **Keyword Analysis**: Simple keyword-based fallback

## Technical Indicators

The system automatically calculates and uses:
- Moving Averages (MA)
- Exponential Moving Averages (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volatility measures
- Price returns and momentum
- Time-based features (day of week, month, quarter)

## Performance Metrics

The model is evaluated using:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (R-squared)
- Direction Accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with TensorFlow/Keras for deep learning
- Streamlit for the web interface
- Scikit-learn for data processing
- NLTK/VADER for sentiment analysis
- Plotly for interactive visualizations

## Disclaimer

This tool is for educational and research purposes only. It should not be used for actual trading decisions. Always consult with financial advisors before making investment decisions.

## Contact

For questions or suggestions, please open an issue on GitHub or contact the project maintainer.
