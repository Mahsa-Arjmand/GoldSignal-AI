import pandas as pd
import numpy as np
import requests
import json
import re
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:

    def __init__(self, news_api_key=None, use_llm=False):
        self.news_api_key = news_api_key
        self.use_llm = use_llm
        self.vader = SentimentIntensityAnalyzer()
        self.llm_endpoint = "http://localhost:1234/v1/chat/completions"
        self.news_cache = []
        
    
    def collect_news_from_sources(self, query="gold price", days=7):
        all_news = []
        
        if self.news_api_key:
            try:
                from newsapi import NewsApiClient
                newsapi = NewsApiClient(api_key=self.news_api_key)
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                response = newsapi.get_everything(
                    q=query,
                    from_param=from_date,
                    language='en',
                    sort_by='publishedAt',
                    page_size=100
                )
                
                for article in response['articles']:
                    all_news.append({
                        'date': article['publishedAt'][:10],
                        'title': article['title'],
                        'description': article['description'] or '',
                        'content': article.get('content', ''),
                        'source': article['source']['name'],
                        'url': article['url'],
                        'source_type': 'news_api'
                    })
                    
            except Exception as e:
                print(f"NewsAPI error: {e}")
        
        try:
            import feedparser
            feeds = [
                'https://finance.yahoo.com/rss/headline?s=GC=F',
                'https://www.kitco.com/news/gold/rss/',
            ]
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:10]:
                        if hasattr(entry, 'published_parsed'):
                            pub_date = datetime(*entry.published_parsed[:3]).strftime('%Y-%m-%d')
                        else:
                            pub_date = datetime.now().strftime('%Y-%m-%d')
                        
                        all_news.append({
                            'date': pub_date,
                            'title': entry.get('title', ''),
                            'description': entry.get('summary', '')[:500],
                            'content': '',
                            'source': feed_url.split('/')[2],
                            'url': entry.get('link', ''),
                            'source_type': 'rss'
                        })
                except:
                    pass
        except:
            pass
        
        if not all_news:
            all_news = self._get_sample_news(days)
        
        self.news_cache = all_news
        return pd.DataFrame(all_news)
    
    def _get_sample_news(self, days=7):
        base_date = datetime.now()
        sample_news = []
        
        templates = [
            {
                'title': 'Federal Reserve Signals Potential Rate Cuts',
                'description': 'Fed indicates it may cut rates in coming months. Lower rates typically boost gold by weakening USD and reducing opportunity cost.',
                'source': 'Reuters',
                'sentiment_bias': 0.7
            },
            {
                'title': 'Geopolitical Tensions Drive Safe-Haven Demand',
                'description': 'Rising global conflicts push investors toward gold. Central banks accelerate purchases amid uncertainty.',
                'source': 'Bloomberg',
                'sentiment_bias': 0.8
            },
            {
                'title': 'Strong Dollar Pressures Gold Prices Lower',
                'description': 'USD strengthens after robust economic data, making gold expensive for foreign buyers.',
                'source': 'CNBC',
                'sentiment_bias': -0.6
            },
            {
                'title': 'Central Banks Add 800 Tons of Gold to Reserves',
                'description': 'Record central bank buying supports gold demand. De-dollarization trend continues.',
                'source': 'World Gold Council',
                'sentiment_bias': 0.75
            },
            {
                'title': 'Gold ETF Holdings Rise for Fifth Week',
                'description': 'Investors increase gold exposure through ETFs as market uncertainty persists.',
                'source': 'ETF.com',
                'sentiment_bias': 0.5
            },
            {
                'title': 'Inflation Data Comes Hotter Than Expected',
                'description': 'CPI exceeds forecasts. While gold hedges inflation, rate cut expectations diminish.',
                'source': 'Financial Times',
                'sentiment_bias': 0.1
            },
            {
                'title': 'Mining Costs Rise, Supporting Gold Price Floor',
                'description': 'Production costs up 15% for major miners, creating higher support level for gold.',
                'source': 'Mining Weekly',
                'sentiment_bias': 0.4
            },
            {
                'title': 'India Gold Demand Surges During Festival Season',
                'description': 'World\'s second-largest consumer sees strong demand, providing seasonal price support.',
                'source': 'Economic Times',
                'sentiment_bias': 0.5
            },
            {
                'title': 'Crypto Rally Draws Investment from Gold',
                'description': 'Bitcoin reaches new highs, attracting speculative capital away from precious metals.',
                'source': 'CoinDesk',
                'sentiment_bias': -0.3
            },
            {
                'title': 'Bond Yields Rise, Creating Headwinds for Gold',
                'description': '10-year Treasury yield reaches 4.5%, increasing opportunity cost of holding gold.',
                'source': 'MarketWatch',
                'sentiment_bias': -0.5
            },
            {
                'title': 'BRICS Nations Launch Gold-Backed Digital Currency',
                'description': 'New gold-backed currency initiative could increase monetary demand for gold.',
                'source': 'RT',
                'sentiment_bias': 0.6
            },
            {
                'title': 'JP Morgan Raises Gold Price Target to $2,800',
                'description': 'Major bank upgrades forecast citing persistent inflation and geopolitical risks.',
                'source': 'Bloomberg',
                'sentiment_bias': 0.7
            }
        ]
        
        for i, template in enumerate(templates):
            news_date = base_date - timedelta(days=i % days)
            sample_news.append({
                'date': news_date.strftime('%Y-%m-%d'),
                'title': template['title'],
                'description': template['description'],
                'content': '',
                'source': template['source'],
                'url': '#',
                'source_type': 'sample',
                'sentiment_bias': template['sentiment_bias']
            })
        
        return sample_news
    
    
    def analyze_with_llm(self, text, model_name="gemma-4-e2b-instruct", temperature=0.1):

        try:
            system_prompt = """You are a professional gold market analyst. Analyze news for gold price impact.

Rules:
- Score from -1.0 (very bearish) to +1.0 (very bullish)
- Gold rises with: lower rates, weaker USD, inflation, geopolitics, central bank buying
- Gold falls with: higher rates, stronger USD, economic stability, rising bond yields

Respond ONLY with valid JSON:
{
    "sentiment_score": 0.0,
    "confidence": 0.8,
    "explanation": "brief 1-2 sentence analysis",
    "key_factors": ["factor1", "factor2"],
    "market_impact": "short_term",
    "suggested_action": "hold"
}"""

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this news:\n{text}"}
                ],
                "temperature": temperature,
                "max_tokens": 300
            }
            
            response = requests.post(
                self.llm_endpoint,
                json=payload,
                timeout=300,  
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            analysis = json.loads(content)
            
            if 'sentiment_score' not in analysis:
                raise ValueError("No sentiment_score in response")
            
            analysis['sentiment_score'] = max(-1.0, min(1.0, float(analysis['sentiment_score'])))
            analysis['confidence'] = float(analysis.get('confidence', 0.5))
            
            return analysis
            
        except requests.exceptions.ConnectionError:
            print(" LM Studio not running. Falling back to VADER.")
            return self._analyze_with_vader(text)
        except requests.exceptions.Timeout:
            print(" LM Studio timeout. Falling back to VADER.")
            return self._analyze_with_vader(text)
        except json.JSONDecodeError:
            print(f" Invalid JSON from LLM. Raw: {content[:100]}...")
            return self._analyze_with_vader(text)
        except Exception as e:
            print(f" LLM error: {str(e)[:100]}")
            return self._analyze_with_vader(text)
    
    
    def _analyze_with_vader(self, text):
        scores = self.vader.polarity_scores(str(text))
        compound = scores['compound']
        
        bullish_words = [
            'rate cut', 'dovish', 'geopolitical tension', 'safe haven',
            'inflation hedge', 'central bank buying', 'weaker dollar',
            'uncertainty', 'recession', 'stimulus', 'quantitative easing',
            'gold buying', 'etf inflow', 'demand surge', 'supply shortage'
        ]
        
        bearish_words = [
            'rate hike', 'hawkish', 'strong dollar', 'economic growth',
            'risk-on', 'equity rally', 'higher yields', 'tightening',
            'taper', 'peace talks', 'stability', 'crypto rally',
            'gold outflow', 'demand drop', 'oversupply'
        ]
        
        text_lower = text.lower()
        
        adjustment = 0
        for word in bullish_words:
            if word in text_lower:
                adjustment += 0.15
        for word in bearish_words:
            if word in text_lower:
                adjustment -= 0.15
        
        adjusted_score = max(-1.0, min(1.0, compound + adjustment))
        
        if adjusted_score > 0.3:
            explanation = "Bullish for gold - positive factors detected"
            action = "buy"
        elif adjusted_score < -0.3:
            explanation = "Bearish for gold - negative factors detected"
            action = "sell"
        else:
            explanation = "Neutral impact on gold prices"
            action = "hold"
        
        factors = []
        for word in bullish_words + bearish_words:
            if word in text_lower:
                factors.append(word)
        
        return {
            'sentiment_score': adjusted_score,
            'confidence': abs(scores['compound']),
            'explanation': explanation,
            'key_factors': factors[:3] if factors else ['general market sentiment'],
            'market_impact': 'short_term',
            'suggested_action': action
        }
    
    
    def analyze_with_textblob(self, text):
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return {
            'sentiment_score': polarity,
            'confidence': 1 - subjectivity,
            'explanation': f"TextBlob polarity: {polarity:.2f}",
            'key_factors': ['text-based sentiment'],
            'market_impact': 'short_term',
            'suggested_action': 'buy' if polarity > 0.2 else ('sell' if polarity < -0.2 else 'hold')
        }
    
    
    def analyze_news_batch(self, news_df, method='auto'):

        if news_df.empty:
            return pd.DataFrame()
        
        if method == 'auto':
            method = 'llm' if self.use_llm else 'vader'
        
        results = []
        total = len(news_df)
        
        for idx, (_, row) in enumerate(news_df.iterrows(), 1):
            text = f"Title: {row['title']}\nDescription: {row.get('description', '')}"
            
            print(f"\n Analyzing {idx}/{total}: {row['title'][:60]}...")
            
            if method == 'llm':
                analysis = self.analyze_with_llm(text)
            elif method == 'vader':
                analysis = self._analyze_with_vader(text)
            elif method == 'textblob':
                analysis = self.analyze_with_textblob(text)
            elif method == 'ensemble':
                analysis = self._ensemble_analysis(text)
            else:
                analysis = self._analyze_with_vader(text)
            
            score = analysis['sentiment_score']
            print(f"   Score: {score:+.2f} | Method: {method}")
            
            results.append({
                'date': row['date'],
                'title': row['title'],
                'source': row.get('source', 'Unknown'),
                'sentiment_score': score,
                'confidence': analysis.get('confidence', 0.5),
                'explanation': analysis['explanation'],
                'key_factors': ', '.join(analysis.get('key_factors', [])),
                'market_impact': analysis.get('market_impact', 'short_term'),
                'suggested_action': analysis.get('suggested_action', 'hold'),
                'url': row.get('url', '#'),
                'method': method
            })
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df['date'] = pd.to_datetime(result_df['date'])
        
        return result_df
    
    def _ensemble_analysis(self, text):
        vader_result = self._analyze_with_vader(text)
        textblob_result = self.analyze_with_textblob(text)
        
        weights = {'vader': 0.4, 'textblob': 0.2, 'llm': 0.4}
        
        try:
            llm_result = self.analyze_with_llm(text)
            ensemble_score = (
                llm_result['sentiment_score'] * weights['llm'] +
                vader_result['sentiment_score'] * weights['vader'] +
                textblob_result['sentiment_score'] * weights['textblob']
            )
            return {**llm_result, 'sentiment_score': ensemble_score}
        except:
            ensemble_score = (
                vader_result['sentiment_score'] * 0.7 +
                textblob_result['sentiment_score'] * 0.3
            )
            return {**vader_result, 'sentiment_score': ensemble_score}
    
    
    def get_aggregate_sentiment(self, sentiment_df, window='1D'):
        if sentiment_df.empty:
            return pd.DataFrame()
        
        df = sentiment_df.copy()
        df.set_index('date', inplace=True)
        
        daily = df['sentiment_score'].resample(window).agg(['mean', 'std', 'count', 'min', 'max'])
        daily.columns = ['avg_sentiment', 'sentiment_volatility', 'news_count', 'min_sentiment', 'max_sentiment']
        
        df['is_positive'] = df['sentiment_score'] > 0.2
        daily['positive_ratio'] = df['is_positive'].resample(window).mean()
        
        return daily.fillna(0)


if __name__ == "__main__":
    print(" Testing Sentiment Analyzer...")
    print("="*50)
    
    analyzer = SentimentAnalyzer(use_llm=False)  
    
    test_text = "Federal Reserve signals rate cuts amid economic uncertainty, boosting gold demand"
    
    print("\n Test News:", test_text)
    
    result = analyzer._analyze_with_vader(test_text)
    print(f"\n VADER Result:")
    print(f"   Score: {result['sentiment_score']:+.2f}")
    print(f"   Explanation: {result['explanation']}")
    print(f"   Action: {result['suggested_action']}")
    
    print("\n Collecting sample news...")
    news_df = analyzer._get_sample_news(days=3)
    print(f"   Found {len(news_df)} news items")
    
    print("\n Analyzing batch...")
    results_df = analyzer.analyze_news_batch(news_df[:5], method='vader')
    print(f"\n Analysis complete: {len(results_df)} results")
    print(f"   Avg Sentiment: {results_df['sentiment_score'].mean():+.2f}")
    
    print("\n" + "="*50)
    print(" Test completed successfully!")