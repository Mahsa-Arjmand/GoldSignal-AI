import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import re
import time
from typing import List, Dict, Optional
import feedparser
import warnings
warnings.filterwarnings('ignore')

class NewsCollector:
    
    def __init__(self, news_api_key: Optional[str] = None):
 
        self.news_api_key = news_api_key
        self.news_data = []
        self.sources = []
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        self.gold_keywords = [
            'gold price', 'gold market', 'gold trading', 'xauusd',
            'precious metals', 'gold forecast', 'gold analysis',
            'gold investment', 'gold demand', 'gold supply',
            'central bank gold', 'gold reserves', 'gold etf',
            'gold mining', 'gold futures', 'gold spot',
            'طلا', 'سکه', 'قیمت طلا', 'بازار طلا'
        ]
        
        self.economic_keywords = [
            'federal reserve', 'interest rate', 'inflation',
            'monetary policy', 'dollar index', 'geopolitical',
            'economic data', 'cpi', 'gdp', 'employment',
            'recession', 'stimulus', 'tapering', 'quantitative easing'
        ]
    
    def collect_from_newsapi(self, query: str = "gold price", days: int = 7) -> List[Dict]:

        if not self.news_api_key:
            print(" No NewsAPI key provided. Skipping NewsAPI source.")
            return []
        
        news_list = []
        
        try:
            from newsapi import NewsApiClient
            
            newsapi = NewsApiClient(api_key=self.news_api_key)
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            queries = [query] + self.gold_keywords[:3] + self.economic_keywords[:2]
            
            for q in queries[:5]:  
                try:
                    response = newsapi.get_everything(
                        q=q,
                        from_param=from_date,
                        to=to_date,
                        language='en',
                        sort_by='relevancy',
                        page_size=100
                    )
                    
                    if response['status'] == 'ok':
                        for article in response['articles']:
                            if not any(n.get('url') == article['url'] for n in news_list):
                                news_list.append({
                                    'date': article['publishedAt'][:10] if article['publishedAt'] else from_date,
                                    'time': article['publishedAt'][11:19] if article['publishedAt'] else '00:00:00',
                                    'title': article['title'] or '',
                                    'description': article['description'] or '',
                                    'content': article.get('content', '') or '',
                                    'source': article['source']['name'] if article.get('source') else 'Unknown',
                                    'url': article['url'] or '',
                                    'image_url': article.get('urlToImage', ''),
                                    'category': 'gold_economy',
                                    'source_type': 'news_api'
                                })
                    
                    time.sleep(0.5)  
                    
                except Exception as e:
                    print(f" Error fetching query '{q}': {e}")
                    continue
            
            print(f" Collected {len(news_list)} articles from NewsAPI")
            
        except ImportError:
            print(" newsapi-python not installed. Install with: pip install newsapi-python")
        except Exception as e:
            print(f" NewsAPI error: {e}")
        
        return news_list
    
    def collect_from_rss_feeds(self) -> List[Dict]:
   
        news_list = []
        
        rss_feeds = [
            {
                'url': 'https://feeds.feedburner.com/KitcoNews',
                'source': 'Kitco',
                'category': 'gold_news'
            },
            {
                'url': 'https://www.investing.com/rss/news_14.rss',
                'source': 'Investing.com',
                'category': 'commodities'
            },
            {
                'url': 'https://www.fxstreet.com/rss/news',
                'source': 'FXStreet',
                'category': 'forex_gold'
            },
            {
                'url': 'https://www.reuters.com/agency/feed/topNews',
                'source': 'Reuters',
                'category': 'top_news'
            },
            {
                'url': 'https://feeds.bloomberg.com/markets/news.rss',
                'source': 'Bloomberg',
                'category': 'markets'
            },
            {
                'url': 'https://www.marketwatch.com/feeds/marketwatch/marketpulse',
                'source': 'MarketWatch',
                'category': 'markets'
            },
            {
                'url': 'https://www.cnbc.com/id/10001147/device/rss/rss.html',
                'source': 'CNBC',
                'category': 'top_news'
            },
            {
                'url': 'https://www.gold.org/feed',
                'source': 'World Gold Council',
                'category': 'gold_industry'
            },
            {
                'url': 'https://www.bullionvault.com/gold-news/rss',
                'source': 'BullionVault',
                'category': 'gold_news'
            }
        ]
        
        for feed_info in rss_feeds:
            try:
                feed = feedparser.parse(feed_info['url'])
                
                if feed.entries:
                    for entry in feed.entries[:20]:  
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        else:
                            pub_date = datetime.now()
                        
                        title = entry.get('title', '')
                        summary = entry.get('summary', '')
                        content = f"{title} {summary}"
                        
                        is_relevant = any(
                            keyword.lower() in content.lower()
                            for keyword in self.gold_keywords + self.economic_keywords
                        )
                        
                        if is_relevant or feed_info['category'] in ['gold_news', 'gold_industry']:
                            news_list.append({
                                'date': pub_date.strftime('%Y-%m-%d'),
                                'time': pub_date.strftime('%H:%M:%S'),
                                'title': title,
                                'description': self._clean_html(summary)[:500],
                                'content': self._clean_html(getattr(entry, 'content', [{}])[0].get('value', '')) if hasattr(entry, 'content') else '',
                                'source': feed_info['source'],
                                'url': entry.get('link', ''),
                                'category': feed_info['category'],
                                'source_type': 'rss_feed'
                            })
                
                time.sleep(0.3)  
            except Exception as e:
                print(f" Error fetching RSS from {feed_info['source']}: {e}")
                continue
        
        print(f" Collected {len(news_list)} articles from RSS feeds")
        return news_list
    
    def collect_from_yahoo_finance(self, symbol: str = "GC=F", days: int = 7) -> List[Dict]:

        news_list = []
        
        try:
            import yfinance as yf
            
            gold = yf.Ticker(symbol)
            
            news = gold.news
            
            if news:
                for article in news[:50]:
                    pub_time = datetime.fromtimestamp(article.get('providerPublishTime', time.time()))
                    
                    if (datetime.now() - pub_time).days <= days:
                        news_list.append({
                            'date': pub_time.strftime('%Y-%m-%d'),
                            'time': pub_time.strftime('%H:%M:%S'),
                            'title': article.get('title', ''),
                            'description': article.get('summary', '')[:500] if article.get('summary') else '',
                            'content': '',
                            'source': article.get('publisher', 'Yahoo Finance'),
                            'url': article.get('link', ''),
                            'category': 'gold_market',
                            'source_type': 'yahoo_finance'
                        })
            
            print(f"Collected {len(news_list)} articles from Yahoo Finance")
            
        except ImportError:
            print(" yfinance not installed. Install with: pip install yfinance")
        except Exception as e:
            print(f" Yahoo Finance error: {e}")
        
        return news_list
    
    def collect_from_twitter_api(self, query: str = "gold price", count: int = 100) -> List[Dict]:

        print(" Twitter API requires authentication. Skipping for now.")
        return []
    
    def collect_from_reddit(self, subreddit: str = "gold", limit: int = 100) -> List[Dict]:

        news_list = []
        
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
            response = requests.get(url, headers=self.headers, timeout=200)
            response.raise_for_status()
            
            data = response.json()
            
            for post in data['data']['children']:
                post_data = post['data']
                created_time = datetime.fromtimestamp(post_data['created_utc'])
                
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                
                news_list.append({
                    'date': created_time.strftime('%Y-%m-%d'),
                    'time': created_time.strftime('%H:%M:%S'),
                    'title': title,
                    'description': selftext[:500] if selftext else title,
                    'content': selftext,
                    'source': f"Reddit r/{subreddit}",
                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                    'category': 'social_media',
                    'source_type': 'reddit',
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0)
                })
            
            print(f" Collected {len(news_list)} posts from Reddit r/{subreddit}")
            
        except Exception as e:
            print(f" Reddit error: {e}")
        
        return news_list
    
    def collect_custom_api(self, url: str, api_key: str = None, params: Dict = None) -> List[Dict]:

        news_list = []
        
        try:
            headers = self.headers.copy()
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                for item in data:
                    news_list.append(self._parse_api_item(item))
            elif isinstance(data, dict):
                items = data.get('articles') or data.get('results') or data.get('data') or []
                for item in items:
                    news_list.append(self._parse_api_item(item))
            
            print(f" Collected {len(news_list)} articles from custom API")
            
        except Exception as e:
            print(f" Custom API error: {e}")
        
        return news_list
    
    def _parse_api_item(self, item: Dict) -> Dict:
        return {
            'date': item.get('date') or item.get('publishedAt') or item.get('created_at', datetime.now().strftime('%Y-%m-%d')),
            'title': item.get('title') or item.get('headline', ''),
            'description': item.get('description') or item.get('summary', ''),
            'content': item.get('content') or item.get('body', ''),
            'source': item.get('source') or item.get('publisher', 'Unknown'),
            'url': item.get('url') or item.get('link', ''),
            'category': item.get('category', 'general'),
            'source_type': 'custom_api'
        }
    
    def collect_all_news(self, days: int = 7) -> pd.DataFrame:
 
        all_news = []
        
        print(" Collecting news from all available sources...")
        
        if self.news_api_key:
            news_api = self.collect_from_newsapi(days=days)
            all_news.extend(news_api)
        
        rss_news = self.collect_from_rss_feeds()
        all_news.extend(rss_news)
        
        yahoo_news = self.collect_from_yahoo_finance(days=days)
        all_news.extend(yahoo_news)
        
        reddit_news = self.collect_from_reddit(limit=50)
        all_news.extend(reddit_news)
        
        unique_news = []
        seen_urls = set()
        
        for news in all_news:
            url = news.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_news.append(news)
            elif not url:
                title = news.get('title', '')
                if title not in [n.get('title', '') for n in unique_news]:
                    unique_news.append(news)
        
        unique_news.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        self.news_data = unique_news
        
        df = pd.DataFrame(unique_news)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            
            df = df[df['title'].str.len() > 0]
            
            df = df.sort_values('date', ascending=False)
            
            print(f"Total unique news collected: {len(df)}")
            print(f" Sources: {df['source_type'].value_counts().to_dict()}")
        
        return df
    
    def filter_relevant_news(self, df: pd.DataFrame, min_relevance: float = 0.3) -> pd.DataFrame:

        if df.empty:
            return df
        
        def calculate_relevance(row):
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            
            score = 0
            max_score = 0
            
            for keyword in self.gold_keywords:
                max_score += 1
                if keyword.lower() in text:
                    score += 1
            
            for keyword in self.economic_keywords:
                max_score += 0.5
                if keyword.lower() in text:
                    score += 0.5
            
            return score / max_score if max_score > 0 else 0
        
        df['relevance_score'] = df.apply(calculate_relevance, axis=1)
        filtered_df = df[df['relevance_score'] >= min_relevance].copy()
        
        print(f" Filtered {len(filtered_df)} relevant news from {len(df)} total")
        
        return filtered_df
    
    def get_sample_news(self, days: int = 7, count: int = 20) -> pd.DataFrame:
   
        sample_templates = [
            {
                'title': 'Federal Reserve Signals Potential Rate Cuts',
                'description': 'Fed Chair suggests monetary policy easing could begin in coming months as inflation shows signs of cooling. This dovish stance typically supports gold prices.',
                'source': 'Reuters',
                'category': 'monetary_policy',
                'sentiment_bias': 0.7
            },
            {
                'title': 'Geopolitical Tensions Escalate, Gold Surges as Safe Haven',
                'description': 'Rising conflicts in multiple regions drive investors toward gold. Central banks accelerate gold purchases amid uncertainty.',
                'source': 'Bloomberg',
                'category': 'geopolitics',
                'sentiment_bias': 0.8
            },
            {
                'title': 'Dollar Strengthens on Strong Jobs Data, Gold Under Pressure',
                'description': 'US employment report beats expectations, strengthening the dollar index. Gold prices fall as dollar-denominated assets become more expensive.',
                'source': 'CNBC',
                'category': 'economic_data',
                'sentiment_bias': -0.6
            },
            {
                'title': 'Central Banks Add 800 Tons of Gold to Reserves in 2024',
                'description': 'World Gold Council reports record central bank buying. Diversification away from dollar continues to support gold demand.',
                'source': 'World Gold Council',
                'category': 'central_bank',
                'sentiment_bias': 0.75
            },
            {
                'title': 'Gold ETF Holdings Rise for Fifth Consecutive Week',
                'description': 'Investors continue to add gold exposure through ETFs. Total holdings reach new monthly high as market uncertainty persists.',
                'source': 'ETF.com',
                'category': 'investment',
                'sentiment_bias': 0.5
            },
            {
                'title': 'Inflation Data Comes in Hotter Than Expected',
                'description': 'CPI rises 3.5% year-over-year, exceeding forecasts. While gold historically benefits from inflation, immediate reaction is muted as rate cut expectations diminish.',
                'source': 'Financial Times',
                'category': 'inflation',
                'sentiment_bias': 0.1
            },
            {
                'title': 'Mining Production Costs Rise, Supporting Gold Floor',
                'description': 'Major gold miners report 15% increase in production costs. Higher cost base provides support for gold prices above $2000.',
                'source': 'Mining Weekly',
                'category': 'supply',
                'sentiment_bias': 0.4
            },
            {
                'title': 'India Gold Demand Expected to Surge 20% During Festivals',
                'description': 'World\'s second-largest gold consumer sees strong demand ahead of Diwali and wedding season. Physical buying provides price support.',
                'source': 'Economic Times',
                'category': 'demand',
                'sentiment_bias': 0.5
            },
            {
                'title': 'Cryptocurrency Rally Draws Some Investment from Gold',
                'description': 'Bitcoin reaches new all-time high, attracting speculative capital. Some investors rotate from gold to crypto assets.',
                'source': 'CoinDesk',
                'category': 'competing_assets',
                'sentiment_bias': -0.3
            },
            {
                'title': 'Technical Analysis: Gold Approaches Key Resistance at $2,500',
                'description': 'Gold prices testing major resistance level. Break above could trigger rally to new highs, while rejection may lead to consolidation.',
                'source': 'Investing.com',
                'category': 'technical',
                'sentiment_bias': 0.2
            },
            {
                'title': 'Bond Yields Rise, Creating Headwinds for Gold',
                'description': '10-year Treasury yield reaches 4.5%, increasing opportunity cost of holding non-yielding gold. Investment demand softens.',
                'source': 'MarketWatch',
                'category': 'bond_market',
                'sentiment_bias': -0.5
            },
            {
                'title': 'China\'s Economic Slowdown Weighs on Commodity Demand',
                'description': 'Chinese GDP growth misses expectations. Reduced industrial demand and consumer spending could impact gold jewelry sales.',
                'source': 'Reuters',
                'category': 'global_economy',
                'sentiment_bias': -0.2
            },
            {
                'title': 'Gold-Backed Digital Currency Launched by BRICS Nations',
                'description': 'BRICS alliance announces gold-backed digital currency initiative. Move could increase gold\'s monetary role and demand.',
                'source': 'RT',
                'category': 'innovation',
                'sentiment_bias': 0.6
            },
            {
                'title': 'Environmental Regulations Impact Gold Mining Operations',
                'description': 'New environmental rules increase compliance costs for miners. Supply constraints could support higher gold prices.',
                'source': 'Environmental Finance',
                'category': 'regulation',
                'sentiment_bias': 0.3
            },
            {
                'title': 'JP Morgan Raises Gold Price Target to $2,800',
                'description': 'Major investment bank upgrades gold forecast citing persistent inflation and geopolitical risks. Bullish outlook from institutional investors.',
                'source': 'Bloomberg',
                'category': 'analyst_view',
                'sentiment_bias': 0.7
            }
        ]
        
        samples = []
        base_date = datetime.now()
        
        for i, template in enumerate(sample_templates[:count]):
            news_date = base_date - timedelta(days=i % days)
            samples.append({
                'date': news_date.strftime('%Y-%m-%d'),
                'time': f'{np.random.randint(6, 22):02d}:{np.random.randint(0, 60):02d}:00',
                'title': template['title'],
                'description': template['description'],
                'content': template['description'],
                'source': template['source'],
                'url': f'https://example.com/news/{i}',
                'category': template['category'],
                'source_type': 'sample',
                'sentiment_bias': template['sentiment_bias']
            })
        
        return pd.DataFrame(samples)
    
    def _clean_html(self, text: str) -> str:
        if not text:
            return ''
        
        clean = re.compile('<.*?>')
        text = re.sub(clean, ' ', text)
        
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&nbsp;', ' ')
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save_news_to_csv(self, df: pd.DataFrame, filename: str = 'gold_news.csv'):

        if df.empty:
            print(" No news to save.")
            return
        
        filepath = f"data/{filename}"
        os.makedirs('data', exist_ok=True)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f" News saved to {filepath}")
    
    def load_news_from_csv(self, filename: str = 'gold_news.csv') -> pd.DataFrame:
   
        filepath = f"data/{filename}"
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            print(f" Loaded {len(df)} news from {filepath}")
            return df
        else:
            print(f" File not found: {filepath}")
            return pd.DataFrame()


def main():
    collector = NewsCollector()
    
    print("Testing News Collector...")
    print("-" * 50)
    
    sample_news = collector.get_sample_news(days=7, count=10)
    print(f"\n Sample News:")
    print(sample_news[['date', 'title', 'source', 'sentiment_bias']].to_string())
    
    filtered = collector.filter_relevant_news(sample_news, min_relevance=0.2)
    print(f"\n Filtered: {len(filtered)} relevant news")
    
    return sample_news


if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    test_news = main()
    print("\n News collector test completed!")