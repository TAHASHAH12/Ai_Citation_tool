import streamlit as st
from openai import OpenAI
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict
import time
import json
from datetime import datetime, timedelta
import traceback
import random
from urllib.parse import urlparse

# Try to import tldextract for proper domain extraction
try:
    import tldextract
    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False

# DataForSEO imports
try:
    from dataforseo_client import configuration as dfs_config, api_client as dfs_api_provider
    from dataforseo_client.api.serp_api import SerpApi
    from dataforseo_client.api.keywords_data_api import KeywordsDataApi
    from dataforseo_client.api.backlinks_api import BacklinksApi
    from dataforseo_client.models.serp_google_organic_live_advanced_request_info import SerpGoogleOrganicLiveAdvancedRequestInfo
    from dataforseo_client.models.keywords_data_google_ads_search_volume_live_request_info import KeywordsDataGoogleAdsSearchVolumeLiveRequestInfo
    from dataforseo_client.models.backlinks_domain_pages_live_request_info import BacklinksDomainPagesLiveRequestInfo
    DATAFORSEO_AVAILABLE = True
except ImportError:
    DATAFORSEO_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Citation Tracker Pro - xFunnel Style",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for xFunnel.ai-style dashboard
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    flex: 1;
    min-width: 200px;
    background: white;
    border: 1px solid #e1e5e9;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    text-align: center;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2d3748;
    margin: 0;
}

.metric-label {
    font-size: 0.875rem;
    color: #718096;
    margin-top: 0.5rem;
    font-weight: 500;
}

.citation-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: all 0.3s ease;
}

.citation-card:hover {
    background: #edf2f7;
    border-color: #667eea;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
}

.query-info {
    background: #e6fffa;
    border: 1px solid #81e6d9;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #234e52;
}

.domain-preview {
    background: #f0fff4;
    border: 1px solid #68d391;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #22543d;
}

.sentiment-positive {
    background: linear-gradient(135deg, #68d391 0%, #38a169 100%);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
}

.sentiment-negative {
    background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
}

.sentiment-neutral {
    background: linear-gradient(135deg, #fbb6ce 0%, #d69e2e 100%);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
}

.citation-detail {
    background: #ffffff;
    border: 2px solid #667eea;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
}

.trend-container {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
@st.cache_resource
def init_openai_client():
    """Initialize OpenAI client with proper error handling"""
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        client.models.list()
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        return None

@st.cache_resource
def init_dataforseo_config():
    """Initialize DataForSEO configuration"""
    if not DATAFORSEO_AVAILABLE:
        return None
    try:
        config = dfs_config.Configuration(
            username=st.secrets["DATAFORSEO_LOGIN"],
            password=st.secrets["DATAFORSEO_PASSWORD"]
        )
        return config
    except Exception as e:
        st.warning(f"DataForSEO configuration failed: {str(e)}")
        return None

# Enhanced Competitor Discovery
def discover_competitors_ai(client, brand, industry_description=""):
    """Discover competitors using AI"""
    prompt = f"""You are a competitive intelligence expert. Given the brand "{brand}" in the {industry_description} industry, identify 5 direct competitors.

Requirements:
- Focus on direct competitors in the same industry/niche
- Include both established players and emerging competitors
- Consider market position, target audience, and business model
- Provide only domain names (e.g., competitor.com)

Brand: {brand}
Industry: {industry_description}

Output format: Return only 5 domain names, one per line, no explanations."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a competitive intelligence expert. Provide accurate, real competitor domain names."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        competitors = [comp.strip().lower() 
                      for comp in response.choices[0].message.content.strip().split('\n') 
                      if comp.strip() and '.' in comp.strip()]
        
        return competitors[:5]
    
    except Exception as e:
        st.error(f"Error discovering competitors: {str(e)}")
        return []

# xFunnel Buyer Journey Stages
XFUNNEL_STAGES = {
    "Problem Unaware": {
        "description": "Users don't know they have a problem",
        "query_templates": [
            "What are the latest trends in {industry}?",
            "How is {industry} evolving?",
            "What should I know about {industry}?"
        ]
    },
    "Problem Aware": {
        "description": "Users recognize they have a problem",
        "query_templates": [
            "What problems exist with {problem_area}?",
            "Why do people struggle with {problem_area}?",
            "What are common {problem_area} challenges?"
        ]
    },
    "Solution Aware": {
        "description": "Users know solutions exist",
        "query_templates": [
            "What solutions exist for {problem_area}?",
            "How to solve {problem_area}?",
            "Best ways to handle {problem_area}?"
        ]
    },
    "Product Aware": {
        "description": "Users know about specific products",
        "query_templates": [
            "What is {product_category}?",
            "How does {product_category} work?",
            "Benefits of {product_category}?"
        ]
    },
    "Most Aware": {
        "description": "Users ready to buy",
        "query_templates": [
            "Best {product_category} platforms",
            "{product_category} comparison",
            "Top {product_category} providers"
        ]
    }
}

def generate_xfunnel_queries(client, brand, industry, problem_area, product_category, stage, num_queries=5):
    """Generate specified number of queries for a specific xFunnel stage"""
    stage_info = XFUNNEL_STAGES[stage]
    templates = stage_info["query_templates"]
    
    prompt = f"""Generate {num_queries} diverse, realistic search queries for the "{stage}" stage of the buyer journey.

Context:
- Brand: {brand}
- Industry: {industry}
- Problem Area: {problem_area}
- Product Category: {product_category}
- Stage Description: {stage_info["description"]}

Use these template patterns but make them natural and varied:
{chr(10).join(templates)}

Requirements:
- Make queries sound like real user searches
- Vary the question formats and intent
- Focus on the {stage} mindset
- Include different angles and perspectives
- Generate exactly {num_queries} unique queries

Output only {num_queries} queries, one per line."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a search behavior expert. Generate realistic user queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=min(600, num_queries * 50),
            temperature=0.8
        )
        
        queries = [q.strip() for q in response.choices[0].message.content.strip().split('\n') 
                  if q.strip() and len(q.strip()) > 10]
        
        return queries[:num_queries]
    
    except Exception as e:
        st.error(f"Error generating queries for {stage}: {str(e)}")
        return []

def simulate_comprehensive_ai_response(client, query, platform, stage):
    """Generate comprehensive AI responses with natural citations"""
    prompt = f"""You are {platform}, responding to a user query at the "{stage}" stage of their buyer journey.

Query: {query}
Buyer Journey Stage: {stage}

Provide a comprehensive, helpful response that:
- Addresses the user's specific stage in the buyer journey
- Naturally includes relevant website citations and sources
- Mentions specific companies, platforms, and domains when relevant
- Uses a conversational but informative tone
- Includes actionable information appropriate for this stage
- Cites authoritative sources and industry leaders

Be thorough and include multiple relevant sources naturally in your response."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are {platform}, a knowledgeable AI assistant providing comprehensive answers with natural citations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Error generating response for {platform}: {str(e)}")
        return ""

def is_valid_domain(domain_string):
    """Check if a string is a valid domain using tldextract"""
    if not TLDEXTRACT_AVAILABLE:
        # Fallback basic validation if tldextract not available
        if not domain_string or len(domain_string) < 4:
            return False
        if any(char in domain_string for char in [' ', '(', ')', '[', ']', '<', '>', '"', "'"]):
            return False
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,}$', domain_string):
            return False
        return True
    
    try:
        if not domain_string or len(domain_string) < 4:
            return False
        
        if any(char in domain_string for char in [' ', '(', ')', '[', ']', '<', '>', '"', "'"]):
            return False
        
        extracted = tldextract.extract(domain_string)
        
        if not extracted.domain or not extracted.suffix:
            return False
        
        valid_tlds = ['com', 'org', 'net', 'edu', 'gov', 'co.uk', 'co', 'io', 'ai', 'tech', 'casino', 'bet', 'app', 'info', 'biz', 'tv', 'me', 'ly']
        if extracted.suffix not in valid_tlds:
            return False
        
        return True
    except:
        return False

def extract_valid_domain(text_or_url):
    """Extract valid domain from URL or text using tldextract"""
    if not TLDEXTRACT_AVAILABLE:
        try:
            if text_or_url.startswith(('http://', 'https://')):
                parsed = urlparse(text_or_url)
                domain = parsed.netloc.lower()
            else:
                domain = text_or_url.lower().strip()
            
            if is_valid_domain(domain):
                return domain
            return None
        except:
            return None
    
    try:
        extracted = tldextract.extract(text_or_url)
        
        if extracted.domain and extracted.suffix:
            if extracted.subdomain and extracted.subdomain != 'www':
                return f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}"
            else:
                return f"{extracted.domain}.{extracted.suffix}"
        
        return None
    except:
        return None

def extract_citations_enhanced_fixed(response_text):
    """FIXED: Enhanced citation extraction with proper domain validation"""
    citations = []
    
    url_patterns = [
        r'https?://[^\s\)\],;"<>\n]+',
        r'www\.[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/[^\s\)\],;"<>]*)?'
    ]
    
    known_brands = [
        'Amazon', 'Google', 'Facebook', 'Meta', 'Twitter', 'X', 'LinkedIn', 'YouTube', 
        'Instagram', 'TikTok', 'Netflix', 'Spotify', 'Apple', 'Microsoft', 'Tesla', 
        'Uber', 'Airbnb', 'eBay', 'PayPal', 'Shopify', 'Zoom', 'Slack', 'Discord', 
        'Reddit', 'Pinterest', 'Snapchat', 'WhatsApp', 'Telegram', 'Dropbox', 
        'GitHub', 'Stack Overflow', 'Wikipedia', 'BBC', 'CNN', 'Forbes', 'TechCrunch', 
        'Wired', 'The Verge', 'Mashable', 'Engadget', 'Ars Technica', 'CoinDesk', 
        'CoinTelegraph', 'Binance', 'Coinbase', 'Kraken', 'FTX', 'Crypto.com', 
        'Stake.com', 'Bet365', 'DraftKings', 'FanDuel', 'BetFair', 'William Hill',
        'Trustpilot', 'AskGamblers', 'Casino Guru', 'Sportsbookreview'
    ]
    
    citation_rank = 1
    found_citations = set()
    
    # Extract complete URLs first
    for pattern in url_patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            url = match.group().rstrip('.,;:!?)')
            domain = extract_valid_domain(url)
            
            if domain and domain not in found_citations:
                start = max(0, match.start() - 150)
                end = min(len(response_text), match.end() + 150)
                context = response_text[start:end].strip()
                
                citations.append({
                    'citation_text': url,
                    'domain': domain,
                    'type': 'URL',
                    'context': context,
                    'position': match.start(),
                    'citation_rank': citation_rank
                })
                found_citations.add(domain)
                citation_rank += 1
    
    # Extract known brand mentions
    for brand in known_brands:
        pattern = rf'\b{re.escape(brand)}(?:\.com|\'s website|\'s platform|\'s site|\s+website|\s+platform|\s+site)?\b'
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        
        for match in matches:
            citation_text = match.group().strip()
            brand_name = brand.lower().replace(' ', '').replace('.', '')
            if brand_name == 'x':
                domain = 'x.com'
            elif brand_name == 'meta':
                domain = 'meta.com'
            elif '.' in brand:
                domain = brand.lower()
            else:
                domain = f"{brand_name}.com"
            
            if domain not in found_citations:
                start = max(0, match.start() - 150)
                end = min(len(response_text), match.end() + 150)
                context = response_text[start:end].strip()
                
                citations.append({
                    'citation_text': citation_text,
                    'domain': domain,
                    'type': 'Brand Mention',
                    'context': context,
                    'position': match.start(),
                    'citation_rank': citation_rank
                })
                found_citations.add(domain)
                citation_rank += 1
    
    # Look for explicit domain mentions
    domain_pattern = r'\b[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\b'
    matches = re.finditer(domain_pattern, response_text, re.IGNORECASE)
    
    for match in matches:
        potential_domain = match.group().strip().lower()
        
        if is_valid_domain(potential_domain) and potential_domain not in found_citations:
            if not any(word in potential_domain for word in ['etc', 'com.', '.etc', 'example']):
                start = max(0, match.start() - 150)
                end = min(len(response_text), match.end() + 150)
                context = response_text[start:end].strip()
                
                citations.append({
                    'citation_text': potential_domain,
                    'domain': potential_domain,
                    'type': 'Domain Mention',
                    'context': context,
                    'position': match.start(),
                    'citation_rank': citation_rank
                })
                found_citations.add(potential_domain)
                citation_rank += 1
    
    citations.sort(key=lambda x: x['position'])
    for i, citation in enumerate(citations, 1):
        citation['citation_rank'] = i
    
    return citations

def classify_sentiment(client, context, citation):
    """Classify sentiment using OpenAI"""
    prompt = f"""Analyze the sentiment towards "{citation}" in this context:

{context}

Classify as: positive, neutral, or negative

Consider:
- How the brand/website is mentioned
- Whether it's recommended or criticized
- The overall tone and context

Respond with only one word: positive, neutral, or negative"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Classify sentiment accurately."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        
        sentiment = response.choices[0].message.content.strip().lower()
        
        if 'positive' in sentiment:
            return 'Positive'
        elif 'negative' in sentiment:
            return 'Negative'
        else:
            return 'Neutral'
    
    except Exception as e:
        return 'Neutral'

def get_domain_authority_score(domain, config):
    """Get domain authority score (enhanced with more realistic scoring)"""
    domain_scores = {
        'google.com': 100, 'youtube.com': 98, 'facebook.com': 96, 'amazon.com': 95,
        'wikipedia.org': 94, 'twitter.com': 93, 'x.com': 93, 'linkedin.com': 92, 
        'instagram.com': 90, 'reddit.com': 85, 'github.com': 82, 'stackoverflow.com': 80, 
        'medium.com': 75, 'forbes.com': 88, 'cnn.com': 87, 'bbc.com': 89, 'techcrunch.com': 82, 
        'wired.com': 78, 'theverge.com': 76, 'mashable.com': 74, 'engadget.com': 72,
        'arstechnica.com': 73, 'coindesk.com': 77, 'cointelegraph.com': 75, 'binance.com': 80, 
        'coinbase.com': 79, 'kraken.com': 72, 'crypto.com': 70, 'stake.com': 65, 
        'bet365.com': 70, 'draftkings.com': 68, 'fanduel.com': 67, 'betfair.com': 65, 
        'williamhill.com': 64, 'trustpilot.com': 78, 'askgamblers.com': 60, 'casino.guru': 58, 
        'sportsbookreview.com': 62, 'casino.org': 65
    }
    
    if domain in domain_scores:
        return domain_scores[domain]
    elif domain.endswith('.edu'):
        return random.randint(70, 90)
    elif domain.endswith('.gov'):
        return random.randint(80, 95)
    elif domain.endswith('.org'):
        return random.randint(40, 80)
    else:
        return random.randint(20, 70)

def generate_trend_data(df, brand, competitors):
    """Generate trend data for visualization"""
    if df.empty:
        return pd.DataFrame()
    
    # Create time series data over the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    trend_data = []
    all_brands = [brand] + competitors
    
    for i in range(31):  # 30 days + today
        current_date = start_date + timedelta(days=i)
        
        for brand_name in all_brands:
            brand_df = df[df['domain'].str.contains(brand_name, case=False, na=False)]
            
            if not brand_df.empty:
                # Base metrics with some daily variation
                base_mentions = len(brand_df)
                daily_variation = random.uniform(0.7, 1.3)
                daily_mentions = max(0, int(base_mentions * daily_variation * random.uniform(0.1, 0.4)))
                
                # Sentiment metrics
                sentiment_counts = brand_df['sentiment'].value_counts().to_dict()
                total_sentiment = sum(sentiment_counts.values()) if sentiment_counts else 1
                
                sentiment_score = 0
                if total_sentiment > 0:
                    pos = sentiment_counts.get('Positive', 0)
                    neg = sentiment_counts.get('Negative', 0)
                    sentiment_score = ((pos - neg) / total_sentiment) * 100
                
                # Add some trend variation
                sentiment_variation = random.uniform(-10, 10)
                daily_sentiment = max(-100, min(100, sentiment_score + sentiment_variation))
                
                visibility = (daily_mentions / max(1, len(df))) * 100
                avg_rank = brand_df['citation_rank'].mean() if daily_mentions > 0 else 0
                
                trend_data.append({
                    'date': current_date,
                    'brand': brand_name,
                    'mentions': daily_mentions,
                    'sentiment_score': daily_sentiment,
                    'visibility': visibility,
                    'avg_rank': avg_rank + random.uniform(-0.5, 0.5) if avg_rank > 0 else 0
                })
    
    return pd.DataFrame(trend_data)

def calculate_xfunnel_metrics(df, brand, competitors):
    """Calculate comprehensive xFunnel-style metrics"""
    if df.empty:
        return {}
    
    metrics = {}
    all_brands = [brand] + competitors
    
    total_queries = df['query'].nunique()
    total_responses = len(df)
    
    for brand_name in all_brands:
        brand_df = df[df['domain'].str.contains(brand_name, case=False, na=False)]
        
        if brand_df.empty:
            metrics[brand_name] = {
                'mentions': 0, 'visibility': 0, 'avg_rank': 0, 'sentiment_score': 0,
                'stage_performance': {}, 'platform_performance': {},
                'share_of_voice': 0, 'feature_familiarity': 0, 'sentiment_distribution': {}
            }
            continue
        
        # Calculate stage performance
        stage_performance = {}
        for stage in XFUNNEL_STAGES.keys():
            stage_df = brand_df[brand_df['stage'] == stage]
            stage_performance[stage] = {
                'mentions': len(stage_df),
                'visibility': (len(stage_df) / len(df[df['stage'] == stage])) * 100 if len(df[df['stage'] == stage]) > 0 else 0
            }
        
        # Calculate platform performance
        platform_performance = {}
        for platform in brand_df['platform'].unique():
            platform_df = brand_df[brand_df['platform'] == platform]
            platform_performance[platform] = {
                'mentions': len(platform_df),
                'avg_rank': platform_df['citation_rank'].mean()
            }
        
        # Sentiment analysis
        sentiment_counts = brand_df['sentiment'].value_counts().to_dict()
        sentiment_score = 0
        if len(brand_df) > 0:
            pos = sentiment_counts.get('Positive', 0)
            neg = sentiment_counts.get('Negative', 0)
            sentiment_score = ((pos - neg) / len(brand_df)) * 100
        
        # Share of voice calculation
        total_mentions = len(df)
        share_of_voice = (len(brand_df) / total_mentions) * 100 if total_mentions > 0 else 0
        
        # Feature familiarity (depth of mentions in bottom funnel)
        bottom_funnel_df = brand_df[brand_df['stage'].isin(['Product Aware', 'Most Aware'])]
        feature_familiarity = len(bottom_funnel_df) / len(brand_df) * 100 if len(brand_df) > 0 else 0
        
        metrics[brand_name] = {
            'mentions': len(brand_df),
            'visibility': (len(brand_df) / total_queries) * 100 if total_queries > 0 else 0,
            'avg_rank': brand_df['citation_rank'].mean(),
            'sentiment_score': sentiment_score,
            'stage_performance': stage_performance,
            'platform_performance': platform_performance,
            'share_of_voice': share_of_voice,
            'feature_familiarity': feature_familiarity,
            'sentiment_distribution': sentiment_counts
        }
    
    return metrics

# Enhanced Dashboard Components
def create_main_dashboard_overview(total_queries, total_responses, personas, industries, regions):
    """Create main dashboard overview with metric cards"""
    st.markdown("## Main Dashboard - Search Engine Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Unique Queries</div>
        </div>
        """.format(total_queries), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">AI Responses</div>
        </div>
        """.format(total_responses), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Personas Analyzed</div>
        </div>
        """.format(personas), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Industry Verticals</div>
        </div>
        """.format(industries), unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Geographic Regions</div>
        </div>
        """.format(regions), unsafe_allow_html=True)

def create_enhanced_sentiment_analysis(df, metrics):
    """Create comprehensive sentiment analysis with separate charts"""
    if df.empty:
        st.info("No sentiment data available")
        return
    
    st.markdown("## 💭 Advanced Sentiment Analysis")
    
    # Overall sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall sentiment pie chart
        sentiment_counts = df['sentiment'].value_counts()
        colors = {'Positive': '#48bb78', 'Neutral': '#ed8936', 'Negative': '#f56565'}
        
        fig_overall = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=.6,
            marker_colors=[colors.get(sentiment, '#a0aec0') for sentiment in sentiment_counts.index],
            textinfo="label+percent",
            textposition="outside"
        )])
        
        fig_overall.update_layout(
            title="Overall Sentiment Distribution",
            height=400,
            showlegend=True
        )
        
        fig_overall.add_annotation(
            text=f"Total<br>Citations<br>{len(df)}",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )
        
        st.plotly_chart(fig_overall, use_container_width=True)
    
    with col2:
        # Sentiment by platform
        platform_sentiment = df.groupby(['platform', 'sentiment']).size().reset_index(name='count')
        
        if not platform_sentiment.empty:
            fig_platform_sentiment = px.bar(
                platform_sentiment,
                x='platform',
                y='count',
                color='sentiment',
                title="Sentiment Distribution by AI Platform",
                color_discrete_map=colors
            )
            fig_platform_sentiment.update_layout(height=400)
            st.plotly_chart(fig_platform_sentiment, use_container_width=True)
    
    # Brand-specific sentiment analysis
    brand_sentiment_data = []
    for brand_name, brand_data in metrics.items():
        sentiment_dist = brand_data.get('sentiment_distribution', {})
        for sentiment, count in sentiment_dist.items():
            brand_sentiment_data.append({
                'Brand': brand_name,
                'Sentiment': sentiment,
                'Count': count,
                'Percentage': (count / sum(sentiment_dist.values())) * 100 if sentiment_dist else 0
            })
    
    if brand_sentiment_data:
        sentiment_df = pd.DataFrame(brand_sentiment_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stacked bar chart for brand sentiment
            fig_brand_sentiment = px.bar(
                sentiment_df,
                x='Brand',
                y='Count',
                color='Sentiment',
                title="Sentiment Distribution by Brand",
                color_discrete_map=colors
            )
            fig_brand_sentiment.update_layout(height=400)
            st.plotly_chart(fig_brand_sentiment, use_container_width=True)
        
        with col2:
            # Sentiment score comparison
            sentiment_scores = []
            for brand_name, brand_data in metrics.items():
                if brand_data.get('mentions', 0) > 0:
                    sentiment_scores.append({
                        'Brand': brand_name,
                        'Sentiment Score': brand_data.get('sentiment_score', 0),
                        'Mentions': brand_data.get('mentions', 0)
                    })
            
            if sentiment_scores:
                scores_df = pd.DataFrame(sentiment_scores)
                fig_scores = px.bar(
                    scores_df,
                    x='Brand',
                    y='Sentiment Score',
                    title="Brand Sentiment Scores",
                    color='Sentiment Score',
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                fig_scores.update_layout(height=400)
                st.plotly_chart(fig_scores, use_container_width=True)

def create_trend_analysis(df, brand, competitors):
    """Create comprehensive trend analysis with line graphs"""
    if df.empty:
        st.info("No data available for trend analysis")
        return
    
    st.markdown("## 📈 Trend Analysis")
    
    # Generate trend data
    trend_df = generate_trend_data(df, brand, competitors)
    
    if trend_df.empty:
        st.info("No trend data available")
        return
    
    # Create trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Mentions trend
        fig_mentions = px.line(
            trend_df,
            x='date',
            y='mentions',
            color='brand',
            title="📊 Brand Mentions Trend (30 Days)",
            markers=True
        )
        fig_mentions.update_layout(
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_mentions.update_traces(line=dict(width=3))
        st.plotly_chart(fig_mentions, use_container_width=True)
    
    with col2:
        # Sentiment trend
        fig_sentiment = px.line(
            trend_df,
            x='date',
            y='sentiment_score',
            color='brand',
            title="💭 Sentiment Score Trend (30 Days)",
            markers=True
        )
        fig_sentiment.update_layout(
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_sentiment.update_traces(line=dict(width=3))
        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Visibility and ranking trends
    col1, col2 = st.columns(2)
    
    with col1:
        # Visibility trend
        fig_visibility = px.area(
            trend_df,
            x='date',
            y='visibility',
            color='brand',
            title="👁️ Brand Visibility Trend (30 Days)"
        )
        fig_visibility.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_visibility, use_container_width=True)
    
    with col2:
        # Average ranking trend
        fig_ranking = px.line(
            trend_df[trend_df['avg_rank'] > 0],  # Only show brands with rankings
            x='date',
            y='avg_rank',
            color='brand',
            title="🎯 Average Citation Rank Trend (30 Days)",
            markers=True
        )
        fig_ranking.update_layout(
            height=400,
            yaxis=dict(autorange="reversed"),  # Lower rank is better
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_ranking.update_traces(line=dict(width=3))
        st.plotly_chart(fig_ranking, use_container_width=True)

def create_query_citation_details(df):
    """Create detailed query citation explorer with clickable interface"""
    if df.empty:
        st.info("No citation data available")
        return
    
    st.markdown("## 🔍 Query Citation Explorer")
    st.markdown("Click on any query below to see detailed citations and AI responses")
    
    # Group by query
    query_groups = df.groupby('query').agg({
        'citation_text': 'count',
        'sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Neutral',
        'platform': lambda x: ', '.join(x.unique()),
        'stage': 'first',
        'domain': lambda x: ', '.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else '')
    }).reset_index()
    
    query_groups.columns = ['Query', 'Citations', 'Dominant Sentiment', 'Platforms', 'Stage', 'Top Domains']
    
    # Create expandable query sections
    for _, row in query_groups.iterrows():
        query = row['Query']
        citations_count = row['Citations']
        sentiment = row['Dominant Sentiment']
        
        # Sentiment color coding
        sentiment_class = f"sentiment-{sentiment.lower()}"
        
        with st.expander(f"🔍 **{query}** ({citations_count} citations)", expanded=False):
            # Query details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Stage:** {row['Stage']}")
                st.markdown(f"**Platforms:** {row['Platforms']}")
            
            with col2:
                st.markdown(f"**Citations:** {citations_count}")
                st.markdown(f"""**Sentiment:** <span class="{sentiment_class}">{sentiment}</span>""", 
                           unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**Top Domains:** {row['Top Domains']}")
            
            st.markdown("---")
            
            # Get all data for this query
            query_data = df[df['query'] == query]
            
            # Group by platform to show AI responses
            for platform in query_data['platform'].unique():
                platform_data = query_data[query_data['platform'] == platform]
                
                if not platform_data.empty:
                    st.markdown(f"### 🤖 {platform} Response")
                    
                    # Show AI response
                    ai_response = platform_data.iloc[0]['ai_response']
                    st.markdown(f"""
                    <div class="citation-detail">
                        <strong>AI Response:</strong><br>
                        {ai_response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show extracted citations
                    st.markdown("**📋 Extracted Citations:**")
                    
                    for _, citation in platform_data.iterrows():
                        sentiment_class = f"sentiment-{citation['sentiment'].lower()}"
                        
                        st.markdown(f"""
                        <div class="citation-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <strong>#{citation['citation_rank']} - {citation['citation_text']}</strong>
                                <span class="{sentiment_class}">{citation['sentiment']}</span>
                            </div>
                            <div style="margin-bottom: 0.5rem;">
                                <strong>Domain:</strong> {citation['domain']} | 
                                <strong>Type:</strong> {citation['type']} | 
                                <strong>DA:</strong> {citation['domain_authority']}
                            </div>
                            <div style="font-style: italic; color: #666;">
                                <strong>Context:</strong> {citation['context'][:200]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")

def create_buyer_journey_metrics(metrics, brand):
    """Create buyer journey metrics visualization"""
    st.markdown("## 🛒 Buying Journey Metrics")
    
    brand_metrics = metrics.get(brand, {})
    
    # Current snapshot metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        visibility = brand_metrics.get('visibility', 0)
        st.metric("Brand Visibility", f"{visibility:.1f}%", "Top-funnel")
    
    with col2:
        avg_rank = brand_metrics.get('avg_rank', 0)
        st.metric("Rank Score", f"{avg_rank:.1f}", "Mid-funnel")
    
    with col3:
        feature_familiarity = brand_metrics.get('feature_familiarity', 0)
        st.metric("Feature Familiarity", f"{feature_familiarity:.1f}%", "Bottom-funnel")
    
    with col4:
        sentiment = brand_metrics.get('sentiment_score', 0)
        st.metric("Sentiment Score", f"{sentiment:.1f}", "Overall tone")
    
    # Buyer journey stage performance
    stage_performance = brand_metrics.get('stage_performance', {})
    if stage_performance:
        stage_data = []
        for stage, data in stage_performance.items():
            stage_data.append({
                'Stage': stage,
                'Mentions': data['mentions'],
                'Visibility': data['visibility']
            })
        
        if stage_data:
            stage_df = pd.DataFrame(stage_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_funnel = go.Figure(go.Funnel(
                    y=stage_df['Stage'],
                    x=stage_df['Mentions'],
                    textinfo="value+percent initial",
                    marker=dict(color=["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe"])
                ))
                fig_funnel.update_layout(title="Buyer Journey Funnel", height=400)
                st.plotly_chart(fig_funnel, use_container_width=True)
            
            with col2:
                fig_visibility = px.bar(
                    stage_df,
                    x='Stage',
                    y='Visibility',
                    title="Visibility by Buyer Journey Stage",
                    color='Visibility',
                    color_continuous_scale="Viridis"
                )
                fig_visibility.update_layout(height=400)
                st.plotly_chart(fig_visibility, use_container_width=True)

def create_citation_analysis_overview(df):
    """Create citation analysis overview"""
    st.markdown("## 📊 Citation Analysis Overview")
    
    if df.empty:
        st.info("No citation data available")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Citations", len(df))
    
    with col2:
        st.metric("Unique Sources", df['domain'].nunique())
    
    with col3:
        st.metric("Companies Mentioned", df['domain'].nunique())
    
    with col4:
        avg_da = df['domain_authority'].mean() if 'domain_authority' in df.columns else 0
        st.metric("Avg Domain Authority", f"{avg_da:.0f}")
    
    # Visualizations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Buyer Journey Phases Donut
        if 'stage' in df.columns:
            stage_counts = df['stage'].value_counts()
            fig_stages = go.Figure(data=[go.Pie(
                labels=stage_counts.index,
                values=stage_counts.values,
                hole=.5,
                title="Buyer Journey Phases"
            )])
            fig_stages.update_layout(height=300)
            st.plotly_chart(fig_stages, use_container_width=True)
    
    with col2:
        # Citation Sources Donut
        source_counts = df['type'].value_counts()
        fig_sources = go.Figure(data=[go.Pie(
            labels=source_counts.index,
            values=source_counts.values,
            hole=.5,
            title="Citation Sources"
        )])
        fig_sources.update_layout(height=300)
        st.plotly_chart(fig_sources, use_container_width=True)
    
    with col3:
        # Answer Engine Donut
        if 'platform' in df.columns:
            platform_counts = df['platform'].value_counts()
            fig_platforms = go.Figure(data=[go.Pie(
                labels=platform_counts.index,
                values=platform_counts.values,
                hole=.5,
                title="Answer Engines"
            )])
            fig_platforms.update_layout(height=300)
            st.plotly_chart(fig_platforms, use_container_width=True)
    
    # Top Citation Breakdown Table
    st.markdown("### 📋 Top Citation Breakdown")
    
    if not df.empty:
        citation_breakdown = []
        for _, row in df.head(20).iterrows():
            citation_breakdown.append({
                'Query': row['query'][:50] + '...' if len(row['query']) > 50 else row['query'],
                'Platform': row.get('platform', 'Unknown'),
                'Domain Authority': row.get('domain_authority', 0),
                'Companies Mentioned': row['domain'],
                'Stage': row.get('stage', 'Unknown'),
                'Rank': row['citation_rank'],
                'Sentiment': row.get('sentiment', 'Unknown')
            })
        
        breakdown_df = pd.DataFrame(citation_breakdown)
        st.dataframe(breakdown_df, use_container_width=True, height=400)

# Main Application
def main():
    st.markdown('<h1 class="main-header">AI Citation Tracker Pro - xFunnel Enhanced</h1>', unsafe_allow_html=True)
    st.markdown("**Complete xFunnel.ai replica with sentiment analysis, trend charts, and detailed query exploration**")
    
    # Check for tldextract availability
    if not TLDEXTRACT_AVAILABLE:
        st.warning("⚠️ For best results, install tldextract: `pip install tldextract`. Using basic domain validation.")
    
    # Initialize clients
    openai_client = init_openai_client()
    dataforseo_config = init_dataforseo_config()
    
    if not openai_client:
        st.error("OpenAI client initialization failed. Please check your API key.")
        return
    
    # Initialize session state
    for key in ['analysis_complete', 'results_df', 'current_metrics', 'competitors']:
        if key not in st.session_state:
            if key == 'analysis_complete':
                st.session_state[key] = False
            elif 'df' in key:
                st.session_state[key] = pd.DataFrame()
            else:
                st.session_state[key] = {} if key != 'competitors' else []
    
    # Enhanced Sidebar Configuration
    with st.sidebar:
        st.header("🎯 xFunnel Configuration")
        
        # Brand Input and Competitor Discovery
        with st.expander("🏢 Brand & Competitor Discovery", expanded=True):
            brand = st.text_input(
                "Your Brand Domain",
                "stake.com",
                help="Enter your main brand domain"
            ).strip().lower()
            
            industry_description = st.text_input(
                "Industry Description",
                "crypto casino and online gambling",
                help="Describe your industry for AI competitor discovery"
            )
            
            competitor_method = st.radio(
                "Competitor Discovery Method:",
                ["Manual Entry", "AI-Powered Discovery"]
            )
            
            if competitor_method == "Manual Entry":
                competitors_input = st.text_area(
                    "Enter Competitors (Max 5)",
                    "bet365.com, draftkings.com, betfair.com, williamhill.com, unibet.com",
                    help="Enter competitor domains separated by commas (maximum 5 competitors)"
                )
                competitors = [c.strip().lower() for c in competitors_input.split(',') if c.strip()]
                if len(competitors) > 5:
                    st.warning(f"You entered {len(competitors)} competitors. Only the first 5 will be used.")
                    competitors = competitors[:5]
                elif len(competitors) == 0:
                    st.warning("Please enter at least one competitor.")
                
                if competitors:
                    st.success(f"Using {len(competitors)} competitors:")
                    for i, comp in enumerate(competitors, 1):
                        st.write(f"{i}. {comp}")
            else:
                if st.button("🤖 Discover Competitors with AI"):
                    with st.spinner("Discovering competitors..."):
                        competitors = discover_competitors_ai(openai_client, brand, industry_description)
                        st.session_state.competitors = competitors
                        if competitors:
                            st.success(f"Found {len(competitors)} competitors!")
                            for i, comp in enumerate(competitors, 1):
                                st.write(f"{i}. {comp}")
                
                competitors = st.session_state.get('competitors', [])
        
        # Query Generation Settings
        with st.expander("🔍 Query Generation Settings", expanded=True):
            problem_area = st.text_input("Problem Area", "online gambling security")
            product_category = st.text_input("Product Category", "crypto casino")
            
            queries_per_stage = st.selectbox(
                "Queries per Funnel Stage",
                options=[3, 5, 7, 10, 15],
                index=1,
                help="Choose how many queries to generate for each buyer journey stage"
            )
            
            selected_stages = st.multiselect(
                "Select Buyer Journey Stages",
                list(XFUNNEL_STAGES.keys()),
                default=list(XFUNNEL_STAGES.keys())
            )
            
            selected_platforms = st.multiselect(
                "AI Platforms to Analyze",
                ["ChatGPT", "Claude", "Gemini", "Perplexity"],
                default=["ChatGPT", "Claude"]
            )
            
            # Show query calculation
            total_queries = len(selected_stages) * queries_per_stage
            total_responses = total_queries * len(selected_platforms)
            
            st.markdown(f"""
            <div class="query-info">
                <strong>📊 Analysis Scope:</strong><br>
                • {queries_per_stage} queries × {len(selected_stages)} stages = <strong>{total_queries} total queries</strong><br>
                • {total_queries} queries × {len(selected_platforms)} platforms = <strong>{total_responses} AI responses</strong><br>
                • Estimated time: <strong>{int(total_responses * 0.3 / 60)} - {int(total_responses * 0.5 / 60)} minutes</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis Controls
        st.markdown("---")
        run_analysis = st.button("🚀 Run Enhanced Analysis", type="primary", use_container_width=True)
        
        if st.session_state.analysis_complete:
            if st.button("🗑️ Clear Results", use_container_width=True):
                for key in ['analysis_complete', 'results_df', 'current_metrics']:
                    if key == 'analysis_complete':
                        st.session_state[key] = False
                    else:
                        st.session_state[key] = pd.DataFrame() if 'df' in key else {}
                st.rerun()
    
    # Main Analysis Logic
    if run_analysis:
        if not brand or not competitors:
            st.error("❌ Please ensure you have a brand and at least one competitor configured.")
            return
        
        if not selected_stages:
            st.error("❌ Please select at least one buyer journey stage.")
            return
        
        if not selected_platforms:
            st.error("❌ Please select at least one AI platform.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            all_results = []
            
            status_text.text("🔄 Generating xFunnel buyer journey queries...")
            progress_bar.progress(5)
            
            # Generate queries for each stage
            all_queries = {}
            for i, stage in enumerate(selected_stages):
                status_text.text(f"📝 Generating {queries_per_stage} queries for {stage}...")
                queries = generate_xfunnel_queries(
                    openai_client, brand, industry_description, 
                    problem_area, product_category, stage, queries_per_stage
                )
                all_queries[stage] = queries
                progress_val = int(5 + (i * 15 / len(selected_stages)))
                progress_bar.progress(progress_val)
            
            # Flatten queries for processing
            total_queries = []
            for stage, queries in all_queries.items():
                for query in queries:
                    total_queries.append((query, stage))
            
            st.success(f"✅ Generated {len(total_queries)} queries across {len(selected_stages)} buyer journey stages!")
            
            # Process each query with each platform
            total_combinations = len(total_queries) * len(selected_platforms)
            processed = 0
            
            for query, stage in total_queries:
                for platform in selected_platforms:
                    progress_val = int(20 + (processed * 60 / total_combinations))
                    progress_bar.progress(progress_val)
                    status_text.text(f"🤖 Processing: {platform} - {query[:50]}...")
                    
                    # Generate AI response
                    ai_response = simulate_comprehensive_ai_response(
                        openai_client, query, platform, stage
                    )
                    
                    if ai_response:
                        # Extract citations
                        citations = extract_citations_enhanced_fixed(ai_response)
                        
                        # Process each citation
                        for citation in citations:
                            domain = citation['domain']
                            
                            # Check if citation matches brand or competitors
                            is_brand = brand in domain.lower()
                            is_competitor = any(comp in domain.lower() for comp in competitors)
                            
                            if is_brand or is_competitor or len(all_results) < 500:
                                # Classify sentiment
                                sentiment = classify_sentiment(
                                    openai_client, citation['context'], citation['citation_text']
                                )
                                
                                # Get domain authority
                                domain_authority = get_domain_authority_score(domain, dataforseo_config)
                                
                                all_results.append({
                                    'query': query,
                                    'stage': stage,
                                    'platform': platform,
                                    'ai_response': ai_response,
                                    'citation_text': citation['citation_text'],
                                    'domain': domain,
                                    'citation_rank': citation['citation_rank'],
                                    'type': citation['type'],
                                    'context': citation['context'],
                                    'sentiment': sentiment,
                                    'is_brand': is_brand,
                                    'is_competitor': is_competitor,
                                    'domain_authority': domain_authority,
                                    'timestamp': datetime.now()
                                })
                    
                    processed += 1
                    time.sleep(0.3)
            
            # Finalize analysis
            status_text.text("📊 Calculating enhanced metrics...")
            progress_bar.progress(90)
            
            if all_results:
                df = pd.DataFrame(all_results)
                metrics = calculate_xfunnel_metrics(df, brand, competitors)
            else:
                df = pd.DataFrame()
                metrics = {}
            
            # Store results
            st.session_state.results_df = df
            st.session_state.current_metrics = metrics
            st.session_state.analysis_complete = True
            
            progress_bar.progress(100)
            status_text.text("✅ Enhanced analysis complete!")
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"""🎉 Enhanced xFunnel Analysis Completed!
            • Generated {queries_per_stage} queries per stage × {len(selected_stages)} stages = {len(total_queries)} total queries
            • Processed {len(selected_platforms)} AI platforms = {total_combinations} total responses
            • Found {len(all_results)} total citations with valid domains
            • Identified {len([r for r in all_results if r['is_brand']])} brand mentions
            • Competitor mentions: {len([r for r in all_results if r['is_competitor']])}""")
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.code(traceback.format_exc())
            return
    
    # Enhanced Results Display
    if st.session_state.analysis_complete:
        df = st.session_state.results_df
        metrics = st.session_state.current_metrics
        
        if not df.empty:
            # Main Dashboard Overview
            create_main_dashboard_overview(
                total_queries=df['query'].nunique(),
                total_responses=len(df),
                personas=len(selected_stages) if 'selected_stages' in locals() else 5,
                industries=1,
                regions=1
            )
            
            st.markdown("---")
            
            # Show sample of extracted domains
            unique_domains = df['domain'].unique()
            st.markdown(f"""
            <div class="domain-preview">
                <strong>🔍 Valid Domains Extracted ({len(unique_domains)} unique):</strong><br>
                {", ".join(unique_domains[:15])}{"..." if len(unique_domains) > 15 else ""}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Enhanced Dashboard Tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "🎯 Overview", "📈 Trends", "💭 Sentiment", "🔍 Query Details", 
                "🏆 Competition", "📊 Platform Insights", "📋 Export"
            ])
            
            with tab1:
                # Buyer Journey Metrics
                create_buyer_journey_metrics(metrics, brand)
                st.markdown("---")
                # Citation Analysis Overview
                create_citation_analysis_overview(df)
            
            with tab2:
                # Enhanced Trend Analysis
                create_trend_analysis(df, brand, competitors)
            
            with tab3:
                # Enhanced Sentiment Analysis
                create_enhanced_sentiment_analysis(df, metrics)
            
            with tab4:
                # Detailed Query Citation Explorer
                create_query_citation_details(df)
            
            with tab5:
                st.subheader("🏆 Competitive Analysis")
                
                # Competitive comparison
                comp_data = []
                for brand_name, brand_metrics in metrics.items():
                    comp_data.append({
                        'Brand': brand_name,
                        'Mentions': brand_metrics.get('mentions', 0),
                        'Share of Voice': brand_metrics.get('share_of_voice', 0),
                        'Avg Rank': brand_metrics.get('avg_rank', 0),
                        'Sentiment Score': brand_metrics.get('sentiment_score', 0)
                    })
                
                if comp_data:
                    comp_df = pd.DataFrame(comp_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Share of voice comparison
                        fig_sov = px.pie(
                            comp_df,
                            values='Share of Voice',
                            names='Brand',
                            title="Share of Voice Comparison"
                        )
                        st.plotly_chart(fig_sov, use_container_width=True)
                    
                    with col2:
                        # Competitive positioning
                        fig_positioning = px.scatter(
                            comp_df,
                            x='Mentions',
                            y='Sentiment Score',
                            size='Share of Voice',
                            color='Brand',
                            title="Competitive Positioning: Mentions vs Sentiment"
                        )
                        st.plotly_chart(fig_positioning, use_container_width=True)
            
            with tab6:
                st.subheader("📊 Platform Insights")
                
                # Platform performance analysis
                platform_data = []
                brand_metrics = metrics.get(brand, {})
                
                for platform, data in brand_metrics.get('platform_performance', {}).items():
                    platform_data.append({
                        'Platform': platform,
                        'Mentions': data['mentions'],
                        'Avg Rank': data['avg_rank']
                    })
                
                if platform_data:
                    platform_df = pd.DataFrame(platform_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_platform_mentions = px.bar(
                            platform_df,
                            x='Platform',
                            y='Mentions',
                            title="Brand Mentions by Platform"
                        )
                        st.plotly_chart(fig_platform_mentions, use_container_width=True)
                    
                    with col2:
                        fig_platform_rank = px.bar(
                            platform_df,
                            x='Platform',
                            y='Avg Rank',
                            title="Average Citation Rank by Platform"
                        )
                        st.plotly_chart(fig_platform_rank, use_container_width=True)
            
            with tab7:
                st.subheader("📋 Data Export & Reports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Export Options")
                    
                    if not df.empty:
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Complete Analysis (CSV)",
                            csv_data,
                            f"enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        json_data = df.to_json(orient='records', indent=2, date_format='iso')
                        st.download_button(
                            "📥 Download Analysis (JSON)",
                            json_data,
                            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json",
                            use_container_width=True
                        )
                
                with col2:
                    st.markdown("#### 📈 Executive Summary")
                    
                    if metrics:
                        brand_metrics = metrics.get(brand, {})
                        
                        executive_report = f"""
# Enhanced AI Citation Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
Brand: {brand}
Industry: {industry_description if 'industry_description' in locals() else 'Not specified'}
Competitors Analyzed: {len(competitors)}

## Analysis Configuration
- Queries per Stage: {queries_per_stage if 'queries_per_stage' in locals() else 'N/A'}
- Funnel Stages: {len(selected_stages) if 'selected_stages' in locals() else 'N/A'}
- AI Platforms: {len(selected_platforms) if 'selected_platforms' in locals() else 'N/A'}

## Key Metrics
- Total Queries Analyzed: {df['query'].nunique()}
- Total AI Responses: {len(df)}
- Valid Domains Extracted: {df['domain'].nunique()}
- Brand Mentions: {brand_metrics.get('mentions', 0)}
- Share of Voice: {brand_metrics.get('share_of_voice', 0):.1f}%
- Sentiment Score: {brand_metrics.get('sentiment_score', 0):.1f}

## Sentiment Distribution
"""
                        
                        sentiment_dist = brand_metrics.get('sentiment_distribution', {})
                        for sentiment, count in sentiment_dist.items():
                            percentage = (count / sum(sentiment_dist.values())) * 100 if sentiment_dist else 0
                            executive_report += f"- {sentiment}: {count} ({percentage:.1f}%)\n"
                        
                        executive_report += f"""

## Buyer Journey Performance
"""
                        
                        for stage, data in brand_metrics.get('stage_performance', {}).items():
                            executive_report += f"- {stage}: {data['mentions']} mentions ({data['visibility']:.1f}% visibility)\n"
                        
                        executive_report += f"""

## Competitive Landscape
"""
                        
                        for comp_brand, comp_metrics in metrics.items():
                            if comp_brand != brand and comp_metrics.get('mentions', 0) > 0:
                                executive_report += f"- {comp_brand}: {comp_metrics['mentions']} mentions, {comp_metrics.get('share_of_voice', 0):.1f}% SOV\n"
                        
                        st.download_button(
                            "📋 Download Executive Report",
                            executive_report,
                            f"executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            "text/plain",
                            use_container_width=True
                        )
        else:
            st.info("No results to display. Please run an analysis first.")
    
    else:
        # Enhanced Welcome Screen
        st.info("""
        🚀 **Welcome to AI Citation Tracker Pro - Enhanced Edition**
        
        **Complete xFunnel.ai replica with advanced sentiment analysis, trend visualization, and detailed query exploration**
        """)
        
        st.markdown("### ✨ New Enhanced Features:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            **💭 Advanced Sentiment Analysis**
            
            • Comprehensive sentiment charts by platform and brand
            • Sentiment score trending over time
            • Color-coded sentiment visualization
            • Detailed sentiment breakdowns
            """)
        
        with col2:
            st.info("""
            **📈 Trend Analysis**
            
            • 30-day trend line graphs
            • Mentions, sentiment, and visibility trends
            • Multi-brand comparison charts
            • Interactive trend visualization
            """)
        
        with col3:
            st.warning("""
            **🔍 Query Explorer**
            
            • Click to explore detailed citations
            • Full AI responses with extracted citations
            • Platform-by-platform analysis
            • Context and sentiment for each citation
            """)
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide:")
        st.markdown("""
        1. **Configure** your brand and competitors
        2. **Set parameters** for analysis scope  
        3. **Select stages** and AI platforms
        4. **Run analysis** to get comprehensive insights
        5. **Explore results** with interactive charts and detailed views
        """)

    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
        🎯 <strong>AI Citation Tracker Pro - Enhanced Edition</strong>
        <br>Complete xFunnel.ai replica with advanced sentiment analysis, trend visualization, and detailed exploration
        <br>Built with ❤️ using Streamlit, OpenAI & DataForSEO
        <br><em>Professional-grade AI citation tracking with enhanced analytics</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
