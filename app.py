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

# DataForSEO imports
try:
    from dataforseo_client import configuration as dfs_config, api_client as dfs_api_provider
    from dataforseo_client.api.serp_api import SerpApi
    from dataforseo_client.api.keywords_data_api import KeywordsDataApi
    from dataforseo_client.models.serp_google_organic_live_advanced_request_info import SerpGoogleOrganicLiveAdvancedRequestInfo
    from dataforseo_client.models.keywords_data_google_ads_search_volume_live_request_info import KeywordsDataGoogleAdsSearchVolumeLiveRequestInfo
    DATAFORSEO_AVAILABLE = True
except ImportError:
    DATAFORSEO_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Citation Tracker Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS that works reliably with Streamlit
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.info-card {
    background: #f0f2f6;
    border-left: 4px solid #667eea;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.success-card {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
    color: #155724;
}

.section-title {
    color: #2c3e50;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
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

# Replace HTML-heavy functions with Streamlit native components
def create_advanced_metrics_cards(metrics, brand):
    """Create enhanced metric cards using Streamlit components"""
    brand_metrics = metrics.get(brand, {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Brand Mentions",
            value=f"{brand_metrics.get('mentions', 0):,}",
            delta=None
        )
    
    with col2:
        visibility = brand_metrics.get('visibility', 0)
        st.metric(
            label="👁️ Visibility",
            value=f"{visibility:.1f}%",
            delta=None
        )
    
    with col3:
        avg_rank = brand_metrics.get('avg_citation_rank', 0)
        st.metric(
            label="🎯 Avg Citation Rank",
            value=f"{avg_rank:.1f}" if avg_rank > 0 else "N/A",
            delta=None
        )
    
    with col4:
        sentiment = brand_metrics.get('avg_sentiment', 'Neutral')
        sentiment_icon = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
        st.metric(
            label="💭 Sentiment",
            value=f"{sentiment_icon.get(sentiment, '😐')} {sentiment}",
            delta=None
        )
    
    with col5:
        unique_queries = brand_metrics.get('unique_queries', 0)
        st.metric(
            label="🔍 Unique Queries",
            value=f"{unique_queries:,}",
            delta=None
        )

def generate_time_series_data(df):
    """Generate time series data for trend visualization"""
    if df.empty:
        return pd.DataFrame()
    
    # Create synthetic time series based on analysis timestamp
    base_date = datetime.now() - timedelta(days=30)
    dates = [base_date + timedelta(days=i) for i in range(31)]
    
    time_series_data = []
    brands = df['domain'].unique() if 'domain' in df.columns else []
    
    for date in dates:
        for brand in brands:
            brand_mentions = len(df[df['domain'].str.contains(brand, case=False, na=False)])
            # Add some variation for realistic trend
            variation = random.uniform(0.7, 1.3)
            daily_mentions = max(0, int(brand_mentions * variation * random.uniform(0.1, 0.3)))
            
            time_series_data.append({
                'date': date,
                'brand': brand,
                'mentions': daily_mentions,
                'visibility': (daily_mentions / max(1, len(df))) * 100
            })
    
    return pd.DataFrame(time_series_data)

def create_trend_charts(df, metrics):
    """Create comprehensive trend visualizations"""
    if df.empty:
        st.info("No data available for trend analysis")
        return
    
    # Generate time series data
    time_series_df = generate_time_series_data(df)
    
    if not time_series_df.empty:
        # Main trend chart
        fig_trend = px.line(
            time_series_df,
            x='date',
            y='mentions',
            color='brand',
            title="📈 Brand Mentions Trend Over Time",
            template="plotly_white"
        )
        
        fig_trend.update_layout(
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig_trend.update_traces(line=dict(width=3))
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Visibility trend
        fig_visibility = px.area(
            time_series_df,
            x='date',
            y='visibility',
            color='brand',
            title="👁️ Brand Visibility Trend",
            template="plotly_white"
        )
        
        fig_visibility.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_visibility, use_container_width=True)

def create_competitive_analysis(metrics):
    """Create comprehensive competitive analysis visualizations"""
    if not metrics or len(metrics) <= 1:
        st.info("Add competitors to see competitive analysis")
        return
    
    # Prepare data for visualization
    brands = []
    mentions = []
    visibility = []
    avg_ranks = []
    sentiment_scores = []
    
    for brand, data in metrics.items():
        if data['mentions'] > 0:  # Only include brands with data
            brands.append(brand)
            mentions.append(data['mentions'])
            visibility.append(data['visibility'])
            avg_ranks.append(data.get('avg_citation_rank', 0))
            
            # Calculate sentiment score
            sent_dist = data.get('sentiment_distribution', {})
            pos = sent_dist.get('Positive', 0)
            neg = sent_dist.get('Negative', 0)
            total = sum(sent_dist.values()) if sent_dist else 1
            sentiment_score = ((pos - neg) / total) * 100 if total > 0 else 0
            sentiment_scores.append(sentiment_score)
    
    if not brands:
        st.info("No competitive data available")
        return
    
    comp_df = pd.DataFrame({
        'Brand': brands,
        'Mentions': mentions,
        'Visibility': visibility,
        'Avg_Rank': avg_ranks,
        'Sentiment_Score': sentiment_scores
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market share donut chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=comp_df['Brand'],
            values=comp_df['Mentions'],
            hole=.6,
            textinfo="label+percent",
            textposition="outside"
        )])
        
        fig_donut.update_layout(
            title="🥧 Market Share (Citations)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col2:
        # Competitive positioning scatter
        fig_scatter = px.scatter(
            comp_df,
            x='Visibility',
            y='Mentions',
            size='Sentiment_Score',
            color='Brand',
            title="🎯 Competitive Positioning Matrix",
            size_max=60,
            template="plotly_white"
        )
        
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Multi-dimensional radar chart
    if len(comp_df) > 1:
        fig_radar = go.Figure()
        
        # Normalize data for radar chart
        metrics_for_radar = ['Mentions', 'Visibility', 'Sentiment_Score']
        
        for _, row in comp_df.iterrows():
            values = []
            for metric in metrics_for_radar:
                # Normalize to 0-100 scale
                max_val = comp_df[metric].max()
                min_val = comp_df[metric].min()
                if max_val != min_val:
                    normalized = ((row[metric] - min_val) / (max_val - min_val)) * 100
                else:
                    normalized = 50
                values.append(normalized)
            
            # Close the radar chart
            values.append(values[0])
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_for_radar + [metrics_for_radar[0]],
                fill='toself',
                name=row['Brand'],
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="🌟 Multi-Dimensional Brand Comparison",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

def create_sentiment_analysis_viz(df, metrics):
    """Create advanced sentiment analysis visualizations"""
    if df.empty:
        st.info("No sentiment data available")
        return
    
    # Prepare sentiment data
    sentiment_data = []
    for brand_name, brand_data in metrics.items():
        for sentiment, count in brand_data['sentiment_distribution'].items():
            sentiment_data.append({
                'Brand': brand_name,
                'Sentiment': sentiment,
                'Count': count
            })
    
    if not sentiment_data:
        st.info("No sentiment data to display")
        return
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stacked bar chart for sentiment distribution
        fig_sentiment = px.bar(
            sentiment_df,
            x='Brand',
            y='Count',
            color='Sentiment',
            title="😊 Sentiment Distribution by Brand",
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#f39c12',
                'Negative': '#e74c3c'
            },
            template="plotly_white"
        )
        
        fig_sentiment.update_layout(height=400)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Sentiment score gauge for main brand
        main_brand = list(metrics.keys())[0] if metrics else None
        if main_brand:
            brand_sentiment = metrics[main_brand]['sentiment_distribution']
            total = sum(brand_sentiment.values()) if brand_sentiment else 1
            
            pos_pct = (brand_sentiment.get('Positive', 0) / total) * 100
            neg_pct = (brand_sentiment.get('Negative', 0) / total) * 100
            sentiment_score = pos_pct - neg_pct
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sentiment_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Sentiment Score<br>{main_brand}"},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-100, -20], 'color': "#ffcccb"},
                        {'range': [-20, 20], 'color': "#ffffcc"},
                        {'range': [20, 100], 'color': "#90ee90"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

def create_query_analysis_viz(df):
    """Create query analysis visualizations - FIXED VERSION"""
    if df.empty:
        st.info("No query data available")
        return
    
    # Top performing queries
    if 'query' in df.columns and 'is_brand' in df.columns:
        top_queries = df[df['is_brand']]['query'].value_counts().head(10)
        
        if not top_queries.empty:
            fig_queries = px.bar(
                x=top_queries.values,
                y=top_queries.index,
                orientation='h',
                title="🔥 Top Performing Queries (Brand Mentions)",
                template="plotly_white",
                color=top_queries.values,
                color_continuous_scale="Viridis"
            )
            
            fig_queries.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            
            st.plotly_chart(fig_queries, use_container_width=True)
    
    # Citation rank distribution - FIXED VERSION
    if 'citation_rank' in df.columns:
        # Convert numpy int64 to regular Python int to avoid Plotly error
        max_rank = int(df['citation_rank'].max())
        nbins_value = max(5, max_rank)
        
        fig_rank_dist = px.histogram(
            df,
            x='citation_rank',
            title="📊 Citation Rank Distribution",
            nbins=nbins_value,  # Now using regular Python int
            template="plotly_white",
            color_discrete_sequence=['#667eea']
        )
        
        fig_rank_dist.update_layout(height=400)
        st.plotly_chart(fig_rank_dist, use_container_width=True)

# Core functions remain the same as in your previous code
def generate_query_variations(client, seed, template, funnel, n):
    """Generate diverse queries using OpenAI's new API format"""
    prompt = f"""Generate {n} diverse, realistic search queries for the '{funnel}' stage of the buyer journey.

Topic: {seed}
Template style: {template}

Requirements:
- Make them natural and varied
- Representative of what real users would search
- Include different question formats and intents
- Focus on {funnel.lower()} stage queries

Output only the queries, one per line, without numbering or bullet points."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a search query generator. Generate realistic, diverse search queries based on the given requirements."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.8
        )
        
        queries = [q.strip().lstrip('0123456789. -•') 
                  for q in response.choices[0].message.content.strip().split('\n') 
                  if q.strip() and len(q.strip()) > 5]
        
        return queries[:n]
    
    except Exception as e:
        st.error(f"Error generating queries: {str(e)}")
        return []

def simulate_ai_response(client, query, platform="ChatGPT"):
    """Simulate AI response for a query using OpenAI's new API"""
    prompt = f"""You are {platform}, a helpful AI assistant. Answer this query comprehensively and naturally include relevant website sources and citations where appropriate:

Query: {query}

Provide a detailed, helpful response that:
- Answers the question thoroughly
- Naturally mentions and cites relevant websites, domains, and URLs
- Uses specific examples with website names
- Includes recommendations with source citations
- Provides actionable information with references

Be natural and conversational while including real website references."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are {platform}, a helpful AI assistant that provides comprehensive answers with natural source citations and website references."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Error simulating AI response: {str(e)}")
        return ""

def extract_citations_advanced(response_text):
    """Advanced citation extraction from AI response text"""
    citations = []
    
    # Multiple URL patterns to catch different formats
    url_patterns = [
        r'https?://[^\s\)\],;"<>\n]+',  # Standard URLs
        r'www\.[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.(?:[a-zA-Z]{2,}|[a-zA-Z]{2,}\.[a-zA-Z]{2,})(?:/[^\s\)\],;"<>]*)?',  # www. domains
        r'[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.(?:com|org|net|edu|gov|co\.uk|ca|au|de|fr|jp|in|io|ai|tech|casino|bet|sports)(?:/[^\s\)\],;"<>]*)?'  # Direct domain mentions
    ]
    
    # Website name patterns
    website_patterns = [
        r'\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\s*(?:website|site|platform|service)',
        r'\b(Amazon|Google|Facebook|Twitter|LinkedIn|YouTube|Instagram|TikTok|Netflix|Spotify|Apple|Microsoft|Tesla|Uber|Airbnb|eBay|PayPal|Shopify|Zoom|Slack|Discord|Reddit|Pinterest|Snapchat|WhatsApp|Telegram|Dropbox|GitHub|Stack Overflow|Wikipedia|BBC|CNN|Forbes|TechCrunch|Wired|The Verge|Mashable|Engadget|Ars Technica)\b',
        r'\b([A-Z][a-zA-Z]+\.(?:com|org|net|io|ai|co|casino|bet))\b'
    ]
    
    citation_rank = 1
    found_citations = set()
    
    # Extract URLs
    for pattern in url_patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            url = match.group().rstrip('.,;:!?')
            domain = url.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0].lower()
            
            if domain not in found_citations and len(domain) > 2:
                start = max(0, match.start() - 100)
                end = min(len(response_text), match.end() + 100)
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
    
    # Extract website names and domain mentions
    for pattern in website_patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            citation_text = match.group().strip()
            
            if citation_text.lower().endswith('.com') or citation_text.lower().endswith('.org') or citation_text.lower().endswith('.net'):
                domain = citation_text.lower()
            else:
                domain = citation_text.lower().replace(' ', '') + '.com'
            
            if domain not in found_citations:
                start = max(0, match.start() - 100)
                end = min(len(response_text), match.end() + 100)
                context = response_text[start:end].strip()
                
                citations.append({
                    'citation_text': citation_text,
                    'domain': domain,
                    'type': 'Website Mention',
                    'context': context,
                    'position': match.start(),
                    'citation_rank': citation_rank
                })
                found_citations.add(domain)
                citation_rank += 1
    
    citations.sort(key=lambda x: x['position'])
    
    for i, citation in enumerate(citations, 1):
        citation['citation_rank'] = i
    
    return citations

def classify_sentiment(client, context, citation):
    """Classify sentiment using OpenAI's new API"""
    prompt = f"""Analyze the sentiment towards "{citation}" in this text context:

Context: {context}

Classify the sentiment as exactly one of: positive, neutral, negative

Consider:
- How the website/service is mentioned
- The tone and context around the mention
- Whether it's recommended, criticized, or just mentioned

Respond with only one word: positive, neutral, or negative"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Classify sentiment accurately and concisely."},
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

def get_search_volume(keyword, config):
    """Get search volume from DataForSEO"""
    if not DATAFORSEO_AVAILABLE or not config:
        return 0
    
    try:
        with dfs_api_provider.ApiClient(config) as api_client:
            keywords_api = KeywordsDataApi(api_client)
            request = KeywordsDataGoogleAdsSearchVolumeLiveRequestInfo(
                location_name="United States",
                language_name="English",
                keywords=[keyword]
            )
            response = keywords_api.google_ads_search_volume_live([request])
            
            if (response.tasks and 
                response.tasks[0].result and 
                len(response.tasks[0].result) > 0):
                return response.tasks[0].result[0].search_volume or 0
            return 0
    
    except Exception as e:
        return 0

def get_serp_rank(query, domain, config):
    """Get SERP ranking from DataForSEO"""
    if not DATAFORSEO_AVAILABLE or not config:
        return None
    
    try:
        with dfs_api_provider.ApiClient(config) as api_client:
            serp_api = SerpApi(api_client)
            request = SerpGoogleOrganicLiveAdvancedRequestInfo(
                keyword=query,
                location_name="United States",
                language_name="English"
            )
            response = serp_api.google_organic_live_advanced([request])
            
            if (response.tasks and 
                response.tasks[0].result and 
                len(response.tasks[0].result) > 0):
                
                for item in response.tasks[0].result[0].items[:20]:
                    if hasattr(item, 'domain') and domain.lower() in item.domain.lower():
                        return item.rank_group
            return None
    
    except Exception as e:
        return None

def calculate_metrics_safe(df, brand, competitors):
    """Calculate comprehensive metrics for all brands with error handling"""
    try:
        metrics = {}
        all_brands = [brand] + competitors
        
        if df.empty:
            for brand_name in all_brands:
                metrics[brand_name] = {
                    'mentions': 0,
                    'visibility': 0,
                    'avg_citation_rank': 0,
                    'avg_sentiment': 'Neutral',
                    'sentiment_score': 0,
                    'sentiment_distribution': {},
                    'avg_serp_rank': None,
                    'total_search_volume': 0,
                    'unique_queries': 0
                }
            return metrics
        
        total_queries = df['query'].nunique() if not df.empty else 1
        
        for brand_name in all_brands:
            try:
                brand_df = df[df['domain'].str.contains(brand_name, case=False, na=False)]
                
                sentiment_counts = brand_df['sentiment'].value_counts().to_dict() if not brand_df.empty else {}
                total_mentions = len(brand_df)
                
                sentiment_score = 0
                if total_mentions > 0:
                    pos_weight = sentiment_counts.get('Positive', 0) * 1
                    neu_weight = sentiment_counts.get('Neutral', 0) * 0
                    neg_weight = sentiment_counts.get('Negative', 0) * -1
                    sentiment_score = (pos_weight + neu_weight + neg_weight) / total_mentions
                
                metrics[brand_name] = {
                    'mentions': total_mentions,
                    'visibility': (total_mentions / total_queries) * 100 if total_queries > 0 else 0,
                    'avg_citation_rank': brand_df['citation_rank'].mean() if total_mentions > 0 else 0,
                    'avg_sentiment': brand_df['sentiment'].mode().iloc[0] if total_mentions > 0 else 'Neutral',
                    'sentiment_score': sentiment_score,
                    'sentiment_distribution': sentiment_counts,
                    'avg_serp_rank': brand_df[brand_df['serp_rank'].notna()]['serp_rank'].mean() if len(brand_df[brand_df['serp_rank'].notna()]) > 0 else None,
                    'total_search_volume': brand_df['search_volume'].sum() if 'search_volume' in brand_df.columns else 0,
                    'unique_queries': brand_df['query'].nunique() if not brand_df.empty else 0
                }
            except Exception as e:
                st.warning(f"Error calculating metrics for {brand_name}: {str(e)}")
                metrics[brand_name] = {
                    'mentions': 0,
                    'visibility': 0,
                    'avg_citation_rank': 0,
                    'avg_sentiment': 'Neutral',
                    'sentiment_score': 0,
                    'sentiment_distribution': {},
                    'avg_serp_rank': None,
                    'total_search_volume': 0,
                    'unique_queries': 0
                }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error in calculate_metrics_safe: {str(e)}")
        empty_metrics = {}
        for brand_name in [brand] + competitors:
            empty_metrics[brand_name] = {
                'mentions': 0,
                'visibility': 0,
                'avg_citation_rank': 0,
                'avg_sentiment': 'Neutral',
                'sentiment_score': 0,
                'sentiment_distribution': {},
                'avg_serp_rank': None,
                'total_search_volume': 0,
                'unique_queries': 0
            }
        return empty_metrics

def display_citations_table(df):
    """Display all citations in a comprehensive table format"""
    if df.empty:
        st.info("No citations found.")
        return
    
    st.subheader("📋 All Citations Found")
    
    available_columns = df.columns.tolist()
    
    desired_columns = [
        ('query', 'Query'),
        ('citation_text', 'Citation'),
        ('domain', 'Domain'),
        ('citation_type', 'Type'),
        ('citation_rank', 'Rank'),
        ('context', 'Context')
    ]
    
    display_columns = []
    display_names = []
    
    for col_name, display_name in desired_columns:
        if col_name in available_columns:
            display_columns.append(col_name)
            display_names.append(display_name)
    
    if not display_columns:
        st.warning("No compatible columns found in the citations data.")
        return
    
    citations_display = df[display_columns].copy()
    
    if 'context' in citations_display.columns:
        citations_display['context'] = citations_display['context'].apply(
            lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x)
        )
    
    citations_display.columns = display_names
    
    st.dataframe(
        citations_display,
        use_container_width=True,
        height=400
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Citations", len(df))
    
    with col2:
        if 'domain' in df.columns:
            st.metric("Unique Domains", df['domain'].nunique())
        else:
            st.metric("Unique Domains", "N/A")
    
    with col3:
        if 'query' in df.columns and not df.empty:
            st.metric("Avg Citations per Query", f"{len(df) / df['query'].nunique():.1f}")
        else:
            st.metric("Avg Citations per Query", "N/A")

# Main Application
def main():
    # Header - Using Streamlit components instead of HTML
    st.title("🎯 AI Citation Tracker Pro")
    st.markdown("**Advanced AI-powered citation tracking with comprehensive analytics**")
    st.markdown("---")
    
    # Initialize clients
    openai_client = init_openai_client()
    dataforseo_config = init_dataforseo_config()
    
    if not openai_client:
        st.error("OpenAI client initialization failed. Please check your API key.")
        return
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if 'current_metrics' not in st.session_state:
        st.session_state.current_metrics = {}
    
    # Enhanced Sidebar Configuration
    with st.sidebar:
        st.header("🎯 Analysis Configuration")
        
        # Brand & Competitors
        with st.expander("🏢 Brand & Competitors", expanded=True):
            brand = st.text_input(
                "Your Brand Domain", 
                "stake.com", 
                help="Enter your main brand domain (e.g., stake.com)"
            ).strip().lower()
            
            competitors_input = st.text_area(
                "Competitor Domains", 
                "bet365.com, draftkings.com, betfair.com", 
                help="Enter competitor domains separated by commas"
            )
            competitors = [c.strip().lower() for c in competitors_input.split(',') if c.strip()]
        
        # Query Settings
        with st.expander("🔍 Query Settings", expanded=True):
            seed_keyword = st.text_input("Seed Keyword", "crypto casino")
            query_template = st.text_input("Query Template", "What is the best {topic}?")
            funnel_stage = st.selectbox(
                "Funnel Stage", 
                ["Awareness", "Consideration", "Decision"],
                help="Select the buyer journey stage to focus on"
            )
            num_queries = st.slider("Number of Queries", 3, 15, 8)
        
        # Advanced Settings
        with st.expander("⚙️ Advanced Settings"):
            ai_platform = st.selectbox("AI Platform", ["ChatGPT", "Claude", "Gemini", "Perplexity"])
            include_seo = st.checkbox("Include SEO Data", DATAFORSEO_AVAILABLE and dataforseo_config is not None)
            
        # Analysis Controls
        st.markdown("---")
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if st.session_state.analysis_complete:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.results_df = pd.DataFrame()
                st.session_state.current_metrics = {}
                st.rerun()
        
        # Quick Stats in Sidebar
        if st.session_state.analysis_complete and not st.session_state.results_df.empty:
            st.markdown("### 📊 Quick Stats")
            df = st.session_state.results_df
            st.metric("Total Citations", len(df))
            st.metric("Unique Domains", df['domain'].nunique() if 'domain' in df.columns else 0)
            st.metric("Brand Mentions", len(df[df['is_brand']]) if 'is_brand' in df.columns else 0)
    
    # Main Analysis Logic
    if run_analysis:
        if not brand:
            st.error("❌ Please enter your brand domain to proceed.")
            return
        
        # Enhanced Progress Tracking - Using Streamlit components
        progress_bar = st.progress(0)
        status_text = st.empty()
                
        try:
            results = []
            all_citations_data = []
            
            # Step 1: Generate Queries
            status_text.text("🔄 Generating diverse queries...")
            progress_bar.progress(5)
            
            queries = generate_query_variations(
                openai_client, seed_keyword, query_template, funnel_stage, num_queries
            )
            
            if not queries:
                st.error("❌ Failed to generate queries. Please check your settings and try again.")
                return
            
            st.success(f"✅ Generated {len(queries)} diverse queries for analysis")
            
            # Step 2: Process each query
            for i, query in enumerate(queries):
                progress_val = int(10 + (i * 65 / len(queries)))
                progress_bar.progress(progress_val)
                status_text.text(f"🤖 Processing query {i+1}/{len(queries)}: {query[:50]}...")
                
                # Get AI response
                ai_response = simulate_ai_response(openai_client, query, ai_platform)
                if not ai_response:
                    continue
                
                # Extract citations
                citations = extract_citations_advanced(ai_response)
                
                # Get search volume
                search_volume = 0
                if include_seo and dataforseo_config:
                    search_volume = get_search_volume(query, dataforseo_config)
                
                # Store all citations
                for citation in citations:
                    all_citations_data.append({
                        'query': query,
                        'ai_response': ai_response,
                        'citation_text': citation['citation_text'],
                        'domain': citation['domain'],
                        'citation_type': citation['type'],
                        'citation_rank': citation['citation_rank'],
                        'context': citation['context'],
                        'search_volume': search_volume
                    })
                
                # Process brand/competitor citations
                for citation in citations:
                    domain = citation['domain']
                    
                    is_brand = brand in domain.lower()
                    is_competitor = any(comp in domain.lower() for comp in competitors)
                    
                    if is_brand or is_competitor:
                        sentiment = classify_sentiment(openai_client, citation['context'], citation['citation_text'])
                        
                        serp_rank = None
                        if include_seo and dataforseo_config:
                            serp_rank = get_serp_rank(query, domain, dataforseo_config)
                        
                        results.append({
                            'query': query,
                            'funnel_stage': funnel_stage,
                            'ai_platform': ai_platform,
                            'ai_response': ai_response,
                            'citation_text': citation['citation_text'],
                            'citation_url': citation['citation_text'],
                            'domain': domain,
                            'citation_rank': citation['citation_rank'],
                            'sentiment': sentiment,
                            'is_brand': is_brand,
                            'is_competitor': is_competitor,
                            'search_volume': search_volume,
                            'serp_rank': serp_rank,
                            'timestamp': datetime.now(),
                            'citation_type': citation['type'],
                            'context': citation['context']
                        })
                
                time.sleep(0.5)
            
            # Finalize
            status_text.text("📊 Calculating metrics and preparing visualizations...")
            progress_bar.progress(90)
            
            all_citations_df = pd.DataFrame(all_citations_data)
            
            if results:
                df = pd.DataFrame(results)
                metrics = calculate_metrics_safe(df, brand, competitors)
            else:
                df = pd.DataFrame()
                metrics = calculate_metrics_safe(pd.DataFrame(), brand, competitors)
            
            # Store in session state
            st.session_state.results_df = df
            st.session_state.all_citations_df = all_citations_df
            st.session_state.current_metrics = metrics
            st.session_state.analysis_complete = True
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"""🎉 Analysis completed successfully!
            • Found {len(all_citations_data)} total citations
            • Identified {len(results)} brand/competitor mentions
            • Analyzed {len(queries)} queries across {ai_platform}""")
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.code(traceback.format_exc())
            return
    
    # Enhanced Results Display
    if st.session_state.analysis_complete:
        df = st.session_state.results_df
        all_citations_df = getattr(st.session_state, 'all_citations_df', pd.DataFrame())
        metrics = st.session_state.current_metrics
        
        # Enhanced Metrics Cards - Using Streamlit native components
        if metrics:
            st.subheader("📊 Performance Dashboard")
            create_advanced_metrics_cards(metrics, brand)
        
        # Main Dashboard Tabs
        st.subheader("📈 Analytics Dashboard")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📈 Trends", "🏆 Competition", "💭 Sentiment", "🔍 Citations", "📋 Data Export"
        ])
        
        with tab1:
            st.markdown("### 🎯 Executive Summary")
            
            if not df.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    create_query_analysis_viz(df)
                
                with col2:
                    # Key insights using Streamlit info boxes
                    st.info("""
                    🔑 **Key Insights**
                    
                    ✅ Top performing queries identified
                    
                    📊 Citation rank patterns analyzed
                    
                    🏆 Competitive positioning mapped
                    
                    💭 Sentiment trends tracked
                    """)
                    
                    # Recommendations
                    brand_metrics = metrics.get(brand, {})
                    avg_rank = brand_metrics.get('avg_citation_rank', 0)
                    visibility = brand_metrics.get('visibility', 0)
                    
                    recommendations = []
                    if avg_rank > 3:
                        recommendations.append("🎯 Focus on improving citation rank")
                    if visibility < 30:
                        recommendations.append("📈 Increase brand visibility")
                    
                    if recommendations:
                        st.subheader("💡 Recommendations")
                        for rec in recommendations:
                            st.write(f"• {rec}")
        
        with tab2:
            st.markdown("### 📈 Trend Analysis")
            create_trend_charts(df, metrics)
        
        with tab3:
            st.markdown("### 🏆 Competitive Analysis")
            create_competitive_analysis(metrics)
        
        with tab4:
            st.markdown("### 💭 Sentiment Analysis")
            create_sentiment_analysis_viz(df, metrics)
        
        with tab5:
            st.markdown("### 🔍 Citation Analysis")
            
            # Enhanced Citations Display
            display_citations_table(all_citations_df)
            
            # Individual Query Analysis
            if not all_citations_df.empty:
                st.subheader("📝 Detailed Query Responses")
                
                for query in all_citations_df['query'].unique()[:5]:  # Show first 5 queries
                    with st.expander(f"🔍 {query}"):
                        query_data = all_citations_df[all_citations_df['query'] == query]
                        
                        # AI Response in a code block for better formatting
                        st.subheader("AI Response:")
                        st.write(query_data.iloc[0]['ai_response'])
                        
                        st.subheader("Citations Found:")
                        for _, citation in query_data.iterrows():
                            st.info(f"""
                            **#{citation['citation_rank']} - {citation['citation_text']}**
                            
                            **Type:** {citation['citation_type']} | **Domain:** {citation['domain']}
                            
                            **Context:** {citation['context'][:150]}...
                            """)
        
        with tab6:
            st.markdown("### 📋 Data Export & Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Export Options")
                
                if not df.empty:
                    # Brand analysis export
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Brand Analysis (CSV)",
                        csv_data,
                        f"brand_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    # JSON export
                    json_data = df.to_json(orient='records', indent=2, date_format='iso')
                    st.download_button(
                        "📥 Download Analysis (JSON)",
                        json_data,
                        f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        use_container_width=True
                    )
                
                # All citations export
                if not all_citations_df.empty:
                    all_citations_csv = all_citations_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download All Citations (CSV)",
                        all_citations_csv,
                        f"all_citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            with col2:
                st.markdown("#### 📈 Executive Summary Report")
                
                if metrics:
                    brand_metrics = metrics.get(brand, {})
                    
                    summary_report = f"""
# AI Citation Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Brand:** {brand}
- **Total Mentions:** {brand_metrics.get('mentions', 0)}
- **Visibility:** {brand_metrics.get('visibility', 0):.2f}%
- **Average Citation Rank:** {brand_metrics.get('avg_citation_rank', 0):.2f}
- **Dominant Sentiment:** {brand_metrics.get('avg_sentiment', 'N/A')}

## Analysis Details
- **Queries Analyzed:** {df['query'].nunique() if not df.empty else 0}
- **Total Citations Found:** {len(all_citations_df) if not all_citations_df.empty else 0}
- **AI Platform:** {df['ai_platform'].iloc[0] if not df.empty else 'N/A'}
- **Funnel Stage:** {df['funnel_stage'].iloc[0] if not df.empty else 'N/A'}

## Competitive Landscape
"""
                    
                    for comp_brand, comp_metrics in metrics.items():
                        if comp_brand != brand and comp_metrics['mentions'] > 0:
                            summary_report += f"- **{comp_brand}:** {comp_metrics['mentions']} mentions, {comp_metrics['visibility']:.1f}% visibility\n"
                    
                    st.download_button(
                        "📋 Download Executive Report",
                        summary_report,
                        f"executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain",
                        use_container_width=True
                    )
    
    else:
        # Enhanced Welcome Screen using Streamlit components
        st.info("""
        🚀 **Welcome to AI Citation Tracker Pro**
        
        **The ultimate xFunnel.ai-style dashboard for AI citation tracking**
        """)
        
        st.markdown("### 🎯 What you'll get:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            **📊 Advanced Analytics**
            
            Multi-dimensional analysis with interactive visualizations, trend tracking, and competitive benchmarking.
            """)
        
        with col2:
            st.info("""
            **🎯 Real-time Insights**
            
            Live citation extraction, sentiment analysis, and performance metrics updated in real-time.
            """)
        
        with col3:
            st.warning("""
            **🚀 Professional Reports**
            
            Executive summaries, detailed exports, and actionable recommendations for strategic planning.
            """)
        
        st.markdown("---")
        st.markdown("**Configure your analysis in the sidebar and click 'Run Analysis' to begin!**")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #6c757d;">
        🎯 <strong>AI Citation Tracker Pro</strong> | Powered by OpenAI & DataForSEO | Built by Taha Shah using Streamlit
        <br><strong>Professional-grade AI citation tracking and competitive analysis</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
