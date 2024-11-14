import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
from datetime import datetime, timedelta

class GscKeywordAnalyzer:
    def __init__(self, site_url, credentials):
        self.site_url = site_url
        self.credentials = credentials
        self.webmasters = build('webmasters', 'v3', credentials=self.credentials)

    def fetch_keyword_data(self, start_date, end_date):
        """Fetch comprehensive keyword data from GSC."""
        request = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['query', 'device', 'country', 'searchType'],
            'rowLimit': 25000
        }
        response = self.webmasters.searchanalytics().query(siteUrl=self.site_url, body=request).execute()
        data = [{'query': row['keys'][0], 'device': row['keys'][1], 'country': row['keys'][2],
                'search_type': row['keys'][3], 'clicks': row['clicks'], 'impressions': row['impressions'],
                'ctr': row['ctr'], 'position': row['position']} for row in response['rows']]
        return pd.DataFrame(data)

    def analyze_keywords(self, df):
        """Perform advanced analysis on the keyword data."""
        # Analyze keyword performance trends
        df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
        keyword_trends = df.groupby(['query', 'month']).mean()[['clicks', 'impressions', 'ctr', 'position']].reset_index()

        # Identify high-potential, low-ranking keywords
        keyword_stats = df.groupby('query')[['clicks', 'impressions', 'ctr', 'position']].mean().reset_index()
        high_potential_keywords = keyword_stats[(keyword_stats['impressions'] > 1000) & (keyword_stats['position'] > 5)].sort_values('position')

        # Cluster keywords using K-Means
        keyword_embeddings = df.groupby('query')[['clicks', 'impressions', 'ctr', 'position']].mean().values
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(keyword_embeddings)
        clusters = KMeans(n_clusters=10, random_state=42).fit(X_reduced)
        df['keyword_cluster'] = clusters.labels_

        return keyword_trends, high_potential_keywords, df

    def generate_optimization_suggestions(self, high_potential_keywords, keyword_df):
        """Provide detailed optimization recommendations."""
        suggestions = []
        for _, row in high_potential_keywords.iterrows():
            query = row['query']
            cluster = keyword_df[keyword_df['query'] == query]['keyword_cluster'].values[0]
            cluster_keywords = keyword_df[keyword_df['keyword_cluster'] == cluster]['query'].tolist()

            # On-page SEO suggestions
            title_suggestion = f"Optimize title tag for '{query}' to better match user intent"
            meta_suggestion = f"Enhance meta description for '{query}' to improve click-through rate"
            content_suggestion = f"Expand content relevance for '{query}' and related keywords: {', '.join(cluster_keywords)}"

            # Technical SEO suggestions
            speed_suggestion = f"Improve page speed for '{query}' to boost rankings and user experience"
            mobile_suggestion = f"Ensure mobile-friendliness for '{query}' page to cater to growing mobile traffic"

            # Link building suggestions
            internal_links_suggestion = f"Improve internal linking structure to support '{query}' and related keywords"
            outreach_suggestion = f"Pursue link building opportunities to increase authority for '{query}'"

            suggestions.extend([title_suggestion, meta_suggestion, content_suggestion, speed_suggestion, mobile_suggestion, internal_links_suggestion, outreach_suggestion])

        return suggestions

def main():
    st.set_page_config(page_title="Advanced GSC Keyword Analysis Tool")
    st.title("Advanced Google Search Console Keyword Analysis")

    # Get user inputs
    site_url = st.text_input("Enter your website URL:", "https://www.example.com")
    start_date = st.date_input("Start Date:", value=datetime.now() - timedelta(days=180))
    end_date = st.date_input("End Date:", value=datetime.now())

    # Authenticate with Google Search Console
    credentials_file = st.file_uploader("Upload your Google service account credentials JSON file", type=['json'])
    if credentials_file is not None:
        credentials = service_account.Credentials.from_service_account_info(st.secrets["gsc_credentials"])
        analyzer = GscKeywordAnalyzer(site_url, credentials)

        # Fetch and analyze keyword data
        keyword_data = analyzer.fetch_keyword_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        keyword_trends, high_potential_keywords, keyword_df = analyzer.analyze_keywords(keyword_data)

        # Display keyword trends
        st.subheader("Keyword Performance Trends")
        fig, ax = plt.subplots(figsize=(12, 6))
        for column in ['clicks', 'impressions', 'ctr', 'position']:
            ax.plot(keyword_trends['month'], keyword_trends[column], label=column)
        ax.set_xlabel('Month')
        ax.legend()
        st.pyplot(fig)

        # Display optimization suggestions
        st.subheader("Optimization Suggestions")
        suggestions = analyzer.generate_optimization_suggestions(high_potential_keywords, keyword_df)
        for suggestion in suggestions:
            st.write(suggestion)

        # Display keyword cluster visualization
        st.subheader("Keyword Cluster Visualization")
        fig, ax = plt.subplots(figsize=(8, 8))
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(keyword_embeddings)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters.labels_)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
