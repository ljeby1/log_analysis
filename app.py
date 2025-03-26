import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load and Preprocess Data
def load_and_clean_data(file_path):
    # Load the CSV file (adjust path as needed)
    df = pd.read_csv(file_path)
    
    # Handle timestamps if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Clean text: lowercase, remove special characters
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    
    stop_words = set(stopwords.words('english'))
    def tokenize_text(text):
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in stop_words]
    
    df['cleaned_description'] = df['description'].apply(clean_text)
    df['tokens'] = df['cleaned_description'].apply(tokenize_text)
    return df

# Step 2: Feature Extraction and Clustering
def analyze_patterns(df):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_description'])
    feature_names = vectorizer.get_feature_names_out()
    
    # Cluster incidents using K-Means
    num_clusters = 5  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    # Extract top terms per cluster
    print("\n=== Cluster Analysis ===")
    for i in range(num_clusters):
        cluster_docs = df[df['cluster'] == i]['cleaned_description']
        print(f"Cluster {i} (size: {len(cluster_docs)}):")
        centroid = kmeans.cluster_centers_[i]
        top_indices = centroid.argsort()[-5:][::-1]  # Top 5 terms
        top_terms = [feature_names[idx] for idx in top_indices]
        print(f"Top terms: {top_terms}")
        print(f"Sample: {cluster_docs.iloc[0]}\n")
    
    # Top overall keywords
    tfidf_sums = tfidf_matrix.sum(axis=0).A1
    top_n = 10
    top_indices = tfidf_sums.argsort()[-top_n:][::-1]
    top_keywords = [feature_names[idx] for idx in top_indices]
    print(f"Top {top_n} keywords across all incidents: {top_keywords}")
    
    return df, tfidf_matrix

# Step 3: Visualization
def visualize_results(df):
    # Plot cluster sizes
    cluster_counts = df['cluster'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar')
    plt.title('Number of Incidents per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.show()
    
    # Plot incidents over time (if timestamps exist)
    if 'timestamp' in df.columns and not df['timestamp'].isna().all():
        plt.figure(figsize=(12, 6))
        df.groupby([df['timestamp'].dt.date, 'cluster']).size().unstack().plot()
        plt.title('Cluster Frequency Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Cluster')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Specify your file path here
    file_path = 'incident_log.csv'  # Replace with your actual file
    
    # Run the analysis
    print("Loading and cleaning data...")
    df = load_and_clean_data(file_path)
    
    print("Analyzing patterns...")
    df, tfidf_matrix = analyze_patterns(df)
    
    print("Generating visualizations...")
    visualize_results(df)
    
    # Save results to CSV
    df[['description', 'cluster'] + (['timestamp'] if 'timestamp' in df.columns else [])].to_csv('clustered_incidents.csv', index=False)
    print("Results saved to 'clustered_incidents.csv'")
