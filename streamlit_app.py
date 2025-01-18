import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score

# Step 1: Load and Clean Data
def load_stock_data(ticker):
    st.write(f"Downloading stock data for: {ticker}")
    stock_data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    stock_data.reset_index(inplace=True)
    return stock_data

def clean_data(df):
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# Step 2: EDA Functions
def summary_statistics(data_cleaned):
    st.subheader("Summary Statistics")
    st.write(data_cleaned.describe())

def plot_distributions(data_cleaned):
    st.subheader("Feature Distributions")
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data_cleaned[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        st.pyplot(plt)

def plot_correlation_matrix(data_cleaned):
    st.subheader("Correlation Matrix")
    correlation = data_cleaned.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    st.pyplot(plt)

# Step 2: Plotting Functions
def plot_closing_trend(data_cleaned):
    # Ensure Date and Close are one-dimensional
    dates = data_cleaned['Date'].values.flatten()  # Ensure 1D array
    close_prices = data_cleaned['Close'].values.flatten()  # Ensure 1D array

    # Plot the closing price trend
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=dates, y=close_prices)
    plt.title('Stock Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    st.pyplot(plt)

def plot_elbow_method(X_scaled):
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    st.pyplot(plt)

# Step 3: Main Function
def main():
    st.title("Stock Price Analysis and Modeling")

    # Sidebar
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

    # Load and clean data
    data = load_stock_data(ticker)
    data_cleaned = clean_data(data)

    # Display raw data
    st.subheader("Raw Data")
    st.write(data_cleaned.head())

    # EDA
    st.subheader("Exploratory Data Analysis (EDA)")
    summary_statistics(data_cleaned)
    plot_distributions(data_cleaned)
    plot_correlation_matrix(data_cleaned)

    # Plot closing price trends
    st.subheader("Stock Closing Price Over Time")
    plot_closing_trend(data_cleaned)

    # Feature Engineering
    data_cleaned['Daily Return'] = data_cleaned['Close'].pct_change()
    data_cleaned.dropna(inplace=True)

    X = data_cleaned[['Open', 'High', 'Low', 'Volume', 'Daily Return']]
    y = (data_cleaned['Close'] > data_cleaned['Open']).astype(int)  # Binary classification

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Classification (Random Forest)
    st.subheader("Classification: Random Forest")
    clf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Classification Results
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # Clustering (KMeans)
    st.subheader("Clustering: KMeans")
    plot_elbow_method(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

# Run the app
if __name__ == "__main__":
    main()
