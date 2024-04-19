import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.graph_objects as go

# App title
st.title('Stock Market Analysis ')

# Load data
def load_data():
    df = pd.read_csv('./milestone3/df_merge.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
df2 = pd.read_csv('./milestone3/df_merge2.csv')
# Setup pages
page = st.sidebar.selectbox("Choose a page", ["Project Overview", "Data Visualization", "Stock Price Prediction"])

if page == "Project Overview":
    st.header("Project Background and Scope")
    st.write("""
    This project integrates three key datasets: Tesla's historical trading data, financial statements, and shareholder comments. 
    The goals are to:
    - Analyze Tesla's stock performance and market volatility.
    - Evaluate financial health and operational efficiency.
    - Assess shareholder sentiments and their impact on stock prices.
    **Questions and Answers**
    
    1.Mengzhen Lian
    
    2.Stock price trend chart (opening price and closing price): Use K-line chart to display the price dynamics of Tesla stock.
    Correlation heat map between variables: Shows the correlation between various financial and sentiment indicators to help users understand which factors may affect each other.
    Regression Scatter Plot of Sentiment Value vs. Closing Price: Exploring the relationship between shareholder sentiment (sentiment scores derived from comment analysis) and stock closing prices.
    3D scatter plot: shows the three-dimensional relationship between long-term debt, positive sentiment and stock price, and further analyzes the impact of financial conditions and market sentiment on stock price.
    Users can select different pages to view specific data views through the sidebar.
    On the data visualization page, users can intuitively see various charts, such as K-line charts, heat maps, and scatter charts, etc. These charts provide visual correlations between stock prices and company financial/sentiment indicators.
    On the stock price prediction page, users can see the comparison between the stock price predicted by the model and the actual stock price, as well as the model's performance indicators (such as RMSE and R²).
    in conclusion Preliminary analysis shows a positive correlation between positive shareholder sentiment and stock price, while a company's operational efficiency (such as financial health) has a complex impact on stock price that may vary based on other variables not considered. The LSTM model demonstrates certain predictive capabilities, but requires further optimization to improve accuracy and reduce prediction errors.
    
    3.Data processing and synchronization: Aligning dates and handling missing data are major challenges when merging data sets from three different sources. Different data frequencies and ranges can lead to data synchronization issues requiring precise pre-processing steps.
    
    4. **What is the aim?**  
       The aim is to understand how Tesla's stock prices are influenced by its financial health and market sentiments.
    5. **What did you find?**  
       Initial analyses suggest that positive sentiments correlate with higher stock prices, whereas operational efficiencies show varied impacts.
    6. **Challenges Encountered?**  
       Handling large datasets and integrating different types of data were challenging, especially ensuring data quality and alignment.
    7. **Skills Wished?**  
       Skills in data processing, machine learning, and sentiment analysis were crucial and improved over the course of this project.
    8. **Next Steps?**  
       Future steps include refining the models, incorporating more granular data, and potentially using more advanced machine learning techniques.
    """)


if page == "Data Visualization":
    st.header("Stock Price Trend (Open and Close Prices)")
    
    # Plot candlestick chart
    fig = make_subplots(rows=2, cols=1)

    trace1 = go.Candlestick(x=df['Date'],
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close*'],
                            name='Candlestick Chart')
    fig.append_trace(trace1, row=1, col=1)

    fig.update_layout(height=600, width=800, title_text="Candlestick Chart for Stock Prices")
    st.plotly_chart(fig)

    st.header("Correlation Heatmap Among Variables")
    # Calculate correlation matrix
    corr = df[['Open', 'High', 'Low', 'Close*', 'Volume', 'neu', 'pos', 'neg', 'compound']].corr()
    # Plot heatmap
    fig2 = px.imshow(corr, text_auto=True, aspect="auto",
                     labels=dict(x="Feature", y="Feature", color="Correlation"),
                     x=corr.columns, y=corr.columns)
    st.plotly_chart(fig2)

    st.header("Regression Scatter Plot Between Sentiment and Closing Price")
    # Plot regression scatter plot
    fig3 = px.scatter(df, x='compound', y='Close*', trendline="ols",
                      labels={'compound': 'Sentiment Score', 'Close*': 'Closing Price'})
    st.plotly_chart(fig3)
    if {'Close*', 'currentLongTermDebt', 'pos'}.issubset(df2.columns):
        df2[['Close*', 'currentLongTermDebt', 'pos']]=df2[['Close*', 'currentLongTermDebt', 'pos']].astype('float').dropna()
        fig = px.scatter_3d(df2, x='currentLongTermDebt', y='pos', z='Close*',
                            color='pos', labels={
                "Operational Efficiency": "Operational Efficiency",
                "Positive Sentiment": "Positive Sentiment (pos)",
                "Stock Price": "Stock Price"
            })
        st.plotly_chart(fig)



elif page == "Stock Price Prediction":
    st.header("Stock Price Prediction Model Using LSTM")

    # 数据标准化

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    df_features = df[['Open', 'Close*', 'pos', 'neg']]
    df_target = df[['Close*']]

    scaled_features = scaler_x.fit_transform(df_features)
    scaled_target = scaler_y.fit_transform(df_target)


    # Adjusted dataset creation to use separate scalers
    def create_dataset(features, target, time_steps=1):
        Xs, ys = [], []
        for i in range(len(features) - time_steps):
            v = features[i:(i + time_steps), :]
            Xs.append(v)
            ys.append(target[i + time_steps])  # target is now a single column
        return np.array(Xs), np.array(ys)


    TIME_STEPS = 3
    X, y = create_dataset(scaled_features, scaled_target.flatten(), TIME_STEPS)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(TIME_STEPS, X.shape[2]), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, shuffle=False)

    # Prediction
    y_pred = model.predict(X_test)

    # Correct inverse scaling
    y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Calculate RMSE and R2
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)

    st.write(f"RMSE: {rmse}")
    st.write(f"R²: {r2}")

    # Plotting actual vs predicted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y_test_inv)), y=y_test_inv.flatten(), mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_pred_inv)), y=y_pred_inv.flatten(), mode='lines', name='Predicted'))
    fig.update_layout(title="Actual vs Predicted Closing Prices", xaxis_title="Index", yaxis_title="Price")
    st.plotly_chart(fig)


