import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
import os
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .success-metric {
        font-size: 1.2rem;
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-metric {
        font-size: 1.2rem;
        color: #FF9800;
        font-weight: bold;
    }
    .error-metric {
        font-size: 1.2rem;
        color: #F44336;
        font-weight: bold;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
        border-radius: 5px;
    }
    .stock-info {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .info-box {
        flex: 1;
        min-width: 150px;
        padding: 15px;
        margin: 5px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .info-label {
        font-size: 0.9rem;
        color: #757575;
        margin-bottom: 5px;
    }
    .info-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #212121;
    }
    .up-value {
        color: #4CAF50;
    }
    .down-value {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Stock Market Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='card'>
    <p>This application uses Long Short-Term Memory (LSTM) networks and technical indicators to predict stock market prices. 
    The model analyzes historical data and market indicators like MACD, OBV, and ROC to improve prediction accuracy.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("<h2 style='text-align: center;'>Parameters</h2>", unsafe_allow_html=True)

# Stock ticker input
ticker = st.sidebar.text_input("Stock Symbol (e.g., AAPL, MSFT)", "AAPL")

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Model parameters
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center;'>Model Parameters</h3>", unsafe_allow_html=True)
lookback = st.sidebar.slider("Lookback Period (Days)", 30, 200, 60)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
prediction_days = st.sidebar.slider("Prediction Horizon (Days)", 7, 90, 30)

# Feature selection
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center;'>Features</h3>", unsafe_allow_html=True)
use_macd = st.sidebar.checkbox("Include MACD", value=True)
use_obv = st.sidebar.checkbox("Include OBV", value=True)
use_roc = st.sidebar.checkbox("Include ROC", value=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Analysis", "Model Training", "Predictions"])


# Function to get stock data
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if len(df) > 0:
            return df
        else:
            st.error(f"No data found for {ticker}. Please check the ticker symbol.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Calculate technical indicators
def add_technical_indicators(df):
    # Make sure we're working with a copy to avoid modifying the original
    df = df.copy()

    # Convert any Series that might be DataFrames to ensure they're 1D
    close_series = df['Close'].values if hasattr(df['Close'], 'values') else df['Close']
    volume_series = df['Volume'].values if hasattr(df['Volume'], 'values') else df['Volume']

    # Make sure they're 1D arrays
    if len(close_series.shape) > 1:
        close_series = close_series.flatten()
    if len(volume_series.shape) > 1:
        volume_series = volume_series.flatten()

    # Create a new DataFrame with correct 1D series
    # This avoids any issues with the ta library expecting 1D data
    temp_df = pd.DataFrame({
        'close': close_series,
        'volume': volume_series
    }, index=df.index)

    # MACD
    macd_ind = ta.trend.MACD(temp_df['close'])
    df['macd'] = macd_ind.macd_diff()

    # On-Balance Volume (OBV)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(temp_df['close'], temp_df['volume']).on_balance_volume()

    # Rate of Change (ROC)
    df['roc'] = ta.momentum.ROCIndicator(temp_df['close'], window=12).roc()

    # Additional indicators
    df['rsi'] = ta.momentum.RSIIndicator(temp_df['close'], window=14).rsi()
    df['ema_9'] = ta.trend.EMAIndicator(temp_df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(temp_df['close'], window=21).ema_indicator()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(temp_df['close'])
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()

    # Remove NaN values
    return df.dropna()


# Prepare data for LSTM
def prepare_data(df, lookback, features_list):
    data = df[features_list].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 0])  # Predict Close price

    X, y = np.array(X), np.array(y)

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler, data


# Build and train LSTM model
def build_lstm_model(X_train, y_train, X_test, y_test, epochs):
    # Model architecture based on the paper
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping and model checkpoint
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model with progress bar
    progress_bar = st.progress(0)
    history = []

    for i in range(epochs):
        h = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test),
                      verbose=0, callbacks=[early_stop])
        history.append(h.history)
        progress_bar.progress((i + 1) / epochs)

    return model, history


# Make predictions
def predict_future(model, X_test, scaler, data, lookback, prediction_days, features_count):
    # First, predict on test data
    predictions = model.predict(X_test)

    # Then predict future days
    last_sequence = X_test[-1].copy()
    future_predictions = []

    for _ in range(prediction_days):
        next_pred = model.predict(np.array([last_sequence]))
        future_predictions.append(next_pred[0, 0])

        # Update sequence for next prediction
        new_row = np.zeros((1, features_count))
        new_row[0, 0] = next_pred[0, 0]  # Set predicted close price
        # We're not predicting other features, so they stay as 0

        last_sequence = np.append(last_sequence[1:], new_row, axis=0)

    # Inverse transform to get actual prices
    dummy_array = np.zeros((len(future_predictions), features_count))
    dummy_array[:, 0] = future_predictions
    future_predictions = scaler.inverse_transform(dummy_array)[:, 0]

    return future_predictions


# Evaluate model performance
def evaluate_model(y_test, test_predictions):
    mse = mean_squared_error(y_test, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    return mse, rmse, mae, r2


# Main function
def main():
    # Load data
    df = load_data(ticker, start_date, end_date)

    if df is None:
        return

    # Add technical indicators
    df_with_indicators = add_technical_indicators(df.copy())

    # Data Analysis Tab
    with tab1:
        st.markdown("<h2 class='sub-header'>Historical Data Analysis</h2>", unsafe_allow_html=True)

        # Stock information card
        latest_data = df.iloc[-1]
        prev_data = df.iloc[-2]
        price_change = latest_data['Close'] - prev_data['Close']
        price_change_pct = (price_change / prev_data['Close']) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"<h3>{ticker} - {latest_data.name.date()}</h3>", unsafe_allow_html=True)

            st.markdown("""
            <div class='stock-info'>
                <div class='info-box'>
                    <div class='info-label'>Open</div>
                    <div class='info-value'>${latest_data['Open']:.2f}</div>
                </div>
                <div class='info-box'>
                    <div class='info-label'>High</div>
                    <div class='info-value'>${latest_data['High']:.2f}</div>
                </div>
                <div class='info-box'>
                    <div class='info-label'>Low</div>
                    <div class='info-value'>${latest_data['Low']:.2f}</div>
                </div>
                <div class='info-box'>
                    <div class='info-label'>Close</div>
                    <div class='info-value ${("up-value" if price_change >= 0 else "down-value")}'>${latest_data['Close']:.2f}</div>
                </div>
                <div class='info-box'>
                    <div class='info-label'>Change</div>
                    <div class='info-value ${("up-value" if price_change >= 0 else "down-value")}'>
                        {price_change:.2f} ({price_change_pct:.2f}%)
                    </div>
                </div>
                <div class='info-box'>
                    <div class='info-label'>Volume</div>
                    <div class='info-value'>{latest_data['Volume']:,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Historical price chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.03,
                            subplot_titles=('Price History', 'Volume'))

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add volume chart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(30, 136, 229, 0.5)'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{ticker} Historical Price and Volume',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            height=600,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Technical Indicators Analysis
        st.markdown("<h3 class='sub-header'>Technical Indicators</h3>", unsafe_allow_html=True)

        # Create tabs for different technical indicators
        indicator_tabs = st.tabs(["MACD & RSI", "OBV & ROC", "Moving Averages", "Bollinger Bands"])

        with indicator_tabs[0]:
            # MACD and RSI plot
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=('MACD', 'RSI'))

            # Plot MACD
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['macd'],
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )

            # Plot RSI
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['rsi'],
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )

            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig.update_layout(height=500, template='plotly_white')
            fig.update_yaxes(title_text="MACD", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

        with indicator_tabs[1]:
            # OBV and ROC plot
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=('On-Balance Volume (OBV)', 'Rate of Change (ROC)'))

            # Plot OBV
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['obv'],
                    name='OBV',
                    line=dict(color='darkgreen')
                ),
                row=1, col=1
            )

            # Plot ROC
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['roc'],
                    name='ROC',
                    line=dict(color='darkorange')
                ),
                row=2, col=1
            )

            # Add zero line for ROC
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

            fig.update_layout(height=500, template='plotly_white')
            fig.update_yaxes(title_text="OBV", row=1, col=1)
            fig.update_yaxes(title_text="ROC", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

        with indicator_tabs[2]:
            # Moving Averages plot
            fig = go.Figure()

            # Plot Close price
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['Close'],
                    name='Close',
                    line=dict(color='black')
                )
            )

            # Plot EMAs
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['ema_9'],
                    name='EMA 9',
                    line=dict(color='blue')
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['ema_21'],
                    name='EMA 21',
                    line=dict(color='red')
                )
            )

            fig.update_layout(
                title='Price with Moving Averages',
                yaxis_title='Price (USD)',
                height=500,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

        with indicator_tabs[3]:
            # Bollinger Bands plot
            fig = go.Figure()

            # Plot Close price
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['Close'],
                    name='Close',
                    line=dict(color='black')
                )
            )

            # Plot Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['bollinger_high'],
                    name='Upper Band',
                    line=dict(color='green')
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['bollinger_low'],
                    name='Lower Band',
                    line=dict(color='red')
                )
            )

            # Fill area between bands
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['bollinger_high'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['bollinger_low'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    fillcolor='rgba(0, 176, 246, 0.2)',
                    showlegend=False
                )
            )

            fig.update_layout(
                title='Bollinger Bands',
                yaxis_title='Price (USD)',
                height=500,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

    # Model Training Tab
    with tab2:
        st.markdown("<h2 class='sub-header'>LSTM Model Training</h2>", unsafe_allow_html=True)

        # Select features for model
        features = ['Close']
        if use_macd:
            features.append('macd')
        if use_obv:
            features.append('obv')
        if use_roc:
            features.append('roc')

        st.markdown(f"<p>Selected features: {', '.join(features)}</p>", unsafe_allow_html=True)

        # Display training progress
        if st.button("Train Model"):
            with st.spinner("Preparing data..."):
                # Prepare data
                X_train, X_test, y_train, y_test, scaler, data = prepare_data(
                    df_with_indicators, lookback, features)

                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                st.session_state.data = data
                st.session_state.features_count = len(features)

            st.markdown("<p>Training LSTM model...</p>", unsafe_allow_html=True)

            # Train model
            model, history = build_lstm_model(X_train, y_train, X_test, y_test, epochs)
            st.session_state.model = model

            # Make predictions on test data
            test_predictions = model.predict(X_test)

            # Inverse transform predictions
            dummy_array = np.zeros((len(test_predictions), len(features)))
            dummy_array[:, 0] = test_predictions.flatten()
            test_predictions_actual = scaler.inverse_transform(dummy_array)[:, 0]

            # Inverse transform actual test values
            dummy_array = np.zeros((len(y_test), len(features)))
            dummy_array[:, 0] = y_test
            y_test_actual = scaler.inverse_transform(dummy_array)[:, 0]

            # Store in session state
            st.session_state.test_predictions = test_predictions_actual
            st.session_state.y_test_actual = y_test_actual

            # Evaluate model
            mse, rmse, mae, r2 = evaluate_model(y_test_actual, test_predictions_actual)

            # Display metrics
            st.markdown("<h3 class='sub-header'>Model Performance</h3>", unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f"<div class='card'><p>MSE</p><p class='{'success-metric' if mse < 10 else 'warning-metric' if mse < 50 else 'error-metric'}'>{mse:.4f}</p></div>",
                    unsafe_allow_html=True)
            with col2:
                st.markdown(
                    f"<div class='card'><p>RMSE</p><p class='{'success-metric' if rmse < 5 else 'warning-metric' if rmse < 10 else 'error-metric'}'>{rmse:.4f}</p></div>",
                    unsafe_allow_html=True)
            with col3:
                st.markdown(
                    f"<div class='card'><p>MAE</p><p class='{'success-metric' if mae < 3 else 'warning-metric' if mae < 7 else 'error-metric'}'>{mae:.4f}</p></div>",
                    unsafe_allow_html=True)
            with col4:
                st.markdown(
                    f"<div class='card'><p>R²</p><p class='{'success-metric' if r2 > 0.8 else 'warning-metric' if r2 > 0.5 else 'error-metric'}'>{r2:.4f}</p></div>",
                    unsafe_allow_html=True)

            # Plot predictions vs actual
            fig = go.Figure()

            # Actual test values
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index[-len(y_test_actual):],
                    y=y_test_actual,
                    name='Actual',
                    line=dict(color='blue')
                )
            )

            # Predicted values
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index[-len(test_predictions_actual):],
                    y=test_predictions_actual,
                    name='Predicted',
                    line=dict(color='red')
                )
            )

            fig.update_layout(
                title='Model Performance: Predicted vs Actual Prices',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                height=500,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            st.success("Model training completed successfully! Go to the Predictions tab to see future forecasts.")

    # Predictions Tab
    with tab3:
        st.markdown("<h2 class='sub-header'>Future Price Predictions</h2>", unsafe_allow_html=True)

        if 'model' not in st.session_state:
            st.warning("Please train the model first in the Model Training tab.")
        else:
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    # Get future predictions
                    future_predictions = predict_future(
                        st.session_state.model,
                        st.session_state.X_test,
                        st.session_state.scaler,
                        st.session_state.data,
                        lookback,
                        prediction_days,
                        st.session_state.features_count
                    )

                    # Generate future dates
                    last_date = df_with_indicators.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)

                    # Create a dataframe for future predictions
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Close': future_predictions
                    })

                    # Plot future predictions along with historical data
                    fig = go.Figure()

                    # Historical close prices
                    fig.add_trace(
                        go.Scatter(
                            x=df_with_indicators.index,
                            y=df_with_indicators['Close'],
                            name='Historical Close',
                            line=dict(color='blue')
                        )
                    )

                    # Test predictions
                    fig.add_trace(
                        go.Scatter(
                            x=df_with_indicators.index[-len(st.session_state.test_predictions):],
                            y=st.session_state.test_predictions,
                            name='Test Predictions',
                            line=dict(color='green', dash='dash')
                        )
                    )

                    # Future predictions
                    fig.add_trace(
                        go.Scatter(
                            x=future_df['Date'],
                            y=future_df['Predicted_Close'],
                            name='Future Predictions',
                            line=dict(color='red')
                        )
                    )

                    # Add confidence interval
                    # Calculate standard deviation of test prediction errors
                    test_errors = st.session_state.y_test_actual - st.session_state.test_predictions
                    std_dev = np.std(test_errors)

                    # Create upper and lower confidence bounds (95% confidence interval)
                    upper_bound = future_df['Predicted_Close'] + 1.96 * std_dev
                    lower_bound = future_df['Predicted_Close'] - 1.96 * std_dev

                    # Add confidence interval to plot
                    fig.add_trace(
                        go.Scatter(
                            x=future_df['Date'],
                            y=upper_bound,
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=future_df['Date'],
                            y=lower_bound,
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            showlegend=False
                        )
                    )

                    fig.update_layout(
                        title='Stock Price Prediction with 95% Confidence Interval',
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        height=600,
                        template='plotly_white',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    # Add vertical separator between historical and predicted data
                    fig.add_vline(x=last_date, line_dash="dash", line_color="gray")

                    st.plotly_chart(fig, use_container_width=True)

                    # Display prediction results in a table
                    st.markdown("<h3 class='sub-header'>Predicted Prices</h3>", unsafe_allow_html=True)

                    # Format the prediction dataframe for display
                    display_df = future_df.copy()
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                    display_df['Predicted_Close'] = display_df['Predicted_Close'].round(2)
                    display_df['Lower Bound (95%)'] = lower_bound.round(2)
                    display_df['Upper Bound (95%)'] = upper_bound.round(2)

                    # Calculate price change and percentage
                    prev_close = df_with_indicators['Close'].iloc[-1]
                    display_df['Change ($)'] = (display_df['Predicted_Close'] - prev_close).round(2)
                    display_df['Change (%)'] = ((display_df['Predicted_Close'] - prev_close) / prev_close * 100).round(
                        2)

                    # Add color to positive and negative changes
                    def color_change(val):
                        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                        return f'color: {color}'

                    # Style the dataframe
                    styled_df = display_df.style.applymap(color_change, subset=['Change ($)', 'Change (%)'])

                    # Display the table
                    st.dataframe(styled_df, use_container_width=True)

                    # Summary statistics
                    st.markdown("<h3 class='sub-header'>Prediction Summary</h3>", unsafe_allow_html=True)

                    # Calculate trend and statistics
                    trend = "Upward" if future_predictions[-1] > future_predictions[0] else "Downward"
                    max_price = np.max(future_predictions)
                    min_price = np.min(future_predictions)
                    avg_price = np.mean(future_predictions)
                    final_change = ((future_predictions[-1] - prev_close) / prev_close) * 100

                    # Display summary
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"""
                        <div class='card'>
                            <p>Overall Trend</p>
                            <p class='{'success-metric' if trend == 'Upward' else 'error-metric'}'>{trend}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class='card'>
                            <p>Max Price</p>
                            <p class='success-metric'>${max_price:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class='card'>
                            <p>Min Price</p>
                            <p class='{'warning-metric' if min_price < prev_close else 'success-metric'}'>${min_price:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"""
                        <div class='card'>
                            <p>Predicted Change</p>
                            <p class='{'success-metric' if final_change > 0 else 'error-metric'}'>{final_change:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Recommendation
                    st.markdown("<h3 class='sub-header'>Investment Recommendation</h3>", unsafe_allow_html=True)

                    # Generate recommendation based on predicted trend
                    if final_change > 5:
                        recommendation = "Strong Buy"
                        desc = f"Our model predicts a significant price increase of {final_change:.2f}% over the next {prediction_days} days."
                        color = "success-metric"
                    elif final_change > 2:
                        recommendation = "Buy"
                        desc = f"Our model predicts a moderate price increase of {final_change:.2f}% over the next {prediction_days} days."
                        color = "success-metric"
                    elif final_change > -2:
                        recommendation = "Hold"
                        desc = f"Our model predicts relatively stable prices with a change of {final_change:.2f}% over the next {prediction_days} days."
                        color = "warning-metric"
                    elif final_change > -5:
                        recommendation = "Sell"
                        desc = f"Our model predicts a moderate price decrease of {final_change:.2f}% over the next {prediction_days} days."
                        color = "error-metric"
                    else:
                        recommendation = "Strong Sell"
                        desc = f"Our model predicts a significant price decrease of {final_change:.2f}% over the next {prediction_days} days."
                        color = "error-metric"

                    st.markdown(f"""
                    <div class='card'>
                        <h4 class='{color}'>{recommendation}</h4>
                        <p>{desc}</p>
                        <p><small>This recommendation is based solely on our predictive model and should not be considered as financial advice. Always consult with a financial advisor before making investment decisions.</small></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Disclaimer
                    st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;'>
                        <p><strong>Disclaimer:</strong> The predictions generated by this application are based on historical data and technical indicators only. 
                        They should not be considered as financial advice. Always do your own research and consult with a professional financial advisor before making investment decisions. 
                        Past performance is not indicative of future results.</p>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()