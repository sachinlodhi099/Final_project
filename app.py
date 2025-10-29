import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta

# Streamlit title
st.title("📈 Stock Price Prediction using Simple RNN")
st.write("Predict next 10 days of stock prices using a Recurrent Neural Network (RNN).")

# File uploader
uploaded_file = st.file_uploader("Upload Stock CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Ensure 'Date' and 'Adj Close' exist
    if 'Date' not in df.columns or 'Adj Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Adj Close' columns.")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        st.line_chart(df['Adj Close'], use_container_width=True)

        # Data scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Adj Close']])

        # Parameters
        n_steps = st.slider("Select lookback window (days)", 30, 120, 60)
        future_days = 10
        epochs = st.number_input("Epochs", 10, 200, 50)
        batch_size = st.number_input("Batch size", 8, 128, 32)

        # Prepare sequences
        X, y = [], []
        for i in range(n_steps, len(scaled_data)):
            X.append(scaled_data[i - n_steps:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split data (80% train, 20% test)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build model
        model = Sequential([
            SimpleRNN(50, activation='tanh', return_sequences=False, input_shape=(n_steps, 1)),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # EarlyStopping
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        with st.spinner("Training model... ⏳"):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es],
                verbose=0
            )

        st.success("✅ Training complete!")

        # Plot training history
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Val Loss')
        ax.set_title("Model Loss Over Epochs")
        ax.legend()
        st.pyplot(fig)

        # Prediction (next 10 days)
        last_sequence = scaled_data[-n_steps:]
        predictions = []
        for _ in range(future_days):
            X_pred = last_sequence.reshape((1, n_steps, 1))
            pred = model.predict(X_pred, verbose=0)
            predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred)

        # Inverse transform predictions
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(future_days)]

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": predicted_prices.flatten()
        })

        st.write("### 📅 Next 10-Day Forecast")
        st.dataframe(forecast_df)

        # Plot predictions
        fig2, ax2 = plt.subplots()
        ax2.plot(df.index, df['Adj Close'], label='Historical')
        ax2.plot(forecast_df['Date'], forecast_df['Predicted Price'], 'r--', label='Forecast')
        ax2.legend()
        ax2.set_title("Stock Price Forecast (Next 10 Days)")
        st.pyplot(fig2)
