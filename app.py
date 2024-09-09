import streamlit as st
import pandas as pd
from helper import train_stock_model, forecast_stock_model

def main():
    st.title('Stock Price Forecasting App 📈💰🤑')

    # Inputs from the user
    ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2024-09-01'))
    forecast_days = st.sidebar.slider('Days to Predict', min_value=1, max_value=365, value=180)

    # Initialize session state for training success
    if "training_successful" not in st.session_state:
        st.session_state.training_successful = False

    # Train Button
    if st.sidebar.button('Train Prophet Model'):
        try:
            with st.spinner('Training Prophet model...'):
                st.session_state.training_successful = train_stock_model(ticker=ticker, start_date=start_date, end_date=end_date, forecast_periods=forecast_days)
                if st.session_state.training_successful:
                    st.success("Training Complete!")
                else:
                    st.error("Training Failed. Please check the logs.")
        except Exception as e:
            st.error(f"An error occurred during training: {str(e)}")

    # Forecast Button (enabled only after training is successful)
    if st.session_state.training_successful:
        if st.sidebar.button('Generate Forecast'):
            try:
                with st.spinner('Generating forecast...'):
                    forecast_img, components_img = forecast_stock_model(ticker=ticker, forecast_periods=forecast_days)
                    
                    if forecast_img:
                        st.markdown(f"**Stock Price Forecast for {ticker}**") 
                        st.image(forecast_img, caption=f'Stock Price Forecast for {ticker}', use_column_width=True)
                        st.success("Forecast Generated Successfully!")
                    else:
                        st.error("Failed to generate forecast plot.")
            except Exception as e:
                st.error(f"An error occurred during forecasting: {str(e)}")

        if st.sidebar.button('Show Forecast Components'):
            try:
                with st.spinner('Generating forecast components...'):
                    forecast_img, components_img = forecast_stock_model(ticker=ticker, forecast_periods=forecast_days)
                    
                    if components_img:
                        st.markdown(f"**Stock Forecast Components for {ticker}**")
                        st.image(components_img, caption=f'Forecast Components Plot for {ticker}', use_column_width=True)
                        st.success("Components Plot Generated Successfully!")
                    else:
                        st.error("Failed to generate components plot.")
            except Exception as e:
                st.error(f"An error occurred during components plotting: {str(e)}")
    else:
        st.sidebar.warning("Please train the model first.")


# Footer
st.markdown("""
<style>
.developer-label {
    position: fixed;
    bottom: 0;
    width: calc(100% - var(--sidebar-width, 0px)); /* Adjust width based on sidebar */
    text-align: center;
    background-color: #f0f0f0;
    padding: 5px 10px;
    border-top: 1px solid #ddd;
    left: var(--sidebar-width, 0px); /* Adjust position based on sidebar */
}
</style>
<div class="developer-label">
    <p>Developed by Kousik Naskar | Email: <a href="mailto:kousik23naskar@gmail.com">kousik23naskar@gmail.com</a></p>
</div>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()