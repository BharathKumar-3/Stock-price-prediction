import streamlit as st
import prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from jugaad_data.nse import stock_df, bhavcopy_save
from datetime import date
import pandas as pd
import yfinance as yf

#  SETTING TODAY DATE
today = date.today()
today_date = today.strftime("%d/%m/%Y").split("/")
YEAR = int(today_date[2])
DATE1 = int(today_date[0])
MONTH = int(today_date[1])
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


def plot_raw_data(stock_date, stock_open, stock_close):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[stock_date], y=df[stock_open], name="stock_open"))
    fig.add_trace(go.Scatter(x=df[stock_date], y=df[stock_close], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


def indian_stock():
    data_file = pd.read_csv("indian stock data/cm14Jul2022bhav.csv")
    ind_stock = data_file["SYMBOL"].tolist()
    selected_stock = st.selectbox("Select stock Symbol", ind_stock, key="<stock_select>")
    data = stock_df(symbol=selected_stock, from_date=date(2018, 1, 1),
                    to_date=date(YEAR, MONTH, DATE1), series="EQ")
    return data


def us_stock():
    us_stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'TSLA')
    selected_stock = st.selectbox('Select dataset for prediction', us_stocks, key="<us_stock>")
    return load_data(selected_stock)


def crypto_stock():
    crypto_stocks = ('BTC-USD', 'ETH-USD', 'USDT-USD', 'USDC-USD', 'BNB-USD', 'BUSD-USD', 'XRP-USD')
    selected_stock = st.selectbox('Select dataset for prediction', crypto_stocks, key="<us_stock>")
    return load_data(selected_stock)


# getting stock name and prediction year from user
stock = ["Crypto currency", "Indian Stock", "US Stock"]
st.title('Stock Forecast App')
selected_stock_database = st.selectbox('Select dataset for prediction', stock, key="<database>")

if selected_stock_database == stock[0]:
    df = crypto_stock()
elif selected_stock_database == stock[1]:
    df = indian_stock()
elif selected_stock_database == stock[2]:
    df = us_stock()

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# displaying raw_data

st.subheader('Raw data')
st.write(df.tail(20))


# plotting raw data
if selected_stock_database == stock[1]:
    plot_raw_data(stock_date='DATE', stock_open='OPEN', stock_close='CLOSE')
else:
    plot_raw_data(stock_date='Date', stock_open='Open', stock_close='Close')

# PREDICTION
if selected_stock_database == stock[1]:
    df_train = df[['DATE', 'CLOSE']]
    df_train = df_train.rename(columns={"DATE": "ds", "CLOSE": "y"})
else:
    df_train = df[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


m = prophet.Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

bitcoin_symbol = [""]
