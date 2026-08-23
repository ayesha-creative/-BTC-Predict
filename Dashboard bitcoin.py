
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# LIGHTGBM
# =========================================================

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BTC Predict",
    page_icon="₿",
    layout="wide"
)


# =========================================================
# CUSTOM GREEN + BLUE + WHITE THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #061A18 0%,
        #072B3D 55%,
        #0A3D4A 100%
    );
    color: white;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #05231F 0%,
        #073B4C 100%
    );

    border-right: 1px solid #16C79A;
}

[data-testid="stSidebar"] * {
    color: white !important;
}


/* HEADINGS */

h1 {
    color: white !important;
    font-weight: 800;
}

h2, h3 {
    color: #EFFFFB !important;
}


/* TEXT */

p {
    color: #D9F5EF !important;
}


/* METRIC CARDS */

[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #073B4C,
        #075E54
    );

    border: 1px solid #16C79A;

    border-radius: 18px;

    padding: 20px;

    box-shadow:
        0px 8px 25px rgba(0, 0, 0, 0.25);
}

[data-testid="stMetricLabel"] {
    color: #B8EDE3 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800;
}


/* BUTTON */

.stButton > button {

    background: linear-gradient(
        90deg,
        #00B894,
        #0984E3
    );

    color: white !important;

    border: none;

    border-radius: 10px;

    font-weight: 700;

    padding: 10px 24px;

    transition: all 0.3s ease;
}

.stButton > button:hover {

    background: linear-gradient(
        90deg,
        #00D2A0,
        #00A8FF
    );

    transform: translateY(-2px);

    box-shadow:
        0px 5px 20px rgba(0, 200, 170, 0.35);
}


/* DATAFRAME */

[data-testid="stDataFrame"] {

    border: 1px solid #16C79A;

    border-radius: 12px;

    overflow: hidden;
}


/* SUCCESS */

[data-testid="stAlert"] {

    background-color: #063F35;

    border: 1px solid #16C79A;

    border-radius: 12px;
}


/* SIDEBAR RADIO */

[data-testid="stSidebar"] .stRadio label {

    color: white !important;

    font-weight: 600;
}


/* CAPTION */

[data-testid="stCaptionContainer"] {

    color: #A8DCD4 !important;
}


/* HORIZONTAL LINE */

hr {

    border-color: #16C79A;

}


/* CUSTOM CARD */

.card {

    background: linear-gradient(
        135deg,
        #073B4C,
        #075E54
    );

    border: 1px solid #16C79A;

    border-radius: 18px;

    padding: 20px;

    margin-bottom: 15px;

}


/* GREEN */

.green {
    color: #00D2A0 !important;
}


/* BLUE */

.blue {
    color: #00A8FF !important;
}


/* WHITE */

.white {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA GENERATION
# =========================================================

@st.cache_data
def fetch_or_generate_data():

    np.random.seed(42)

    periods = 1500

    dates = pd.date_range(
        start="2025-01-01",
        periods=periods,
        freq="h"
    )

    returns = np.random.normal(
        0.0003,
        0.015,
        size=periods
    )

    price_path = 60000 * np.exp(
        np.cumsum(returns)
    )

    df = pd.DataFrame({

        "Timestamp": dates,

        "Open": price_path * (
            1 + np.random.normal(
                0,
                0.002,
                periods
            )
        ),

        "High": price_path * (
            1 + np.abs(
                np.random.normal(
                    0,
                    0.005,
                    periods
                )
            )
        ),

        "Low": price_path * (
            1 - np.abs(
                np.random.normal(
                    0,
                    0.005,
                    periods
                )
            )
        ),

        "Close": price_path,

        "Volume": np.random.uniform(
            100,
            5000,
            periods
        )
    })

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_features(df):

    data = df.copy()

    data.sort_values(
        "Timestamp",
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )

    # Target

    data["Target_Delta"] = (
        data["Close"].shift(-1)
        - data["Close"]
    )

    # Price Change

    data["Price_Change"] = (
        data["Close"]
        - data["Open"]
    )

    # Percentage Change

    data["Pct_Change"] = (
        data["Close"].pct_change()
    )

    # Log Return

    data["Log_Return"] = np.log(
        data["Close"]
        / data["Close"].shift(1)
    )

    # EMA

    ema20 = data["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = data["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    ema200 = data["Close"].ewm(
        span=200,
        adjust=False
    ).mean()

    data["EMA_20_Ratio"] = (
        data["Close"] - ema20
    ) / data["Close"]

    data["EMA_50_Ratio"] = (
        data["Close"] - ema50
    ) / data["Close"]

    data["EMA_200_Ratio"] = (
        data["Close"] - ema200
    ) / data["Close"]

    # =====================================================
    # RSI
    # =====================================================

    delta = data["Close"].diff()

    gain = (
        delta
        .where(delta > 0, 0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta
        .where(delta < 0, 0)
        .rolling(14)
        .mean()
    )

    rs = gain / (loss + 1e-8)

    data["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # =====================================================
    # MACD
    # =====================================================

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_Norm"] = (
        macd / data["Close"]
    )

    data["MACD_Signal_Norm"] = (
        signal / data["Close"]
    )

    # =====================================================
    # BOLLINGER BANDS
    # =====================================================

    rolling_mean = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    rolling_std = (
        data["Close"]
        .rolling(20)
        .std()
    )

    upper = (
        rolling_mean
        + rolling_std * 2
    )

    lower = (
        rolling_mean
        - rolling_std * 2
    )

    data["BB_PctB"] = (
        (data["Close"] - lower)
        / (upper - lower + 1e-8)
    )

    data["BB_Width"] = (
        (upper - lower)
        / rolling_mean
    )

    # =====================================================
    # VOLATILITY
    # =====================================================

    data["Volatility"] = (
        data["Pct_Change"]
        .rolling(20)
        .std()
    )

    # =====================================================
    # LAG FEATURES
    # =====================================================

    for lag in [1, 2, 3, 6, 12, 24]:

        data[f"Lag_{lag}_Return"] = (
            data["Pct_Change"]
            .shift(lag)
        )

    data.dropna(
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )

    return data


# =========================================================
# MODEL TRAINING
# =========================================================

@st.cache_resource
def train_models():

    raw_df = fetch_or_generate_data()

    df = build_features(raw_df)

    feature_cols = [
        column
        for column in df.columns
        if column not in [
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Target_Delta"
        ]
    ]

    X = df[feature_cols]

    y = df["Target_Delta"]

    split_idx = int(
        len(df) * 0.80
    )

    X_train = X.iloc[:split_idx]

    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]

    y_test = y.iloc[split_idx:]

    close_previous = (
        df["Close"]
        .iloc[split_idx - 1:-1]
        .values
    )

    actual_close = (
        df["Close"]
        .iloc[split_idx:]
        .values
    )

    # Scaling

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # Models

    models = {

        "Random Forest":
        RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42
        ),

        "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
    }

    if HAS_LIGHTGBM:

        models["LightGBM"] = (
            lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                random_state=42,
                verbose=-1
            )
        )

    results = {}

    predictions = {}

    trained_models = {}

    # Train models

    for name, model in models.items():

        model.fit(
            X_train_scaled,
            y_train
        )

        pred_delta = model.predict(
            X_test_scaled
        )

        pred_close = (
            close_previous
            + pred_delta
        )

        mae = mean_absolute_error(
            actual_close,
            pred_close
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual_close,
                pred_close
            )
        )

        r2 = r2_score(
            actual_close,
            pred_close
        )

        mape = np.mean(
            np.abs(
                (
                    actual_close
                    - pred_close
                )
                / actual_close
            )
        ) * 100

        results[name] = {

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2,

            "MAPE": mape
        }

        predictions[name] = pred_close

        trained_models[name] = model

    metrics_df = pd.DataFrame(
        results
    ).T

    best_model_name = (
        metrics_df["MAE"]
        .idxmin()
    )

    return (
        raw_df,
        df,
        metrics_df,
        predictions,
        best_model_name,
        scaler,
        trained_models,
        feature_cols
    )


# =========================================================
# LOAD MODEL
# =========================================================

(
    raw_df,
    df,
    metrics_df,
    predictions,
    best_model_name,
    scaler,
    trained_models,
    feature_cols
) = train_models()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "# ₿ BTC Predict"
)

st.sidebar.caption(
    "AI Bitcoin Price Prediction"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "💰 Live Price",
        "🔮 Prediction",
        "📊 Analysis",
        "📅 Historical Data",
        "🤖 Models"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### ⚡ System Status"
)

st.sidebar.success(
    "Model Online"
)

if HAS_LIGHTGBM:

    st.sidebar.info(
        "LightGBM Available"
    )

else:

    st.sidebar.warning(
        "LightGBM Not Installed"
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title(
        "₿ Bitcoin Price Prediction"
    )

    st.caption(
        "AI-powered Bitcoin market analysis and machine learning prediction dashboard"
    )

    # Current values

    current_price = (
        df["Close"].iloc[-1]
    )

    previous_price = (
        df["Close"].iloc[-2]
    )

    change = (
        (current_price - previous_price)
        / previous_price
    ) * 100

    best_prediction = (
        predictions[
            best_model_name
        ][-1]
    )

    prediction_change = (
        (best_prediction - current_price)
        / current_price
    ) * 100

    # =====================================================
    # METRIC CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💰 Current Price",
            f"${current_price:,.2f}",
            f"{change:+.2f}%"
        )

    with col2:

        st.metric(
            "🔮 Predicted Price",
            f"${best_prediction:,.2f}",
            f"{prediction_change:+.2f}%"
        )

    with col3:

        sentiment = (
            "Bullish"
            if prediction_change > 0
            else "Bearish"
        )

        st.metric(
            "📊 Market Sentiment",
            sentiment
        )

    with col4:

        r2 = metrics_df.loc[
            best_model_name,
            "R2"
        ]

        st.metric(
            "🤖 Model R²",
            f"{r2:.2%}"
        )

    st.markdown("---")

    # =====================================================
    # PRICE CHART
    # =====================================================

    st.markdown(
        "### 📈 Bitcoin Price Chart"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Timestamp"],
            y=df["Close"],
            mode="lines",
            name="BTC Price",
            line=dict(
                color="#00D2A0",
                width=3
            )
        )
    )

    fig.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="#072B3D",
        plot_bgcolor="#072B3D",
        font=dict(
            color="white"
        ),
        xaxis=dict(
            gridcolor="#14505C"
        ),
        yaxis=dict(
            gridcolor="#14505C"
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # =====================================================
    # BOTTOM CARDS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🤖 Best Performing Model"
        )

        st.success(
            f"🏆 {best_model_name}"
        )

    with col2:

        st.markdown(
            "### 📊 Prediction Direction"
        )

        if prediction_change > 0:

            st.success(
                f"📈 Bullish — {prediction_change:+.2f}%"
            )

        else:

            st.error(
                f"📉 Bearish — {prediction_change:.2f}%"
            )


# =========================================================
# LIVE PRICE
# =========================================================

elif page == "💰 Live Price":

    st.title(
        "💰 Bitcoin Market Data"
    )

    current_price = (
        df["Close"].iloc[-1]
    )

    previous_price = (
        df["Close"].iloc[-2]
    )

    change = (
        (current_price - previous_price)
        / previous_price
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "BTC / USDT",
            f"${current_price:,.2f}"
        )

    with col2:

        st.metric(
            "24H Change",
            f"{change:+.2f}%"
        )

    with col3:

        st.metric(
            "Volume",
            f"{df['Volume'].iloc[-1]:,.0f}"
        )

    st.markdown(
        "### 📋 Recent Market Data"
    )

    st.dataframe(
        raw_df.tail(20),
        width="stretch"
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.title(
        "🔮 Bitcoin Price Prediction"
    )

    current = (
        df["Close"].iloc[-1]
    )

    prediction = (
        predictions[
            best_model_name
        ][-1]
    )

    change = (
        (prediction - current)
        / current
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Price",
            f"${current:,.2f}"
        )

    with col2:

        st.metric(
            "Predicted Price",
            f"${prediction:,.2f}"
        )

    with col3:

        st.metric(
            "Expected Change",
            f"{change:+.2f}%"
        )

    st.markdown(
        f"### 🤖 Prediction Model: {best_model_name}"
    )

    test_start = int(
        len(df) * 0.80
    )

    test_dates = df[
        "Timestamp"
    ].iloc[test_start:]

    actual = df[
        "Close"
    ].iloc[test_start:]

    predicted = predictions[
        best_model_name
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=actual,
            name="Actual Price",
            line=dict(
                color="#00D2A0",
                width=3
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=predicted,
            name="Predicted Price",
            line=dict(
                color="#00A8FF",
                width=3,
                dash="dash"
            )
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="#072B3D",
        plot_bgcolor="#072B3D",
        font=dict(
            color="white"
        ),
        xaxis=dict(
            gridcolor="#14505C"
        ),
        yaxis=dict(
            gridcolor="#14505C"
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

elif page == "📊 Analysis":

    st.title(
        "📊 Technical Analysis"
    )

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "RSI",
            f"{latest['RSI']:.2f}"
        )

    with col2:

        st.metric(
            "Volatility",
            f"{latest['Volatility']:.4f}"
        )

    with col3:

        st.metric(
            "MACD",
            f"{latest['MACD_Norm']:.4f}"
        )

    with col4:

        st.metric(
            "BB %B",
            f"{latest['BB_PctB']:.2f}"
        )

    st.markdown(
        "### 📈 Technical Indicators"
    )

    indicators = pd.DataFrame({

        "Indicator": [

            "EMA 20 Ratio",

            "EMA 50 Ratio",

            "EMA 200 Ratio",

            "RSI",

            "MACD",

            "Bollinger %B",

            "Bollinger Width",

            "Volatility"
        ],

        "Value": [

            latest["EMA_20_Ratio"],

            latest["EMA_50_Ratio"],

            latest["EMA_200_Ratio"],

            latest["RSI"],

            latest["MACD_Norm"],

            latest["BB_PctB"],

            latest["BB_Width"],

            latest["Volatility"]
        ]
    })

    st.dataframe(
        indicators,
        width="stretch"
    )


# =========================================================
# HISTORICAL DATA
# =========================================================

elif page == "📅 Historical Data":

    st.title(
        "📅 Historical Bitcoin Data"
    )

    st.caption(
        "Historical OHLCV market information"
    )

    st.dataframe(
        raw_df.sort_values(
            "Timestamp",
            ascending=False
        ),
        width="stretch"
    )


# =========================================================
# MODELS
# =========================================================

elif page == "🤖 Models":

    st.title(
        "🤖 Machine Learning Models"
    )

    st.caption(
        "Performance comparison of trained prediction models"
    )

    display_metrics = metrics_df.copy()

    st.dataframe(
        display_metrics.style.format({

            "MAE": "{:.2f}",

            "RMSE": "{:.2f}",

            "R2": "{:.4f}",

            "MAPE": "{:.2f}%"

        }),
        width="stretch"
    )

    st.markdown("---")

    st.success(
        f"🏆 Best Performing Model: {best_model_name}"
    )

    # Model comparison chart

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metrics_df.index,
            y=metrics_df["MAE"],
            name="MAE",
            marker_color="#00D2A0"
        )
    )

    fig.update_layout(
        title="Model MAE Comparison",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#072B3D",
        plot_bgcolor="#072B3D",
        font=dict(
            color="white"
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#A8DCD4;">
        ₿ <b>BTC Predict</b> |
        Machine Learning Bitcoin Analysis |
        Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

