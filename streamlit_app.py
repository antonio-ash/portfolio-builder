import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Build your own portfolio", layout="wide")

# 💻 Green-on-black retro style
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: black !important;
        color: #00FF00 !important;
        font-family: 'Georgia', serif;
    }

    .stButton > button {
        color: #00FF00 !important;
        background-color: black !important;
        border: 1px solid #00FF00 !important;
    }

    .stNumberInput input {
        color: #00FF00 !important;
        background-color: black !important;
        border: 1px solid #00FF00 !important;
    }

    .stSlider > div {
        color: #00FF00 !important;
    }

    div[data-baseweb="slider"] > div:first-child {
        background: #00FF00 !important;
    }

    div[data-baseweb="slider"] > div:nth-child(2) {
        background: #003300 !important;
    }

    div[data-baseweb="slider"] span[role="slider"] {
        background-color: #00FF00 !important;
        border: 2px solid #00FF00 !important;
    }

    .stMarkdown p, .stTable td, .stTable th {
        color: #00FF00 !important;
        background-color: black !important;
        text-align: center !important;
    }

    table td, table th {
        color: #00FF00 !important;
        background-color: black !important;
        text-align: center !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_returns():
    return pd.read_csv("returns_data.csv", index_col=0, parse_dates=True)

@st.cache_data
def load_ticker_names():
    df = pd.read_csv("ticker_name_mapping.csv")
    return dict(zip(df['Symbol'], df['Company']))

returns_df = load_returns()
ticker_name_map = load_ticker_names()

expected_annual_returns = returns_df.mean() * 12
annualized_cov_matrix = returns_df.cov() * 12

# Title
st.title("📼 Generate My Portfolio")

# Step 1: Certainty Equivalent
st.markdown(
    """
    💬 <span style='color:#00FF00'>
    You hold a lottery ticket that gives you:<br>
    📄 50% chance of winning <strong>€1000</strong><br>
    📄 50% chance of winning <strong>€0</strong><br>
    </span>
    <br>
    <span style='color:#00FF00'>How much would you sell it for? (€)</span>
    """,
    unsafe_allow_html=True
)

ce_input = st.number_input("", min_value=0.0, step=10.0, value=500.0)
risk_score = ce_input / 500
risk_score = min(max(risk_score, 0.01), 0.99)

# Step 2: Number of stocks
st.markdown(
    "<span style='color:#00FF00'>🎯 How many stocks should your portfolio include?</span>",
    unsafe_allow_html=True
)
n_stocks = st.number_input("", min_value=5, max_value=100, value=30, step=1)

# Step 3: Portfolio Simulation
def simulate_random_portfolios_constrained(expected_annual_returns, annualized_cov_matrix, n_stocks, n_portfolios, risk_score):
    np.random.seed(42)
    tickers = expected_annual_returns.index.tolist()
    n_total_stocks = len(tickers)
    results = []
    gamma = 1 / (risk_score + 1e-6)

    for _ in range(n_portfolios):
        selected_indices = np.random.choice(n_total_stocks, n_stocks, replace=False)
        selected_tickers = [tickers[i] for i in selected_indices]

        weights = np.random.rand(n_stocks)
        weights /= np.sum(weights)

        sub_returns = expected_annual_returns[selected_tickers]
        sub_cov = annualized_cov_matrix.loc[selected_tickers, selected_tickers]

        port_return = np.dot(weights, sub_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(sub_cov, weights)))
        utility = port_return - gamma * port_volatility

        results.append({
            'Expected Return': port_return,
            'Volatility': port_volatility,
            'Utility': utility,
            'Weights': pd.Series(weights, index=selected_tickers)
        })

    best_result = max(results, key=lambda x: x['Utility'])
    return best_result

# Step 4: Generate Portfolio
if st.button("Generate Portfolio") or "result" in st.session_state:
    if "result" not in st.session_state:
        st.session_state.result = simulate_random_portfolios_constrained(
            expected_annual_returns=expected_annual_returns,
            annualized_cov_matrix=annualized_cov_matrix,
            n_stocks=n_stocks,
            n_portfolios=5000,
            risk_score=risk_score
        )
    
    result = simulate_random_portfolios_constrained(
        expected_annual_returns=expected_annual_returns,
        annualized_cov_matrix=annualized_cov_matrix,
        n_stocks=n_stocks,
        n_portfolios=5000,
        risk_score=risk_score
    )

    st.subheader("✅ Recommended Portfolio Allocation")

    final_weights = result["Weights"].sort_values(ascending=False)
    weights_percent = (final_weights * 100).round(2)

    weights_named = pd.DataFrame({
        "Company": [ticker_name_map.get(t, "Unknown") for t in final_weights.index],
        "Ticker": final_weights.index,
        "Allocation (%)": ["{:.2f}%".format(w) for w in weights_percent.values]
    }).set_index("Ticker")

    st.markdown(weights_named.to_html(escape=False, index=True, justify='center'), unsafe_allow_html=True)

    expected_return = result['Expected Return']
    volatility = result['Volatility']

    st.markdown(f"**📈 Expected Return:** `{expected_return * 100:.2f}%`")
    st.markdown(f"**📉 Volatility:** `{volatility * 100:.2f}%`")

    # Step 5: Investment projection
    st.markdown("💰 <span style='color:#00FF00'>How much would you like to invest? (€)</span>", unsafe_allow_html=True)
    initial_investment = st.number_input("", min_value=1000, step=1000, value=10000)

    # 5-year projection
    future_value = initial_investment * (1 + expected_return) ** 5

    # Compute return range using lognormal assumption
    mu = result['Expected Return']
    sigma = result['Volatility']
    investment = initial_investment  # Make sure this is defined earlier

    lower_bound = investment * np.exp(mu - sigma)
    upper_bound = investment * np.exp(mu + sigma)

    st.markdown(
        f"📊 Over 5 years, your investment could grow to **€{future_value:,.0f}**, "
        f"assuming the portfolio performs in line with historical trends."
    )

    st.markdown(
        f"<span style='color:#00FF00'>💡 In a typical year, your €{investment:,.0f} investment could fluctuate between "
        f"€{lower_bound:,.0f} and €{upper_bound:,.0f}, based on portfolio volatility.</span>",
        unsafe_allow_html=True
)




