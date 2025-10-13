import streamlit as st
import pandas as pd
import numpy as np

# 💻 Green-on-black retro style
st.markdown(
    """
    <style>
    /* Import Orbitron from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

    /* Apply Orbitron to all main text */
    html, body, .stApp {
        background-color: black !important;
        color: #00FF00 !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Title glow if desired */
    h1 {
        color: #00FF00 !important;
        text-shadow: 0 0 10px #00FF00;
    }

thead tr th, tbody tr td {
    font-family: 'Orbitron', sans-serif !important;
    color: #00FF00 !important;
}

    /* Button styling */
    .stButton > button {
    color: #00FF00 !important;
    background-color: black !important;
    border: 1px solid #00FF00 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 0.6em 1.5em !important;
    text-transform: uppercase !important;
    text-shadow: 0 0 8px #00FF00 !important;
    }

    /* Number input boxes */
    .stNumberInput input {
        color: #00FF00 !important;
        background-color: black !important;
        border: 1px solid #00FF00 !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Slider handle text */
    .stSlider > div {
        color: #00FF00 !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    div[data-baseweb="slider"] > div:first-child {
        background: #00FF00 !important;
    }

    div[data-baseweb="slider"] > div:nth-child(2) {
        background: #003300 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='color:#00FF00; font-family: 'Orbitron', serif;'>🧾 Generate My Portfolio</h1>",
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

# Step 1: Certainty Equivalent
st.markdown(
    """
    💬 <span style="color:#00FF00; font-family: 'Orbitron', sans-serif;">
    You hold a lottery ticket that gives you:<br>
    📄 50% chance of winning <strong>€1000</strong><br>
    📄 50% chance of winning <strong>€0</strong><br>
    </span>
    <br>
    <span style="color:#00FF00; font-family: 'Orbitron', sans-serif;">How much would you sell it for? (€)</span>
    """,
    unsafe_allow_html=True
)

ce_input = st.number_input("", min_value=0.0, step=10.0, value=500.0)
risk_score = ce_input / 500
risk_score = min(max(risk_score, 0.01), 0.99)

# Step 2: Number of stocks
st.markdown(
    """
    <span style="color:#00FF00; font-family: 'Orbitron', sans-serif;">
    🎯 How many stocks should your portfolio include?
    </span>
    """,
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

    st.markdown(
    """
    <h2 style="color:#00FF00; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 8px #00FF00;">
    ✅ Recommended Portfolio Allocation
    </h2>
    """,
    unsafe_allow_html=True
    )

    final_weights = result["Weights"].sort_values(ascending=False)
    weights_percent = (final_weights * 100).round(2)

    weights_named = pd.DataFrame({
        "Company": [ticker_name_map.get(t, "Unknown") for t in final_weights.index],
        "Ticker": final_weights.index,
        "Allocation (%)": ["{:.2f}%".format(w) for w in weights_percent.values]
    }).set_index("Ticker")
    # Convert DataFrame to styled HTML table
    html_table = weights_named.to_html(index=True, classes='custom-table', border=0)
    st.markdown(
        """
        <style>
        .custom-table {
            width: 100%;
            font-family: 'Orbitron', sans-serif;
            color: #00FF00;
            background-color: black;
            border-collapse: collapse;
            font-size: 16px;
            margin-top: 20px;
        }

        .custom-table th, .custom-table td {
            border: 1px solid #004400;
            padding: 10px 14px;
            text-align: left;
            text-shadow: 0 0 4px #00FF00;
        }

        .custom-table thead {
            background-color: #001100;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # === Display the table ===
    st.markdown(html_table, unsafe_allow_html=True)


    expected_return = result['Expected Return']
    volatility = result['Volatility']
    st.markdown(
    f"""
    📈 <span style='color:#00FF00; font-family: Orbitron, sans-serif;'>
    <strong>Expected Return:</strong> {expected_return * 100:.2f}%
    </span><br>
    📉 <span style='color:#00FF00; font-family: Orbitron, sans-serif;'>
    <strong>Volatility:</strong> {volatility * 100:.2f}%
    </span>
    """,
    unsafe_allow_html=True
)

# Step 5: Investment projection
st.markdown("💰 <span style='color:#00FF00; font-family: Orbitron, sans-serif;'>How much would you like to invest? (€)</span>", unsafe_allow_html=True)
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
    f"""
    📊 <span style='color:#00FF00; font-family: Orbitron, sans-serif;'>
    Over 5 years, your investment could grow to <strong>€{future_value:,.0f}</strong>
    </span><br>
    📉 <span style='color:#00FF00; font-family: Orbitron, sans-serif;'>
    Expected yearly range: <strong>€{lower_bound:,.0f} – €{upper_bound:,.0f}</strong>
    </span>
    """,
    unsafe_allow_html=True
)






