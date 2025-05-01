# App Trading Options using BS model
import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import qfin as qf
from matplotlib import rcParams

# Set page config
st.set_page_config(
    page_title="Trading Options Using the Black-Scholes Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    /* General styles */
    body {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Metric container */
    .metric-container {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Plot container */
    .plot-container {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Warning box */
    .warning-box {
        background-color: #ffe6e6;
        border-left: 6px solid #ff0000;
        padding: 1rem;
        margin-top: 2rem;
        font-weight: bold;
        color: #ff0000;
    }

    /* Run simulation button */
    .stButton>button {
        background-color: #ff6600;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #e65c00;
        color: white;
    }

    </style>
""", unsafe_allow_html=True)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10

st.markdown("""
    <p style="font-size: 12px; text-align: center;">
        Created by: <a href="https://www.linkedin.com/in/luca-girlando-775463302/" target="_blank">Luca Girlando</a>
    </p>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 style="color: black;">📊 Trading Options Using the Black-Scholes Model</h1>', unsafe_allow_html=True)
st.markdown("""
This interactive tool demonstrates how to identify trading opportunities by comparing 
Black-Scholes theoretical prices with market maker quotes for **call/put options**. The strategy involves selling 
call options when market prices are higher than theoretical values. 
So you'll see that with the default data, the strategy works when running a high number of Monte Carlo simulations with the call option, but not with the put.
""")

# Sidebar with user inputs
with st.sidebar:
    st.header("⚙️ Option Parameters")
    option_type = st.radio("Option Type", ["Call", "Put"], index=0)
    strategy_type = st.radio("Strategy Direction", ["Buy (Long)", "Sell (Short)"], index=0, help="Buy = Buy options, Sell = Sell options")
    S = st.number_input("Current Stock Price (S)", value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.3, step=0.01, help="Annualized volatility of the underlying asset")
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.15, 0.05, step=0.01, help="Annual risk-free interest rate")
    t = st.slider("Time to Maturity (years)", 0.1, 5.0, 1.0, step=0.1, help="Time until option expiration")
    market_price = st.number_input("Market Maker Ask Price", value=14.10, step=0.1, help="Current market price for the option")
    n_simulations = st.slider("Number of Simulations", 100, 100000, 1000, step=100, help="More simulations increase result accuracy but take longer")

# Black-Scholes formula
def black_scholes(S, K, sigma, r, t, option_type="call"):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    
    if option_type.lower() == "call":
        C = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
        return C
    elif option_type.lower() == "put":
        P = K * np.exp(-r*t) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return P
    else:
        raise ValueError("Option type must be either 'call' or 'put'")

# Calculate theoretical price
theoretical_price = black_scholes(S, K, sigma, r, t, option_type.lower())

# Display formulas
with st.expander("📚 Black-Scholes Formula Details", expanded=True):
    if option_type == "Call":
        st.markdown(r"""
        The Black-Scholes **call** option price is given by:

        $$
        C(S, t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)
        $$

        where:
        $$
        d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}
        $$
        $$
        d_2 = d_1 - \sigma\sqrt{T-t}
        $$
        """)
    else:
        st.markdown(r"""
        The Black-Scholes **put** option price is given by:

        $$
        P(S, t) = K e^{-r(T-t)} N(-d_2) - S_t N(-d_1)
        $$

        where:
        $$
        d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}
        $$
        $$
        d_2 = d_1 - \sigma\sqrt{T-t}
        $$
        """)
    
    st.markdown(r"""
    - $S$ = Current stock price
    - $K$ = Strike price
    - $T-t$ = Time to maturity
    - $r$ = Risk-free interest rate
    - $\sigma$ = Volatility of the stock
    - $N(\cdot)$ = Cumulative distribution function of the standard normal distribution
    """)

# Display calculated values
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(f"Theoretical {option_type} Price", f"{theoretical_price:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Market Maker Ask Price", f"{market_price:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    edge = theoretical_price - market_price
    st.metric("Theoretical Edge", f"{edge:.2f}", delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)

# Single simulation visualization
st.header(f"📉 Single {option_type} Option {strategy_type.split(' ')[0]} Simulation")
st.markdown(f"""
Below shows one possible path for the underlying stock price and the resulting P/L 
from {strategy_type.lower()} the {option_type.lower()} option at the market price.
""")

path = qf.simulations.GeometricBrownianMotion(S, r, sigma, 1/252, t)

with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Terminal Value of {strategy_type} {option_type} Option", fontsize=14, pad=20)
    ax.hlines(K, 0, 252, label='Strike Price', color='#e74c3c', linewidth=2)
    ax.plot(path.simulated_path, label='Stock Price Path', color='#3498db', linewidth=2)

    if strategy_type == "Sell (Short)":
        if option_type == "Call":
            terminal_payoff = -max(path.simulated_path[-1] - K, 0)
            if terminal_payoff == 0:
                ax.vlines(252, path.simulated_path[-1], K, color='#2ecc71', 
                        label=f"Profit (Keep ${market_price:.2f} premium)", linewidth=3)
            else:
                ax.vlines(252, K, path.simulated_path[-1], color='#e74c3c', 
                        label=f"Loss (Pay ${-terminal_payoff:.2f})", linewidth=3)
        else:
            terminal_payoff = -max(K - path.simulated_path[-1], 0)
            if terminal_payoff == 0:
                ax.vlines(252, K, path.simulated_path[-1], color='#2ecc71', 
                        label=f"Profit (Keep ${market_price:.2f} premium)", linewidth=3)
            else:
                ax.vlines(252, path.simulated_path[-1], K, color='#e74c3c', 
                        label=f"Loss (Pay ${-terminal_payoff:.2f})", linewidth=3)
   
    elif strategy_type == "Buy (Long)":
        if option_type == "Call":
            terminal_payoff = max(path.simulated_path[-1] - K, 0)
            if terminal_payoff > market_price:
                ax.vlines(252, K, path.simulated_path[-1], color='#2ecc71',
                          label=f"Profit: +${terminal_payoff - market_price:.2f}", linewidth=3)
            else:
                ax.vlines(252, path.simulated_path[-1], K, color='#e74c3c',
                          label=f"Loss: -${market_price - terminal_payoff:.2f}", linewidth=3)
        else:
            terminal_payoff = max(K - path.simulated_path[-1], 0)
            if terminal_payoff > market_price:
                ax.vlines(252, path.simulated_path[-1], K, color='#2ecc71',
                          label=f"Profit: +${terminal_payoff - market_price:.2f}", linewidth=3)
            else:
                ax.vlines(252, K, path.simulated_path[-1], color='#e74c3c',
                          label=f"Loss: -${market_price - terminal_payoff:.2f}", linewidth=3)

    ax.legend()
    ax.set_xlabel("Time Step (days)")
    ax.set_ylabel("Stock Price")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Run Monte Carlo
st.header("🎲 Monte Carlo Simulation")

if st.button("Run Monte Carlo Simulation"):
    terminal_prices = []
    pnl = []

    for _ in range(n_simulations):
        p = qf.simulations.GeometricBrownianMotion(S, r, sigma, 1/252, t).simulated_path[-1]
        terminal_prices.append(p)

        if option_type == "Call":
            payoff = max(p - K, 0)
        else:
            payoff = max(K - p, 0)

        if strategy_type == "Sell (Short)":
            pnl.append(market_price - payoff)
        else:
            pnl.append(payoff - market_price)

    pnl = np.array(pnl)

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(pnl, bins=50, color='#1abc9c', edgecolor='black')
    ax2.set_title(f"Distribution of Profit & Loss from {n_simulations} Simulations", fontsize=14)
    ax2.set_xlabel("Profit / Loss")
    ax2.set_ylabel("Frequency")
    ax2.axvline(x=np.mean(pnl), color='red', linestyle='--', linewidth=2, label=f"Mean P&L: ${np.mean(pnl):.2f}")
    ax2.legend()
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

    st.success(f"Average P&L over {n_simulations} simulations: ${np.mean(pnl):.2f}")
    st.info(f"Standard Deviation of P&L: ${np.std(pnl):.2f}")
    st.info(f"Probability of Profit: {(pnl > 0).mean()*100:.2f}%")

# Disclaimer
st.markdown("""
<div class="warning-box">
    ⚠️ Note: This is for educational purposes only. Options trading involves substantial risk.
</div>
""", unsafe_allow_html=True)

 
