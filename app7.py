# App Trading Options using BS model
import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import qfin as qf
from matplotlib import rcParams

st.set_page_config(
    page_title="Trading Options Using the Black-Scholes Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    /* Light Theme - Professionale e leggibile */
    --primary-dark: #2d3436;         /* Grigio antracite per testo */
    --primary-light: #f5f6fa;       /* Grigio chiarissimo per sfondi */
    --card-bg: #ffffff;             /* Bianco puro per cards */
    --card-border: #dfe6e9;         /* Grigio chiaro per bordi */
    --accent-blue: #0984e3;         /* Blu vivo ma professionale */
    --accent-red: #d63031;          /* Rosso per avvisi */
    --accent-green: #00b894;        /* Verde professionale */
    --sidebar-bg: #2d3436;          /* Grigio scuro per sidebar */
    --sidebar-text: #ffffff;        /* Bianco per testo sidebar */
    --metric-bg: #f1f2f6;           /* Grigio molto chiaro per metriche */
    
    /* Dark Theme - Già valido */
    --dark-primary: #f5f6fa;
    --dark-bg: #1e272e;
    --dark-card: #2d3436;
    --dark-border: #3c484f;
}

/* Base Styles */
* {
    font-family: 'Roboto', 'Segoe UI', sans-serif;
}

body {
    background-color: var(--primary-light);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--primary-dark) !important;
    font-weight: 600;
}

.stMarkdown, .stText {
    color: var(--primary-dark) !important;
}

/* Sidebar */
.sidebar .sidebar-content {
    background: var(--sidebar-bg) !important;
    color: var(--sidebar-text) !important;
}

.sidebar .stRadio label,
.sidebar .stNumberInput label,
.sidebar .stSlider label {
    color: var(--sidebar-text) !important;
}

/* Cards */
.metric-container {
    background-color: var(--metric-bg) !important;
    border-radius: 8px;
    padding: 16px;
    border: 1px solid var(--card-border);
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.plot-container {
    background-color: var(--card-bg) !important;
    border-radius: 8px;
    padding: 20px;
    border: 1px solid var(--card-border);
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 24px;
}

/* Buttons */
.stButton>button {
    background-color: var(--accent-blue);
    color: white !important;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 500;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #0768b9;
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

/* Warning Box */
.warning-box {
    background-color: #ffeaa7;
    color: #d63031;
    padding: 16px;
    border-radius: 8px;
    border-left: 4px solid var(--accent-red);
    margin: 16px 0;
}

/* Dark Mode Overrides */
@media (prefers-color-scheme: dark) {
    :root {
        --primary-dark: var(--dark-primary);
        --primary-light: var(--dark-bg);
        --card-bg: var(--dark-card);
        --card-border: var(--dark-border);
        --metric-bg: #34495e;
    }
    
    .warning-box {
        background-color: #3d3d3d;
        color: #ff7675;
        border-left-color: #d63031;
    }
}

/* Specific Fixes */
[data-testid="stHeader"] {
    color: var(--primary-dark) !important;
}

.st-emotion-cache-10trblm {
    color: var(--primary-dark) !important;
}

.st-emotion-cache-1kyxreq {
    color: var(--primary-dark) !important;
}
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10

st.markdown("""
    <p style="font-size: 12px; text-align: center;">
        Created by: <a href="https://www.linkedin.com/in/luca-girlando-775463302/" target="_blank">Luca Girlando</a>
    </p>
""", unsafe_allow_html=True)

st.title("📊 Trading Options Using the Black-Scholes Model")
st.markdown("""
This interactive tool demonstrates how to identify trading opportunities by comparing 
Black-Scholes theoretical prices with market maker quotes for **call/put options**. The strategy involves selling 
call options when market prices are higher than theoretical values. 
So you'll see that with the default data, the strategy works when running a high number of Monte Carlo simulations with the call option, but not with the put.
""")

with st.sidebar:
    st.header("⚙️ Option Parameters")
    option_type = st.radio("Option Type", ["Call", "Put"], index=0)
    strategy_type = st.sidebar.radio("Strategy Direction",["Buy (Long)", "Sell (Short)"],index=0,help="Buy = Buy options, Sell = Sell options")
    S = st.number_input("Current Stock Price (S)", value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.3, step=0.01, help="Annualized volatility of the underlying asset")
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.15, 0.05, step=0.01, help="Annual risk-free interest rate")
    t = st.slider("Time to Maturity (years)", 0.1, 5.0, 1.0, step=0.1, help="Time until option expiration")
    market_price = st.number_input("Market Maker Ask Price", value=14.10, step=0.1, help="Current market price for the option")
    n_simulations = st.slider("Number of Simulations", 100, 100000, 1000, step=100, help="More simulations increase result accuracy but take longer")

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

theoretical_price = black_scholes(S, K, sigma, r, t, option_type.lower())

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
    else:
        if option_type == "Call":
            terminal_payoff = max(path.simulated_path[-1] - K, 0)
            if terminal_payoff == 0:
                ax.vlines(252, path.simulated_path[-1], K, color='#e74c3c', 
                        label=f"Loss (Lose ${market_price:.2f} premium)", linewidth=3)
            else:
                ax.vlines(252, K, path.simulated_path[-1], color='#2ecc71', 
                        label=f"Profit (Gain ${terminal_payoff:.2f})", linewidth=3)
        else:
            terminal_payoff = max(K - path.simulated_path[-1], 0)
            if terminal_payoff == 0:
                ax.vlines(252, K, path.simulated_path[-1], color='#e74c3c', 
                        label=f"Loss (Lose ${market_price:.2f} premium)", linewidth=3)
            else:
                ax.vlines(252, path.simulated_path[-1], K, color='#2ecc71', 
                        label=f"Profit (Gain ${terminal_payoff:.2f})", linewidth=3)

    ax.set_xlabel('Time (trading days)', fontsize=12)
    ax.set_ylabel('Stock Price', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

if strategy_type == "Sell (Short)":
    pl_single = market_price + terminal_payoff
else:
    pl_single = terminal_payoff - market_price

st.markdown('<div class="metric-container">', unsafe_allow_html=True)
st.metric("Single Simulation P/L", f"{pl_single:.2f}", 
          help=f"Profit/Loss from {strategy_type.lower()} one {option_type.lower()} option at market price")
st.markdown('</div>', unsafe_allow_html=True)

st.header("🎲 Monte Carlo Simulation Results")
st.markdown(f"""
Running {n_simulations:,} simulations to estimate the expected P/L from this {option_type.lower()} option strategy.
The underlying stock price follows Geometric Brownian Motion:
""")
st.markdown(r"""
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

where:
- $\mu$ = Drift rate (risk-free rate in risk-neutral measure)
- $\sigma$ = Volatility
- $W_t$ = Wiener process (Brownian motion)
""")

if st.button("🚀 Run Monte Carlo Simulation", key="monte_carlo"):
    st.write("Running simulations...")
    
    premium = market_price
    pls = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_simulations):
        path = qf.simulations.GeometricBrownianMotion(S, r, sigma, 1/252, t)
        
        if option_type == "Call":
            terminal_value = max(path.simulated_path[-1] - K, 0)
        else:
            terminal_value = max(K - path.simulated_path[-1], 0)
            
        pls.append(terminal_value - premium)
        
        if i % (n_simulations//100) == 0:
            progress_bar.progress(i/n_simulations)
            status_text.text(f"Progress: {i/n_simulations*100:.1f}% complete")
    
    expected_pl = np.mean(pls)
    
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(f"Expected P/L per {option_type} Option", f"{expected_pl:.2f}",
             help=f"Average profit/loss per {option_type.lower()} option over all simulations")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.set_title(f"{option_type} Strategy Equity Curve", fontsize=14, pad=20)
        ax2.plot(np.cumsum(pls), label="Account Equity", color='#3498db', linewidth=2)
        ax2.set_xlabel('Option Trade', fontsize=12)
        ax2.set_ylabel('Portfolio Value', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ Why does the number of simulations matter?", expanded=True):
        st.markdown("""
        The number of simulations significantly impacts the reliability of our results because:
        
        1. **Law of Large Numbers**: As we increase simulations (1000 → 10000 → 100000), 
           our estimate converges to the true expected value.
        
        2. **Variance Reduction**: More simulations reduce the standard error, making the 
           expected P/L more stable and reliable.
        
        3. **Extreme Events**: Rare but impactful events are better captured with higher 
           simulation counts, giving a more complete picture of potential outcomes.
        
        4. **Confidence Intervals**: With more data, we can be more confident that our 
           estimated edge is statistically significant.
        
        In practice, 10,000+ simulations are typically needed for stable results in 
        options pricing applications.
        """)
    
    with st.container():
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.set_title(f"Distribution of {option_type} P/L Outcomes", fontsize=14, pad=20)
        ax3.hist(pls, bins=50, color='#3498db', edgecolor='#2980b9')
        ax3.axvline(expected_pl, color='#e74c3c', linestyle='--', 
                   linewidth=2, label=f'Mean P/L: {expected_pl:.2f}')
        ax3.set_xlabel('P/L per Option', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

st.header("⚠️ Model Limitations")
st.markdown("""
While the Black-Scholes model provides a theoretical framework, real-world trading has challenges:
- **Non-constant volatility**: Volatility clusters and changes over time
- **Market jumps**: Real markets experience sudden jumps not captured by GBM
- **Liquidity constraints**: Market maker quotes may not always be available
- **Transaction costs**: Not accounted for in this simulation
- **Discrete hedging**: Continuous hedging is impossible in practice
- **Dividends**: The basic model doesn't account for dividend payments
- **Interest rate changes**: Assumes constant risk-free rate
- **Early exercise**: For American options, early exercise isn't considered
""")

st.markdown("""
<div class="warning-box">
    <strong>Note:</strong> This is for educational purposes only. Options trading involves substantial risk.
</div>
""", unsafe_allow_html=True)
