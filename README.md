# Options Trading Strategy Simulator with Black-Scholes Model

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Overview
An interactive web application that simulates options trading strategies using the Black-Scholes model. The tool allows users to:
- Compare theoretical option prices with market quotes
- Visualize P/L scenarios for both call and put options
- Test long/short strategies via Monte Carlo simulations
- Understand the limitations of the Black-Scholes model

## Key Features
- **Black-Scholes Calculator**: Computes theoretical prices for European call/put options
- **Strategy Simulation**: 
  - Buy (Long) or Sell (Short) strategies
  - Single-path visualization with P/L breakdown
- **Monte Carlo Analysis**: 
  - 1,000-100,000 simulated price paths
  - Equity curve and P/L distribution
- **Interactive Parameters**:
  - Adjustable strike price, volatility, risk-free rate
  - Customizable market price and expiration

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
