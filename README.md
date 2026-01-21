# 🏦 Mining & Energy Risk Terminal (2026 Edition)

### *Quantitative Risk Management in the era of the 'Great Schism'*

This terminal was built to analyze the 2023–2026 market regime—a period defined by the breakdown of traditional commodity correlations. While Gold remains the primary safe-haven during the **2026 Greenland Gambit**, Silver has emerged as the structural leader, outperforming Gold on both absolute and risk-adjusted (Sharpe) bases.

## 🚀 Key Features

- **The 'Horse Race' (Base 100):** Real-time tracking of growth attribution across metals and energy.
- **Rolling Efficiency:** 252-day rolling Sharpe ratios to identify the "Efficiency Pivot" where Silver decoupled from the pack.
- **GARCH-Based Risk Engine:** Moving beyond simple historical volatility to estimate "Tail Risk" using GARCH(1,1) models with Student-t innovations.
- **Expected Shortfall (ES):** A deep look at "Average Crash Depth"—calculating exactly how much a position is expected to lose during a tail-risk event.
- **Live Price Verification:** A real-time spot price window pulled via Yahoo Finance to ground the math in current market reality.

## 🛠️ Methodology: Beyond Value-at-Risk

Standard risk models often underestimate the "Fat Tails" (kurtosis) of commodity markets. This terminal employs a more robust approach:

1. **GARCH (1,1):** Models volatility clustering, acknowledging that high-volatility periods (like the current geopolitical environment) tend to persist.
2. **Student-t Innovations:** Used to capture the extreme outliers observed in 2025-2026 that a standard Normal distribution would miss.
3. **99% Expected Shortfall:** Unlike VaR, which only tells you the *threshold* of a loss, ES calculates the average loss *beyond* that threshold. This is vital for navigating "Black Swan" events, such as the 15% single-day slump Silver experienced in late December.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/mining-energy-risk-terminal.git](https://github.com/yourusername/mining-energy-risk-terminal.git)
   cd mining-energy-risk-terminal