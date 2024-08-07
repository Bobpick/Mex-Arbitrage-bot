# Crypto Arbitrage Bot

## Overview

This Crypto Arbitrage Bot is an advanced trading system designed to capitalize on price differences across multiple cryptocurrency exchanges. It focuses on the MEX/USDT trading pair and utilizes real-time data, historical analysis, and machine learning to make informed trading decisions.


## Demo

Check out a demo of the Crypto Arbitrage Bot in action:

[Watch the Demo Video](https://github.com/Bobpick/Mex-Arbitrage-bot/blob/main/Mex_Bot.mp4?raw=true)

## THIS PROGRAM IS STILL IN THE TESTING PHASES!! 


## Features

- **Multi-exchange support**: LBank, Bitrue, AscendEx
- **Real-time price monitoring**
- **Historical data analysis and backtesting**
- **Machine learning-based trade prediction**
- **Risk management and position sizing**
- **Performance tracking and reporting**
- **Sandbox mode for testing**
- **Configurable settings via JSON**
- **Secure API key management using environment variables**

## Requirements

- Python 3.7+
- `ccxt` library
- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `aiohttp`
- `win10toast` (for Windows notifications)
- `python-dotenv`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/crypto-arbitrage-bot.git
    cd crypto-arbitrage-bot
    ```

2. Install required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up your environment variables:
    Create a `.env` file in the project root and add your API keys:
    ```sh
    LBANK_API_KEY=your_lbank_api_key
    LBANK_SECRET=your_lbank_secret
    BITRUE_API_KEY=your_bitrue_api_key
    BITRUE_SECRET=your_bitrue_secret
    ASCENDEX_API_KEY=your_ascendex_api_key
    ASCENDEX_SECRET=your_ascendex_secret
    ```

4. Configure the bot:
    Edit `config.json` to set your preferences:
    ```json
    {
      "exchange_ids": ["lbank", "bitrue", "ascendex"],
      "polling_interval": 60,
      "sandbox_mode": true,
      "wallet_addresses": {
        "lbank": "your_lbank_wallet_address",
        "bitrue": "your_bitrue_wallet_address",
        "ascendex": "your_ascendex_wallet_address"
      }
    }
    ```

## Usage

Run the bot with:

```sh
python Mex_Bot.py
```


The bot will start monitoring prices, analyzing opportunities, and executing trades based on the configured settings.
Key Components

    Portfolio: Manages the bot's holdings and tracks performance.
    SandboxExchange: Simulates exchange behavior for testing.
    RealExchange: Interfaces with actual cryptocurrency exchanges.
    LBankAPI, BitrueAPI, AscendExAPI: Custom API handlers for specific exchanges.
    SimpleMetrics: Tracks and stores various performance metrics.

Machine Learning

The bot uses a Random Forest Classifier to predict profitable trade opportunities based on historical data. It continuously refines its model as new data becomes available.
Backtesting

Historical data is used to simulate past market conditions and optimize trading strategies. The backtest function analyzes potential profits and helps determine the best threshold for executing trades.
Risk Management

The bot implements several risk management strategies:

    Position sizing based on account balance and risk tolerance
    Stop-loss orders to limit potential losses
    Diversification across multiple exchanges

Logging and Notifications

Detailed logs are saved to arbitrage_bot.log. On Windows, desktop notifications provide real-time updates on significant events.
Performance Tracking

The bot generates comprehensive performance reports, including:

    Daily, weekly, and monthly profits
    Total portfolio value and changes
    Individual coin balance changes

Disclaimer

This bot is for educational and research purposes only. Cryptocurrency trading carries significant risk. Always do your own research and never trade with money you can't afford to lose.
Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
License

This project is licensed under the MIT License - see the LICENSE file for details.
