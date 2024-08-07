import ccxt.async_support as ccxt
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.tsa.stattools as ts
import asyncio
import aiohttp
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from win10toast import ToastNotifier
from ratelimit import limits, sleep_and_retry
import os
import random
from dotenv import load_dotenv
import time
import json
from collections import defaultdict
import hmac
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DAILY_PROFIT = 'daily_profit'
WEEKLY_PROFIT = 'weekly_profit'
MONTHLY_PROFIT = 'monthly_profit'

# Load environment variables
load_dotenv()

# Configuration
with open('config.json') as config_file:
    config = json.load(config_file)
    wallet_addresses = config.get('wallet_addresses', {})

exchange_ids = config['exchange_ids']
exchange_keys = {
    exchange: {
        'apiKey': os.getenv(f'{exchange.upper()}_API_KEY'),
        'secret': os.getenv(f'{exchange.upper()}_SECRET')
    }
    for exchange in exchange_ids
}


class Portfolio:
    def __init__(self, num_exchanges):
        self.num_exchanges = num_exchanges
        self.initial_mex = 80_000_000 * num_exchanges
        self.initial_balances = {'USDT': 0, 'MEX': self.initial_mex}
        self.current_balances = self.initial_balances.copy()
        self.initial_total_value = 0
        self.start_time = datetime.now()
        self.trade_history = []

    async def initial_conversion(self, exchanges):
        for exchange in exchanges:
            try:
                ticker = await exchange.fetch_ticker('MEX/USDT')
                mex_to_convert = self.initial_mex / (2 * self.num_exchanges)
                usdt_received = mex_to_convert * ticker['last']

                # Update portfolio balances
                self.current_balances['MEX'] -= mex_to_convert
                self.current_balances['USDT'] += usdt_received

                # Update exchange balances
                await exchange.create_order('MEX/USDT', 'limit', 'sell', mex_to_convert, ticker['last'])

                logging.info(
                    f"Converted {mex_to_convert:.2f} MEX to {usdt_received:.2f} USDT on {exchange.exchange_id}")
            except Exception as e:
                logging.error(f"Error during initial conversion on {exchange.exchange_id}: {e}")

        # Update balances after conversion
        await self.update_balances(exchanges)

    async def update_balances(self, exchanges):
        total_balance = {'USDT': 0, 'MEX': 0}
        for exchange in exchanges:
            try:
                balance = await exchange.fetch_balance()
                for coin in ['USDT', 'MEX']:
                    if coin in balance['total']:
                        total_balance[coin] += balance['total'][coin]
                logging.info(f"Balance for {exchange.exchange_id}: {balance['total']}")
            except Exception as e:
                logging.error(f"Error fetching balance from {exchange.exchange_id}: {e}")

        self.current_balances = total_balance
        logging.info(f"Updated portfolio balances: {self.current_balances}")

    async def calculate_total_value(self, exchanges):
        await self.update_balances(exchanges)
        total_value = self.current_balances['USDT']
        try:
            ticker = await exchanges[0].fetch_ticker("MEX/USDT")
            mex_price = ticker['last']
            total_value += self.current_balances['MEX'] * mex_price
        except Exception as e:
            logging.error(f"Error fetching price for MEX: {e}")

        if self.initial_total_value == 0:
            self.initial_total_value = total_value

        return total_value

    def record_trade(self, trade, exchange_id):
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': trade['symbol'],
            'side': trade['side'],
            'amount': trade['amount'],
            'price': trade['price'],
            'exchange': exchange_id
        })
    def get_coin_changes(self):
        changes = {}
        for coin in self.current_balances:
            initial = self.initial_balances[coin]
            current = self.current_balances[coin]
            change = current - initial
            percent_change = (change / initial * 100) if initial > 0 else 0
            changes[coin] = {
                'initial': initial,
                'current': current,
                'change': change,
                'percent_change': percent_change
            }
        return changes

    def get_total_value_change(self, current_total_value):
        change = current_total_value - self.initial_total_value
        percent_change = (change / self.initial_total_value * 100) if self.initial_total_value > 0 else 0
        return change, percent_change

    def get_performance_report(self):
        total_mex_bought = sum(trade['amount'] for trade in self.trade_history if trade['side'] == 'buy')
        total_mex_sold = sum(trade['amount'] for trade in self.trade_history if trade['side'] == 'sell')
        total_usdt_spent = sum(
            trade['amount'] * trade['price'] for trade in self.trade_history if trade['side'] == 'buy')
        total_usdt_gained = sum(
            trade['amount'] * trade['price'] for trade in self.trade_history if trade['side'] == 'sell')

        total_profit = total_usdt_gained - total_usdt_spent

        report = f"Performance Report\n"
        report += f"Time period: {self.start_time} to {datetime.now()}\n"
        report += f"Total trades: {len(self.trade_history)}\n"
        report += f"Total MEX bought: {total_mex_bought:.0f}\n"
        report += f"Total MEX sold: {total_mex_sold:.0f}\n"
        report += f"Total USDT spent: {total_usdt_spent:.8f}\n"
        report += f"Total USDT gained: {total_usdt_gained:.8f}\n"
        report += f"Total profit: {total_profit:.8f} USDT\n\n"

        report += "Balance Changes:\n"
        for coin in ['MEX', 'USDT']:
            initial = self.initial_balances[coin]
            current = self.current_balances[coin]
            change = current - initial
            percent_change = (change / initial * 100) if initial > 0 else 0
            if coin == 'MEX':
                report += f"{coin}: Initial: {initial:.0f}, Current: {current:.0f}, "
                report += f"Change: {change:.0f} ({percent_change:.2f}%)\n"
            else:
                report += f"{coin}: Initial: {initial:.8f}, Current: {current:.8f}, "
                report += f"Change: {change:.8f} ({percent_change:.2f}%)\n"

        return report
class SimpleMetrics:
    def __init__(self):
        self.metrics = defaultdict(float)
        self.start_time = time.time()

    def set(self, name, value):
        self.metrics[name] = value

    def inc(self, name, value=1):
        self.metrics[name] += value

    def dec(self, name, value=1):
        self.metrics[name] -= value

    def get_all(self):
        return dict(self.metrics)

    def save_to_file(self, filename='metrics.json'):
        with open(filename, 'w') as f:
            json.dump(self.get_all(), f)

# Initialize the metrics object
metrics = SimpleMetrics()

import random
from datetime import datetime

class SandboxExchange:
    def __init__(self, exchange_id):
        self.exchange_id = exchange_id
        self.balances = {'USDT': 0, 'MEX': 80_000_000}
        self.orders = []
        self.trades = []
        self.last_price = 0.00000345  # Starting price

    async def fetch_ticker(self, symbol):
        if symbol == 'MEX/USDT':
            # Generate a small random price change
            price_change = random.uniform(-0.00000005, 0.00000005)
            self.last_price += price_change

            # Ensure the price doesn't go below 0
            self.last_price = max(self.last_price, 0.00000001)

            return {
                'symbol': symbol,
                'last': self.last_price,
                'bid': self.last_price * 0.9999,  # Simulating a small spread
                'ask': self.last_price * 1.0001
            }
        else:
            raise ValueError(f"Unsupported symbol: {symbol}")

    async def create_order(self, symbol, type, side, amount, price=None):
        if symbol != 'MEX/USDT':
            raise ValueError(f"Unsupported symbol: {symbol}")

        if type != 'limit':
            raise ValueError(f"Unsupported order type: {type}")

        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}")

        if price is None:
            price = self.last_price

        order_id = f"sandbox-{len(self.orders) + 1}"
        order = {
            'id': order_id,
            'symbol': symbol,
            'type': type,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'closed',  # Assume all orders are filled immediately
            'timestamp': datetime.now().timestamp()
        }
        self.orders.append(order)

        # Update balances
        if side == 'buy':
            cost = amount * price
            if self.balances['USDT'] >= cost:
                self.balances['USDT'] -= cost
                self.balances['MEX'] += amount
            else:
                raise ValueError(f"Insufficient USDT balance. Required: {cost}, Available: {self.balances['USDT']}")
        else:  # sell
            if self.balances['MEX'] >= amount:
                self.balances['MEX'] -= amount
                self.balances['USDT'] += amount * price
            else:
                raise ValueError(f"Insufficient MEX balance. Required: {amount}, Available: {self.balances['MEX']}")

        # Record the trade
        trade = {
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'cost': amount * price,
            'timestamp': order['timestamp']
        }
        self.trades.append(trade)

        return order

    async def fetch_balance(self):
        return {
            'total': self.balances.copy(),
            'free': self.balances.copy(),  # In this simple simulation, all balance is free
            'used': {currency: 0 for currency in self.balances}
        }

    def get_trade_history(self):
        return self.trades

    async def fetch_order(self, id):
        for order in self.orders:
            if order['id'] == id:
                return order
        raise ValueError(f"Order not found: {id}")

    async def fetch_orders(self, symbol=None, since=None, limit=None):
        orders = self.orders
        if symbol:
            orders = [order for order in orders if order['symbol'] == symbol]
        if since:
            orders = [order for order in orders if order['timestamp'] >= since]
        if limit:
            orders = orders[:limit]
        return orders

    async def fetch_trades(self, symbol=None, since=None, limit=None):
        trades = self.trades
        if symbol:
            trades = [trade for trade in trades if trade['symbol'] == symbol]
        if since:
            trades = [trade for trade in trades if trade['timestamp'] >= since]
        if limit:
            trades = trades[:limit]
        return trades

    async def cancel_order(self, id):
        for i, order in enumerate(self.orders):
            if order['id'] == id:
                if order['status'] == 'closed':
                    raise ValueError("Cannot cancel a closed order")
                cancelled_order = self.orders.pop(i)
                cancelled_order['status'] = 'canceled'
                return cancelled_order
        raise ValueError(f"Order not found: {id}")
class RealExchange:
    def __init__(self, exchange_id, api_key, secret, wallet_address):
        self.exchange_id = exchange_id
        self.wallet_address = wallet_address
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })

    async def transfer(self, coin, amount, to_exchange):
        try:
            # Withdraw from this exchange to the destination wallet address
            withdrawal = await self.exchange.withdraw(coin, amount, to_exchange.wallet_address)

            # Wait for confirmation (this would need to be implemented based on the exchange's API)
            await self.wait_for_withdrawal_confirmation(withdrawal['id'])

            return True
        except Exception as e:
            logging.error(f"Transfer failed: {e}")
            return False

    async def get_deposit_address(self, coin):
        return self.wallet_address


async def initial_conversion(self, exchanges):
    for exchange in exchanges:
        try:
            ticker = await exchange.fetch_ticker('MEX/USDT')
            mex_to_convert = self.initial_mex / (2 * self.num_exchanges)
            usdt_received = mex_to_convert * ticker['last']

            # Update balances
            self.current_balances['MEX'] -= mex_to_convert
            self.current_balances['USDT'] += usdt_received

            # Update the exchange's balances as well
            await exchange.create_order('MEX/USDT', 'market', 'sell', mex_to_convert)

            exchange_id = getattr(exchange, 'exchange_id', getattr(exchange, 'id', 'Unknown'))
            logging.info(f"Converted {mex_to_convert:.2f} MEX to {usdt_received:.2f} USDT on {exchange_id}")
        except Exception as e:
            exchange_id = getattr(exchange, 'exchange_id', getattr(exchange, 'id', 'Unknown'))
            logging.error(f"Error during initial conversion on {exchange_id}: {e}")

        self.current_balances = total_balance
        logging.info(f"Updated balances: {self.current_balances}")
    async def calculate_total_value(self, exchanges):
        await self.update_balances(exchanges)
        total_value = self.current_balances['USDT']
        try:
            ticker = await exchanges[0].fetch_ticker("MEX/USDT")
            mex_price = ticker['last']
            total_value += self.current_balances['MEX'] * mex_price
        except Exception as e:
            logging.error(f"Error fetching price for MEX: {e}")

        if self.initial_total_value == 0:
            self.initial_total_value = total_value

        return total_value

    def record_trade(self, trade):
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': trade['symbol'],
            'side': trade['side'],
            'amount': trade['amount'],
            'price': trade['price'],
            'exchange': trade['exchange']
        })

    def get_coin_changes(self):
        changes = {}
        for coin in self.current_balances:
            initial = self.initial_balances[coin]
            current = self.current_balances[coin]
            change = current - initial
            percent_change = (change / initial * 100) if initial > 0 else 0
            changes[coin] = {
                'initial': initial,
                'current': current,
                'change': change,
                'percent_change': percent_change
            }
        return changes

    def get_total_value_change(self, current_total_value):
        change = current_total_value - self.initial_total_value
        percent_change = (change / self.initial_total_value * 100) if self.initial_total_value > 0 else 0
        return change, percent_change


    def get_performance_report(self):
        total_mex_bought = sum(trade['amount'] for trade in self.trade_history if trade['side'] == 'buy')
        total_mex_sold = sum(trade['amount'] for trade in self.trade_history if trade['side'] == 'sell')
        total_usdt_spent = sum(trade['amount'] * trade['price'] for trade in self.trade_history if trade['side'] == 'buy')
        total_usdt_gained = sum(trade['amount'] * trade['price'] for trade in self.trade_history if trade['side'] == 'sell')

        total_profit = total_usdt_gained - total_usdt_spent

        report = f"Performance Report\n"
        report += f"Time period: {self.start_time} to {datetime.now()}\n"
        report += f"Total trades: {len(self.trade_history)}\n"
        report += f"Total MEX bought: {total_mex_bought:.0f}\n"
        report += f"Total MEX sold: {total_mex_sold:.0f}\n"
        report += f"Total USDT spent: {total_usdt_spent:.8f}\n"
        report += f"Total USDT gained: {total_usdt_gained:.8f}\n"
        report += f"Total profit: {total_profit:.8f} USDT\n\n"

        report += "Balance Changes:\n"
        for coin in ['MEX', 'USDT']:
            initial = self.initial_balances[coin]
            current = self.current_balances[coin]
            change = current - initial
            percent_change = (change / initial * 100) if initial > 0 else 0
            if coin == 'MEX':
                report += f"{coin}: Initial: {initial:.0f}, Current: {current:.0f}, "
                report += f"Change: {change:.0f} ({percent_change:.2f}%)\n"
            else:
                report += f"{coin}: Initial: {initial:.8f}, Current: {current:.8f}, "
                report += f"Change: {change:.8f} ({percent_change:.2f}%)\n"

        return report
class LBankAPI:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.base_url = 'https://api.lbkex.com'
        self.exchange = ccxt.lbank({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })

    def _sign_request(self, params):
        params['api_key'] = self.api_key
        params['timestamp'] = str(int(time.time() * 1000))
        params = dict(sorted(params.items()))

        sign_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(self.secret.encode(), sign_string.encode(), hashlib.md5).hexdigest().upper()
        params['sign'] = signature
        return params

    async def fetch_ticker(self, symbol):
        return await self.exchange.fetch_ticker(symbol)

    @sleep_and_retry
    @limits(calls=500, period=10)
    async def create_order(self, symbol, type, side, amount, price=None):
        params = {
            'symbol': symbol,
            'type': type,
            'side': side,
            'amount': amount,
            'price': price
        }
        signed_params = self._sign_request(params)
        response = await self.exchange.request('POST', '/v2/supplement/create_order.do', signed_params)
        return response

    @sleep_and_retry
    @limits(calls=200, period=10)
    async def fetch_balance(self):
        params = {}
        signed_params = self._sign_request(params)
        response = await self.exchange.request('POST', '/v2/supplement/user_info_account.do', signed_params)
        return response

class BitrueAPI:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.base_url = 'https://fapi.bitrue.com'
        self.exchange = ccxt.bitrue({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })

    def _sign_request(self, endpoint, params=None):
        timestamp = int(time.time() * 1000)
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()]) if params else ""
        signature = hmac.new(self.secret.encode(), f"{timestamp}{query_string}".encode(), hashlib.sha256).hexdigest()
        return {
            'X-CH-APIKEY': self.api_key,
            'X-CH-TS': str(timestamp),
            'X-CH-SIGN': signature
        }

    async def fetch_ticker(self, symbol):
        return await self.exchange.fetch_ticker(symbol)

    async def create_order(self, symbol, type, side, amount, price=None):
        return await self.exchange.create_order(symbol, type, side, amount, price)

    async def fetch_balance(self):
        return await self.exchange.fetch_balance()
class AscendExAPI:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.base_url = 'https://ascendex.com'
        self.exchange = ccxt.ascendex({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })

    def _sign_request(self, endpoint, params=None):
        timestamp = int(time.time() * 1000)
        message = f"{timestamp}+{endpoint}"
        signature = hmac.new(self.secret.encode(), message.encode(), hashlib.sha256).hexdigest()
        return {
            'x-auth-key': self.api_key,
            'x-auth-timestamp': str(timestamp),
            'x-auth-signature': signature
        }

    async def fetch_ticker(self, symbol):
        return await self.exchange.fetch_ticker(symbol)

    async def create_order(self, symbol, type, side, amount, price=None):
        return await self.exchange.create_order(symbol, type, side, amount, price)

    async def fetch_balance(self):
        return await self.exchange.fetch_balance()


def __init__(self):
    self.initial_balances = {'USDT': 0, 'MEX': 0}
    self.current_balances = {'USDT': 0, 'MEX': 0}
    self.initial_total_value = 0
    self.start_time = datetime.now()
    self.trade_history = []
    async def update_balances(self, exchanges):
        total_balance = {}
        for exchange in exchanges:
            try:
                balance = await exchange.fetch_balance()
                for coin, amount in balance['total'].items():
                    if coin not in total_balance:
                        total_balance[coin] = 0
                    total_balance[coin] += amount
            except Exception as e:
                logging.error(f"Error fetching balance from {exchange.id}: {e}")

        if not self.initial_balances:
            self.initial_balances = total_balance.copy()
        self.current_balances = total_balance

    async def calculate_total_value(self, exchanges):
        total_value = self.current_balances['USDT']
        try:
            ticker = await exchanges[0].fetch_ticker("MEX/USDT")
            mex_price = ticker['last']
            total_value += self.current_balances['MEX'] * mex_price
        except Exception as e:
            logging.error(f"Error fetching price for MEX: {e}")

        if self.initial_total_value == 0:
            self.initial_total_value = total_value

        return total_value

    def get_coin_changes(self):
        changes = {}
        for coin in self.current_balances:
            if coin in self.initial_balances:
                initial = self.initial_balances[coin]
                current = self.current_balances[coin]
                change = current - initial
                percent_change = (change / initial * 100) if initial > 0 else 0
                changes[coin] = {
                    'initial': initial,
                    'current': current,
                    'change': change,
                    'percent_change': percent_change
                }
            else:
                changes[coin] = {
                    'initial': 0,
                    'current': self.current_balances[coin],
                    'change': self.current_balances[coin],
                    'percent_change': 100
                }
        return changes

    def get_total_value_change(self, current_total_value):
        change = current_total_value - self.initial_total_value
        percent_change = (change / self.initial_total_value * 100) if self.initial_total_value > 0 else 0
        return change, percent_change

    def record_trade(self, trade):
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': trade['symbol'],
            'side': trade['side'],
            'amount': trade['amount'],
            'price': trade['price'],
            'exchange': trade['exchange']
        })

    def get_performance_report(self):
        total_mex_bought = sum(trade['amount'] for trade in self.trade_history if trade['side'] == 'buy')
        total_mex_sold = sum(trade['amount'] for trade in self.trade_history if trade['side'] == 'sell')
        total_usdt_spent = sum(
            trade['amount'] * trade['price'] for trade in self.trade_history if trade['side'] == 'buy')
        total_usdt_gained = sum(
            trade['amount'] * trade['price'] for trade in self.trade_history if trade['side'] == 'sell')

        total_profit = total_usdt_gained - total_usdt_spent

        report = f"Performance Report\n"
        report += f"Time period: {self.start_time} to {datetime.now()}\n"
        report += f"Total trades: {len(self.trade_history)}\n"
        report += f"Total MEX bought: {total_mex_bought:.2f}\n"
        report += f"Total MEX sold: {total_mex_sold:.2f}\n"
        report += f"Total USDT spent: {total_usdt_spent:.2f}\n"
        report += f"Total USDT gained: {total_usdt_gained:.2f}\n"
        report += f"Total profit: {total_profit:.2f} USDT\n\n"

        report += "Balance Changes:\n"
        for coin in ['MEX', 'USDT']:
            initial = self.initial_balances[coin]
            current = self.current_balances[coin]
            change = current - initial
            percent_change = (change / initial * 100) if initial > 0 else 0
            report += f"{coin}: Initial: {initial:.2f}, Current: {current:.2f}, "
            report += f"Change: {change:.2f} ({percent_change:.2f}%)\n"

        return report

# Set up logging
logging.basicConfig(filename='arbitrage_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
backtest_data = {}
trade_history = []
profit_history = {'daily': [], 'weekly': [], 'monthly': []}
toaster = ToastNotifier()

# Metric names
DAILY_PROFIT = 'daily_profit'
WEEKLY_PROFIT = 'weekly_profit'
MONTHLY_PROFIT = 'monthly_profit'


async def fetch_price(session, exchange_id, exchange, symbol='MEX/USDT', retries=3):
    for attempt in range(retries):
        try:
            if isinstance(exchange, SandboxExchange):
                ticker = await exchange.fetch_ticker(symbol)
            elif exchange_id == 'ascendex':
                ascendex_api = AscendExAPI(exchange_keys['ascendex']['apiKey'], exchange_keys['ascendex']['secret'])
                ticker = await ascendex_api.fetch_ticker(symbol)
            elif exchange_id == 'bitrue':
                bitrue_api = BitrueAPI(exchange_keys['bitrue']['apiKey'], exchange_keys['bitrue']['secret'])
                ticker = await bitrue_api.fetch_ticker(symbol)
            elif exchange_id == 'lbank':
                lbank_api = LBankAPI(exchange_keys['lbank']['apiKey'], exchange_keys['lbank']['secret'])
                ticker = await lbank_api.fetch_ticker(symbol)
            else:
                ticker = await exchange.fetch_ticker(symbol)
            return exchange_id, ticker['last']
        except Exception as e:
            logging.warning(f"Error fetching {symbol} price from {exchange_id}: {e}")
            if attempt == retries - 1:
                return exchange_id, None
            await asyncio.sleep(2 ** attempt)

async def get_prices(symbol='MEX/USDT'):
    async with aiohttp.ClientSession() as session:
        if config['sandbox_mode']:
            exchanges = [SandboxExchange(exchange_id) for exchange_id in exchange_ids]
            logging.info("Running in sandbox mode")
        else:
            exchanges = [
                ccxt.Exchange({
                    'id': exchange_id,
                    'apiKey': exchange_keys[exchange_id]['apiKey'],
                    'secret': exchange_keys[exchange_id]['secret'],
                    'enableRateLimit': True,
                    'asyncio_session': session,
                })
                for exchange_id in exchange_ids
            ]

        tasks = [fetch_price(session, exchange.exchange_id if isinstance(exchange, SandboxExchange) else exchange.id, exchange, symbol) for exchange in exchanges]
        results = await asyncio.gather(*tasks)

        return {exchange_id: price for exchange_id, price in results if price is not None}
def calculate_arbitrage(prices):
    if len(prices) < 2:
        return 0, None, None, 0, 0

    min_price_exchange = min(prices, key=prices.get)
    max_price_exchange = max(prices, key=prices.get)
    min_price = prices[min_price_exchange]
    max_price = prices[max_price_exchange]

    price_difference = max_price - min_price
    percentage_difference = (price_difference / min_price) * 100
    mex_amount = 40_000_000
    potential_profit_usdt = price_difference * mex_amount

    logging.info(f"Arbitrage opportunity: {min_price_exchange} -> {max_price_exchange}")
    logging.info(f"Price difference: {price_difference:.8f} USDT")
    logging.info(f"Percentage difference: {percentage_difference:.2f}%")
    logging.info(f"Potential profit for 40M MEX: {potential_profit_usdt:.2f} USDT")

    return percentage_difference, min_price_exchange, max_price_exchange, price_difference, potential_profit_usdt
async def execute_trade(exchange, symbol, side, amount, price):
    logging.info(f"Executing trade: {exchange.id} {side} {amount} {symbol} at {price}")
    try:
        order = await exchange.create_order(symbol, 'limit', side, amount, price)
        logging.info(f"Order created: {order}")
        return order
    except Exception as e:
        logging.error(f"Error executing trade on {exchange.id}: {e}", exc_info=True)
        return None


async def execute_arbitrage(min_price_exchange, max_price_exchange, min_price, max_price, symbol='MEX/USDT'):
    logging.info(
        f"Executing arbitrage: {min_price_exchange} (buy at {min_price}) -> {max_price_exchange} (sell at {max_price})")
    try:
        if config['sandbox_mode']:
            exchange_min = next(ex for ex in exchanges if ex.exchange_id == min_price_exchange)
            exchange_max = next(ex for ex in exchanges if ex.exchange_id == max_price_exchange)
        else:
            exchange_min = next(ex for ex in exchanges if ex.id == min_price_exchange)
            exchange_max = next(ex for ex in exchanges if ex.id == max_price_exchange)

        # Calculate position size based on account balance and risk
        balance = await exchange_min.fetch_balance()
        available_balance = balance['total']['USDT']
        risk_per_trade = 0.01  # 1% risk per trade
        position_size = available_balance * risk_per_trade / (max_price - min_price)

        logging.info(f"Calculated position size: {position_size}")

        # Execute buy order
        buy_order = await execute_trade(exchange_min, symbol, 'buy', position_size, min_price)
        if buy_order:
            logging.info(f"Buy order executed: {buy_order}")
            # Execute sell order
            sell_order = await execute_trade(exchange_max, symbol, 'sell', position_size, max_price)
            if sell_order:
                logging.info(f"Sell order executed: {sell_order}")
                profit = (max_price - min_price) * position_size
                logging.info(f"Arbitrage executed. Profit: {profit:.8f} USDT")
                return profit
            else:
                logging.warning("Sell order failed. Closing buy position.")
                # If sell fails, close the buy position
                await execute_trade(exchange_min, symbol, 'sell', position_size, min_price)
        else:
            logging.warning("Buy order failed.")

        return 0
    except Exception as e:
        logging.error(f"Error in execute_arbitrage: {e}", exc_info=True)
        return 0
async def fetch_historical_data(exchange, symbol, start_date):
    try:
        logging.debug(f"Fetching historical data for {exchange.id}")
        if config['sandbox_mode']:
            return generate_mock_data(start_date)
        elif exchange.id == 'ascendex':
            ascendex_api = AscendExAPI(exchange_keys['ascendex']['apiKey'], exchange_keys['ascendex']['secret'])
            ohlcv = await ascendex_api.exchange.fetch_ohlcv(symbol, '1h', int(start_date.timestamp() * 1000), limit=1000)
        elif exchange.id == 'bitrue':
            bitrue_api = BitrueAPI(exchange_keys['bitrue']['apiKey'], exchange_keys['bitrue']['secret'])
            ohlcv = await bitrue_api.exchange.fetch_ohlcv(symbol, '1h', int(start_date.timestamp() * 1000), limit=1000)
        elif exchange.id == 'lbank':
            lbank_api = LBankAPI(exchange_keys['lbank']['apiKey'], exchange_keys['lbank']['secret'])
            ohlcv = await lbank_api.exchange.fetch_ohlcv(symbol, '1h', int(start_date.timestamp() * 1000), limit=1000)
        else:
            ohlcv = await exchange.fetch_ohlcv(symbol, '1h', int(start_date.timestamp() * 1000), limit=1000)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        logging.debug(f"Successfully fetched data for {exchange.id}: {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data from {exchange.id}: {e}")
        return None

def generate_mock_data(start_date):
    logging.debug("Generating mock historical data")
    end_date = datetime.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq='1h')
    mock_data = []
    last_price = 10000  # Starting price
    for date in date_range:
        change = np.random.normal(0, 100)  # Random price change
        last_price += change
        mock_data.append([date.timestamp() * 1000, last_price, last_price + 50, last_price - 50, last_price, 1000])
    df = pd.DataFrame(mock_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    logging.debug(f"Generated mock data: {len(df)} rows")
    return df

async def fetch_all_historical_data(symbol='MEX/USDT'):
    logging.info("Starting fetch_all_historical_data")
    start_date = datetime.now() - timedelta(days=180)  # 6 months ago
    logging.debug(f"Fetching data from {start_date} to now for symbol {symbol}")

    async with aiohttp.ClientSession() as session:
        exchanges = [
            ccxt.Exchange({
                'id': exchange_id,
                'apiKey': exchange_keys[exchange_id]['apiKey'],
                'secret': exchange_keys[exchange_id]['secret'],
                'enableRateLimit': True,
                'asyncio_session': session,
            })
            for exchange_id in exchange_ids
        ]

        tasks = [fetch_historical_data(exchange, symbol, start_date) for exchange in exchanges]
        results = await asyncio.gather(*tasks)

        for exchange in exchanges:
            await exchange.close()

        historical_data = {exchange.id: df for exchange, df in zip(exchanges, results) if df is not None}

        logging.info(f"Finished fetch_all_historical_data. Data fetched for {len(historical_data)} exchanges.")
        for exchange, data in historical_data.items():
            if data is not None:
                logging.debug(
                    f"Data for {exchange}: shape {data.shape}, date range {data.index.min()} to {data.index.max()}")
            else:
                logging.warning(f"No data fetched for {exchange}")

        return historical_data

def backtest(historical_data):
    logging.debug(f"Starting backtest with data for {len(historical_data)} exchanges")
    results = []
    if not historical_data or all(df.empty for df in historical_data.values()):
        logging.error("Historical data is empty or all DataFrames are empty")
        return pd.DataFrame()

    reference_exchange = next(iter(historical_data))
    for date in historical_data[reference_exchange].index:
        prices = {exchange: data.loc[date, 'close'] for exchange, data in historical_data.items() if date in data.index}
        if len(prices) >= 2:
            arbitrage_profit, min_exchange, max_exchange, price_diff, potential_profit = calculate_arbitrage(prices)
            results.append({
                'date': date,
                'arbitrage_profit': arbitrage_profit,
                'min_exchange': min_exchange,
                'max_exchange': max_exchange,
                'price_difference': price_diff,
                'potential_profit': potential_profit
            })

    logging.debug(f"Finished backtest. Generated {len(results)} results.")
    return pd.DataFrame(results)

def optimize_strategy(backtest_results):
    if 'arbitrage_profit' not in backtest_results.columns:
        logging.error("'arbitrage_profit' column not found in backtest results")
        return 0, 0

    thresholds = np.arange(0.01, 1.0, 0.01)  # Adjusted thresholds
    best_threshold = 0
    best_profit = 0

    for threshold in thresholds:
        profitable_trades = backtest_results[backtest_results['arbitrage_profit'] > threshold]
        total_profit = profitable_trades['potential_profit'].sum()

        if total_profit > best_profit:
            best_profit = total_profit
            best_threshold = threshold

    return best_threshold, best_profit
def calculate_profits(portfolio):
    now = datetime.now()
    daily_cutoff = now - timedelta(days=1)
    weekly_cutoff = now - timedelta(weeks=1)
    monthly_cutoff = now - timedelta(days=30)

    daily_profit = sum(trade['amount'] * (trade['price'] if trade['side'] == 'sell' else -trade['price'])
                       for trade in portfolio.trade_history if trade['time'] > daily_cutoff)
    weekly_profit = sum(trade['amount'] * (trade['price'] if trade['side'] == 'sell' else -trade['price'])
                        for trade in portfolio.trade_history if trade['time'] > weekly_cutoff)
    monthly_profit = sum(trade['amount'] * (trade['price'] if trade['side'] == 'sell' else -trade['price'])
                         for trade in portfolio.trade_history if trade['time'] > monthly_cutoff)

    return daily_profit, weekly_profit, monthly_profit


def prepare_ml_data(portfolio):
    logging.info("Preparing ML data...")
    df = pd.DataFrame(portfolio.trade_history)

    if df.empty:
        logging.info("Trade history is empty.")
        return None

    logging.info(f"Trade history columns: {df.columns.tolist()}")
    logging.info(f"Sample data:\n{df.head().to_string()}")
    logging.info(f"Data types:\n{df.dtypes}")
    logging.info(f"Number of trades in dataset: {len(df)}")

    # Here, add the necessary data preparation steps
    df['timestamp'] = pd.to_datetime(df['time'])
    df = df.sort_values('timestamp')
    df['average_price'] = df['price']
    df['returns'] = df.groupby('symbol')['price'].pct_change()
    df['volatility'] = df.groupby('symbol')['returns'].rolling(window=min(10, len(df))).std().reset_index(0, drop=True)

    # You'll need to define how to calculate 'target' based on your strategy
    # For example, you could use future returns:
    df['target'] = df.groupby('symbol')['returns'].shift(-1) > 0

    return df.dropna()
def train_ml_model(df):
    X = df[['average_price', 'returns', 'volatility']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"ML model accuracy: {accuracy:.2f}")

    return model


def show_notifications():
    if trade_history:
        last_trade = trade_history[-1]
        message = f"Last trade profit: {last_trade['profit']:.2f} USDT, {last_trade['profit_percentage']:.2f}%"
        toaster.show_toast("Arbitrage Bot Notification", message, duration=10)
    else:
        toaster.show_toast("Arbitrage Bot Notification", "No trades executed yet", duration=10)


async def main():
    exchanges = []  # Initialize exchanges at the beginning of the function
    try:
        logging.info("Starting main function")

        # Fetch historical data and perform backtesting initially
        historical_data = await fetch_all_historical_data()
        backtest_results = backtest(historical_data)
        best_threshold, best_profit = optimize_strategy(backtest_results)

        logging.info(
            f"Backtesting results: Best threshold = {best_threshold:.2f}%, Best total profit = {best_profit:.2f}%")

        portfolio = Portfolio(num_exchanges=len(exchange_ids))

        # Initialize exchanges
        if config['sandbox_mode']:
            exchanges = [SandboxExchange(exchange_id) for exchange_id in exchange_ids]
            logging.info("Running in sandbox mode")
        else:
            exchanges = [
                RealExchange(
                    exchange_id,
                    exchange_keys[exchange_id]['apiKey'],
                    exchange_keys[exchange_id]['secret'],
                    wallet_addresses.get(exchange_id, '')
                )
                for exchange_id in exchange_ids
            ]
            logging.info("Running in live mode")

        # Perform initial conversion
        await portfolio.initial_conversion(exchanges)
        await portfolio.update_balances(exchanges)

        while True:
            try:
                logging.info("Fetching current prices...")
                prices = await get_prices()

                if not prices:
                    logging.warning("Failed to fetch prices from any exchange. Skipping this iteration.")
                    await asyncio.sleep(config['polling_interval'])
                    continue

                logging.info(f"Fetched prices: {prices}")
                logging.info(f"Current balances before trade:")
                for exchange in exchanges:
                    logging.info(f"{exchange.exchange_id}: {exchange.balances}")

                if len(prices) >= 2:
                    arbitrage_profit, min_price_exchange, max_price_exchange, price_diff, potential_profit = calculate_arbitrage(
                        prices)
                    metrics.set('arbitrage_profit', arbitrage_profit)
                    should_execute = False

                    if arbitrage_profit > best_threshold:
                        try:
                            df = prepare_ml_data(portfolio)
                            if df is not None and len(df) >= 100:  # Only proceed with ML if we have enough data
                                logging.info(f"Prepared ML data shape: {df.shape}")

                                # Split the data for training and evaluation
                                X = df[['average_price', 'returns', 'volatility']]
                                y = df['target']
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                                    random_state=42)

                                # Train the model
                                model = train_ml_model(X_train, y_train)

                                # Evaluate the model
                                accuracy = model.score(X_test, y_test)
                                logging.info(f"Current model accuracy: {accuracy:.2f}")

                                # Make prediction
                                latest_data = df.iloc[-1][['average_price', 'returns', 'volatility']]
                                prediction = model.predict([latest_data])[0]

                                if prediction:
                                    logging.info(f"ML model predicts profitable trade. Executing arbitrage...")
                                    should_execute = True
                                else:
                                    logging.info("ML model predicts unprofitable trade. Skipping arbitrage.")
                            else:
                                logging.info(f"Not enough data for ML prediction. Using fallback strategy.")
                                # Simple fallback strategy
                                should_execute = arbitrage_profit > 0.5  # You can adjust this threshold
                                if should_execute:
                                    logging.info(f"Fallback strategy suggests executing arbitrage.")
                                else:
                                    logging.info(f"Fallback strategy suggests skipping arbitrage.")

                            if should_execute:
                                # Calculate position size (you may want to adjust this logic)
                                position_size = min(exchanges[0].balances['MEX'],
                                                    exchanges[1].balances['MEX']) * 0.1  # Use 10% of available MEX

                                # Execute buy order on the lower-priced exchange
                                buy_exchange = next(ex for ex in exchanges if ex.exchange_id == min_price_exchange)
                                buy_order = await buy_exchange.create_order('MEX/USDT', 'limit', 'buy', position_size,
                                                                            prices[min_price_exchange])

                                # Execute sell order on the higher-priced exchange
                                sell_exchange = next(ex for ex in exchanges if ex.exchange_id == max_price_exchange)
                                sell_order = await sell_exchange.create_order('MEX/USDT', 'limit', 'sell',
                                                                              position_size, prices[max_price_exchange])

                                profit = (prices[max_price_exchange] - prices[min_price_exchange]) * position_size
                                logging.info(f"Arbitrage executed. Profit: {profit:.8f} USDT")

                                # Record trades in portfolio
                                portfolio.record_trade(buy_order, buy_exchange.exchange_id)
                                portfolio.record_trade(sell_order, sell_exchange.exchange_id)

                                # Update balances
                                await portfolio.update_balances(exchanges)
                            else:
                                logging.info("Skipping arbitrage based on prediction or fallback strategy.")

                        except Exception as e:
                            logging.error(f"Error in ML prediction or trade execution: {e}", exc_info=True)

                await portfolio.update_balances(exchanges)
                current_total_value = await portfolio.calculate_total_value(exchanges)

                logging.info(f"Updated balances after trade:")
                for exchange in exchanges:
                    logging.info(f"{exchange.exchange_id}: {exchange.balances}")

                coin_changes = portfolio.get_coin_changes()
                total_value_change, total_value_percent_change = portfolio.get_total_value_change(current_total_value)

                report = portfolio.get_performance_report()
                report += f"\nTotal Portfolio Value: ${current_total_value:.2f}\n"
                report += f"Total Value Change: ${total_value_change:+.2f} ({total_value_percent_change:+.2f}%)\n"

                daily_profit, weekly_profit, monthly_profit = calculate_profits(portfolio)
                report += f"\nDaily Profit: ${daily_profit:.2f}\n"
                report += f"Weekly Profit: ${weekly_profit:.2f}\n"
                report += f"Monthly Profit: ${monthly_profit:.2f}\n"

                logging.info(report)

                metrics.set(DAILY_PROFIT, daily_profit)
                metrics.set(WEEKLY_PROFIT, weekly_profit)
                metrics.set(MONTHLY_PROFIT, monthly_profit)
                metrics.set('total_portfolio_value', current_total_value)
                metrics.set('total_value_change', total_value_change)

                metrics.save_to_file('arbitrage_metrics.json')

                await asyncio.sleep(config['polling_interval'])

            except ccxt.ExchangeError as e:
                if 'LBank' in str(e):
                    logging.error(f"LBank specific error: {e}")
                else:
                    logging.error(f"Exchange error: {e}")
                await asyncio.sleep(60)

            except Exception as e:
                logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    except Exception as e:
        logging.error(f"Critical error in main function: {e}", exc_info=True)

    finally:
        for exchange in exchanges:
            if hasattr(exchange, 'close'):
                await exchange.close()
if __name__ == "__main__":
    asyncio.run(main())
