# This is the original QuantConnect/LEAN strategy reference.
# It is not executed by the Streamlit app.
# The Streamlit app converts the indicator and signal ideas into advisory dashboard logic only.

from AlgorithmImports import *
import numpy as np


class LinearAlphaKCSqueeze(QCAlgorithm):
    def initialize(self):
        # 1. Backtest Settings
        self.set_start_date(2020, 1, 1)
        self.set_cash(1000000)

        # 2. Asset & Resolution
        self._security = self.add_cfd("XAUUSD", Resolution.HOUR, Market.OANDA)
        self._symbol = self._security.symbol

        # 3. Trade Settings
        self._max_leverage = 10.0
        self._security.set_leverage(self._max_leverage)
        self._risk_per_trade = 0.05  # 5% Account Risk
        self._atr_multiplier = 3.0   # Stop Loss Distance
        self._tp_multiplier = 6.0    # Take Profit Distance (2:1 Reward-to-Risk)

        # 4. Parameters
        self._bb_length = 20
        self._bb_mult = 2.0
        self._kc_length = 20
        self._kc_mult = 1.5
        self._trend_length = 200

        # 5. Indicators
        self._bb = self.bb(self._symbol, self._bb_length, self._bb_mult, MovingAverageType.SIMPLE)
        self._kc = self.kch(self._symbol, self._kc_length, self._kc_mult, MovingAverageType.SIMPLE)
        self._trend_sma = self.sma(self._symbol, self._trend_length, Resolution.HOUR)
        self._atr = self.atr(self._symbol, 14, MovingAverageType.WILDERS, Resolution.HOUR)

        # 6. Momentum Calculation Helpers
        self._max = self.max(self._symbol, self._kc_length, Resolution.HOUR, Field.HIGH)
        self._min = self.min(self._symbol, self._kc_length, Resolution.HOUR, Field.LOW)
        self._sma = self.sma(self._symbol, self._kc_length, Resolution.HOUR)
        self._mom_source_window = RollingWindow[float](self._kc_length)
        self._val_window = RollingWindow[float](2)
        self._sqz_on_window = RollingWindow[bool](2)
        self._trailing_stop = 0
        self._take_profit = 0
        self.set_warm_up(self._trend_length, Resolution.HOUR)

    def on_data(self, data: Slice):
        if self._symbol not in data or data[self._symbol] is None:
            return

        if self.is_warming_up or not self._bb.is_ready or not self._atr.is_ready:
            return

        # --- A. Indicator Calculations ---
        current_sqz_on = (self._bb.lower_band.current.value >= self._kc.lower_band.current.value) and \
                         (self._bb.upper_band.current.value <= self._kc.upper_band.current.value)
        self._sqz_on_window.add(current_sqz_on)

        midline = ((self._max.current.value + self._min.current.value) / 2 + self._sma.current.value) / 2
        mom_source = self.securities[self._symbol].price - midline
        self._mom_source_window.add(mom_source)

        if not self._mom_source_window.is_ready or not self._sqz_on_window.is_ready:
            return

        y_values = np.array(list(self._mom_source_window))[::-1]
        x_values = np.arange(len(y_values))
        slope, intercept = np.polyfit(x_values, y_values, 1)
        current_val = slope * (self._kc_length - 1) + intercept
        self._val_window.add(current_val)

        if not self._val_window.is_ready:
            return

        # --- B. Strategy Logic ---
        price = self.securities[self._symbol].price

        if not self.portfolio.invested:
            # Entry logic: Volatility expansion + Trend confirmation
            squeeze_fired = self._sqz_on_window[1] and not self._sqz_on_window[0]

            if squeeze_fired:
                curr_mom = self._val_window[0]
                prev_mom = self._val_window[1]
                if curr_mom > 0 and curr_mom > prev_mom and price > self._trend_sma.current.value:
                    atr_unit = self._atr.current.value
                    stop_dist = atr_unit * self._atr_multiplier

                    # Position sizing based on risk
                    qty = (self.portfolio.total_portfolio_value * self._risk_per_trade) / stop_dist

                    # Buffer to prevent Buying Power errors
                    max_possible_qty = (self.portfolio.total_portfolio_value * (self._max_leverage - 0.2)) / price
                    final_qty = min(qty, max_possible_qty)

                    self.market_order(self._symbol, final_qty)
                    self._trailing_stop = price - stop_dist
                    self._take_profit = price + (atr_unit * self._tp_multiplier)

        else:
            # --- C. Exit Logic ---
            # 1. Take Profit (Hard Exit)
            if price >= self._take_profit:
                self.liquidate(self._symbol, "Take Profit Hit")
                return

            # 2. ATR Trailing Stop (Hard Exit)
            current_atr_buffer = self._atr.current.value * self._atr_multiplier
            self._trailing_stop = max(self._trailing_stop, price - current_atr_buffer)

            if price < self._trailing_stop:
                self.liquidate(self._symbol, "Trailing Stop Hit")
                return

            # 3. Momentum Fade (Soft Exit)
            curr_val = self._val_window[0]
            prev_val = self._val_window[1]
            if curr_val <= 0 or (curr_val < prev_val and curr_val < (0.5 * prev_val)):
                self.liquidate(self._symbol, "Momentum Fade")

    def on_warmup_finished(self):
        self.log(f"Strategy Initialized. Risk per trade: {self._risk_per_trade}")
