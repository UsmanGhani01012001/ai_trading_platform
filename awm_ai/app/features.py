"""
AWM AI Feature Engineering Module
=================================
Computes features for HMM (regime detection) and LGBM (TAKE/SKIP decision)

Based on smarttrader.ipynb training specifications:
- HMM: 19 features → 4-state regime model
- LGBM: 9 features (including HMM probabilities) → TAKE/SKIP decision
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque


class FeatureBuffer:
    """
    Rolling buffer to store historical bars for feature calculation.
    Maintains enough history for all indicators (warmup ~30 bars minimum).
    """
    
    def __init__(self, maxlen: int = 300):
        self.bars: deque = deque(maxlen=maxlen)
        self.df: Optional[pd.DataFrame] = None
        
    def add_bar(self, bar: dict) -> bool:
        """Add new bar and rebuild dataframe. Returns True if enough data."""
        self.bars.append(bar)
        self._rebuild_df()
        return len(self.bars) >= 30  # minimum warmup
        
    def _rebuild_df(self):
        """Convert deque to DataFrame for vectorized calculations"""
        if len(self.bars) > 0:
            self.df = pd.DataFrame(list(self.bars))
            self.df['time'] = pd.to_datetime(self.df['time'], format='mixed')
            self.df = self.df.sort_values('time').reset_index(drop=True)
    
    def get_df(self) -> Optional[pd.DataFrame]:
        return self.df
    
    def __len__(self):
        return len(self.bars)
    
    def clear(self):
        self.bars.clear()
        self.df = None


class IndicatorCalculator:
    """
    Calculates all technical indicators needed for HMM and LGBM models.
    All calculations are vectorized using pandas/numpy.
    """
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator - returns (macd_line, signal_line, histogram)"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX with +DI and -DI"""
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        # True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high - prev_high
        minus_dm = prev_low - low
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smoothed values (Wilder's smoothing)
        atr_smooth = tr.ewm(span=period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()
        
        # DI values
        plus_di = 100 * (plus_dm_smooth / atr_smooth.replace(0, np.nan))
        minus_di = 100 * (minus_dm_smooth / atr_smooth.replace(0, np.nan))
        
        # DX and ADX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100 * (di_diff / di_sum.replace(0, np.nan))
        adx_val = dx.ewm(span=period, adjust=False).mean()
        
        return adx_val.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> pd.Series:
        """Stochastic %K"""
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        
        denom = (highest_high - lowest_low).replace(0, np.nan)
        stoch_k = 100 * (close - lowest_low) / denom
        return stoch_k.fillna(50)
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price (cumulative for session)"""
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum().replace(0, np.nan)
        return cumulative_tp_vol / cumulative_vol
    
    @staticmethod
    def bollinger_bandwidth(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger Bandwidth"""
        sma = close.rolling(window=period, min_periods=1).mean()
        std = close.rolling(window=period, min_periods=1).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        bandwidth = (upper - lower) / sma.replace(0, np.nan)
        return bandwidth.fillna(0)
    
    @staticmethod
    def slope(series: pd.Series, period: int = 5) -> pd.Series:
        """Calculate slope over period"""
        return (series - series.shift(period)) / period


class HMMFeatureBuilder:
    """
    Builds feature vector for HMM model (regime/state detection).
    
    HMM Features (19 total):
    1. ema_fast_minus_ema_slow
    2. price_minus_vwap
    3. adx_smoothed
    4. plus_di
    5. minus_di
    6. macd_hist
    7. macd_hist_slope
    8. rsi
    9. rsi_slope
    10. atr
    11. atr_normalized
    12. bollinger_bandwidth
    13. candle_body_to_range_ratio
    14. cumulative_delta_slope
    15. volume_normalized
    16. session_phase
    17. time_bucket_id
    18. rolling_drawdown
    19. win_loss_ratio
    """
    
    HMM_FEATURE_NAMES = [
        'ema_fast_minus_ema_slow',
        'price_minus_vwap',
        'adx_smoothed',
        'plus_di',
        'minus_di',
        'macd_hist',
        'macd_hist_slope',
        'rsi',
        'rsi_slope',
        'atr',
        'atr_normalized',
        'bollinger_bandwidth',
        'candle_body_to_range_ratio',
        'cumulative_delta_slope',
        'volume_normalized',
        'session_phase',
        'time_bucket_id',
        'rolling_drawdown',
        'win_loss_ratio',
    ]
    
    # Parameters
    EMA_FAST = 9
    EMA_SLOW = 21
    
    def __init__(self):
        self.calc = IndicatorCalculator()
        # Track wins/losses for rolling metrics
        self.trade_results: deque = deque(maxlen=20)  # last 20 trades
        self.peak_equity = 0.0
        self.current_equity = 0.0
    
    def update_trade_result(self, is_win: bool, pnl: float = 0.0):
        """Call this after each trade to track win/loss ratio and drawdown"""
        self.trade_results.append(1 if is_win else 0)
        self.current_equity += pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)
    
    def build_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Build HMM feature array from bar DataFrame.
        Returns features for the LATEST bar as numpy array (for scaler).
        Returns None if not enough data.
        """
        if df is None or len(df) < 30:
            return None
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        open_price = df['open'].astype(float)
        volume = df['volume'].astype(float)
        time_col = pd.to_datetime(df['time'])
        
        # 1. EMA fast minus EMA slow
        ema_fast = self.calc.ema(close, self.EMA_FAST)
        ema_slow = self.calc.ema(close, self.EMA_SLOW)
        ema_fast_minus_ema_slow = (ema_fast - ema_slow).iloc[-1]
        
        # 2. Price minus VWAP
        vwap = self.calc.vwap(high, low, close, volume)
        price_minus_vwap = close.iloc[-1] - vwap.iloc[-1]
        
        # 3-5. ADX, +DI, -DI
        adx_val, plus_di, minus_di = self.calc.adx(high, low, close)
        adx_smoothed = adx_val.iloc[-1]
        plus_di_val = plus_di.iloc[-1]
        minus_di_val = minus_di.iloc[-1]
        
        # 6-7. MACD histogram and slope
        _, _, macd_hist = self.calc.macd(close)
        macd_hist_val = macd_hist.iloc[-1]
        macd_hist_slope = self.calc.slope(macd_hist, 5).iloc[-1]
        
        # 8-9. RSI and slope
        rsi = self.calc.rsi(close)
        rsi_val = rsi.iloc[-1]
        rsi_slope = self.calc.slope(rsi, 5).iloc[-1]
        
        # 10-11. ATR and normalized
        atr = self.calc.atr(high, low, close)
        atr_val = atr.iloc[-1]
        atr_normalized = atr_val / close.iloc[-1] if close.iloc[-1] != 0 else 0
        
        # 12. Bollinger Bandwidth
        bb_width = self.calc.bollinger_bandwidth(close)
        bollinger_bandwidth = bb_width.iloc[-1]
        
        # 13. Candle body to range ratio
        candle_range = high.iloc[-1] - low.iloc[-1]
        candle_body = abs(close.iloc[-1] - open_price.iloc[-1])
        candle_body_to_range_ratio = candle_body / candle_range if candle_range > 0 else 0.5
        
        # 14. Cumulative delta slope (approximated using volume direction)
        # Positive if close > open (bullish), negative if close < open
        delta_direction = np.sign(close - open_price)
        cumulative_delta = (delta_direction * volume).cumsum()
        cumulative_delta_slope = self.calc.slope(cumulative_delta, 5).iloc[-1]
        
        # 15. Volume normalized
        vol_sma = self.calc.sma(volume, 20)
        volume_normalized = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1.0
        
        # 16. Session phase (0=pre-market, 1=morning, 2=midday, 3=afternoon, 4=after-hours)
        hour = time_col.iloc[-1].hour
        if hour < 9:
            session_phase = 0  # pre-market
        elif hour < 12:
            session_phase = 1  # morning
        elif hour < 14:
            session_phase = 2  # midday/lunch
        elif hour < 16:
            session_phase = 3  # afternoon
        else:
            session_phase = 4  # after-hours
        
        # 17. Time bucket ID (30-min buckets, 0-47)
        time_bucket_id = (hour * 2 + time_col.iloc[-1].minute // 30) % 48
        
        # 18. Rolling drawdown
        if self.peak_equity > 0:
            rolling_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        else:
            rolling_drawdown = 0.0
        
        # 19. Win/loss ratio
        if len(self.trade_results) > 0:
            wins = sum(self.trade_results)
            total = len(self.trade_results)
            win_loss_ratio = wins / total
        else:
            win_loss_ratio = 0.5  # default neutral
        
        # Build feature array in correct order
        features = np.array([
            ema_fast_minus_ema_slow,
            price_minus_vwap,
            adx_smoothed,
            plus_di_val,
            minus_di_val,
            macd_hist_val,
            macd_hist_slope if not np.isnan(macd_hist_slope) else 0.0,
            rsi_val,
            rsi_slope if not np.isnan(rsi_slope) else 0.0,
            atr_val,
            atr_normalized,
            bollinger_bandwidth,
            candle_body_to_range_ratio,
            cumulative_delta_slope if not np.isnan(cumulative_delta_slope) else 0.0,
            volume_normalized,
            session_phase,
            time_bucket_id,
            rolling_drawdown,
            win_loss_ratio,
        ], dtype=np.float64)
        
        # Replace any remaining NaN/inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features


class LGBMFeatureBuilder:
    """
    Builds feature vector for LGBM model (TAKE/SKIP decision).
    
    LGBM Features (9 total):
    1. close_minus_ema_fast
    2. macd
    3. rsi
    4. stochastic
    5. atr
    6. delta
    7. hmm_prob_0  (from HMM prediction)
    8. hmm_prob_1  (from HMM prediction)
    9. hmm_prob_2  (from HMM prediction)
    """
    
    LGBM_FEATURE_NAMES = [
        'close_minus_ema_fast',
        'macd',
        'rsi',
        'stochastic',
        'atr',
        'delta',
        'hmm_prob_0',
        'hmm_prob_1',
        'hmm_prob_2',
    ]
    
    EMA_FAST = 9
    
    def __init__(self):
        self.calc = IndicatorCalculator()
    
    def build_features(self, df: pd.DataFrame, hmm_probs: np.ndarray) -> Optional[np.ndarray]:
        """
        Build LGBM feature array from bar DataFrame and HMM probabilities.
        
        Args:
            df: DataFrame with OHLCV data
            hmm_probs: Array of 4 probabilities from HMM (we use first 3)
        
        Returns features for the LATEST bar as numpy array (for scaler).
        Returns None if not enough data.
        """
        if df is None or len(df) < 30:
            return None
        
        if hmm_probs is None or len(hmm_probs) < 3:
            return None
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        open_price = df['open'].astype(float)
        volume = df['volume'].astype(float)
        
        # 1. Close minus EMA fast
        ema_fast = self.calc.ema(close, self.EMA_FAST)
        close_minus_ema_fast = close.iloc[-1] - ema_fast.iloc[-1]
        
        # 2. MACD (the line, not histogram)
        macd_line, _, _ = self.calc.macd(close)
        macd_val = macd_line.iloc[-1]
        
        # 3. RSI
        rsi = self.calc.rsi(close)
        rsi_val = rsi.iloc[-1]
        
        # 4. Stochastic
        stoch = self.calc.stochastic(high, low, close)
        stochastic_val = stoch.iloc[-1]
        
        # 5. ATR
        atr = self.calc.atr(high, low, close)
        atr_val = atr.iloc[-1]
        
        # 6. Delta (volume direction * volume for latest bar)
        delta_direction = 1 if close.iloc[-1] >= open_price.iloc[-1] else -1
        delta_val = delta_direction * volume.iloc[-1]
        
        # 7-9. HMM probabilities (first 3 states)
        hmm_prob_0 = hmm_probs[0]
        hmm_prob_1 = hmm_probs[1]
        hmm_prob_2 = hmm_probs[2]
        
        # Build feature array in correct order
        features = np.array([
            close_minus_ema_fast,
            macd_val,
            rsi_val,
            stochastic_val,
            atr_val,
            delta_val,
            hmm_prob_0,
            hmm_prob_1,
            hmm_prob_2,
        ], dtype=np.float64)
        
        # Replace any NaN/inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features