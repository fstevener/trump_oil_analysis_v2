import pandas as pd
import numpy as np


def prepare_wti_ohlc(df):
    df = df.copy()
    
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").set_index("timestamp_utc")
    
    df["log_close"] = np.log(df["close"])
    
    # core returns
    df["ret_1"] = df["log_close"].diff()
    
    # OHLC range
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_return"] = np.log(df["close"]) - np.log(df["open"])
    
    return df


def add_returns(df):
    """
    Multi-lag Returns & Momentum
    """
    lags = [1, 5, 15, 30, 60, 240]
    
    for l in lags:
        df[f"ret_{l}"] = df["log_close"].diff(l)
    
    # momentum (rolling mean returns)
    for w in [5, 15, 60]:
        df[f"mom_{w}"] = df["ret_1"].rolling(w).mean()
    
    return df


def parkinson_vol(df, window):
    """
    Parkinson Volatility
    """
    hl = np.log(df["high"] / df["low"])
    return (hl ** 2).rolling(window).mean() / (4 * np.log(2))


def garman_klass_vol(df):
    """
    Garman-Klass Volatility
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    
    return (
        0.5 * (log_hl ** 2)
        - (2 * np.log(2) - 1) * (log_co ** 2)
    )


def add_volatility(df):
    """
    computes Parkinson Vol and Garman-Klaas Vol
    """
    windows = [5, 15, 30, 60, 240]
    
    for w in windows:
        df[f"vol_close_{w}"] = df["ret_1"].rolling(w).std()
        df[f"vol_parkinson_{w}"] = parkinson_vol(df, w)
    
    df["vol_gk"] = garman_klass_vol(df)
    
    # regime signal
    df["vol_ratio_5_60"] = df["vol_close_5"] / (df["vol_close_60"] + 1e-8)
    
    return df


def add_trend_features(df):
    """ Trend Features (OHLC-aware)"""
    for w in [5, 15, 60, 240]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        
        df[f"dist_ema_{w}"] = (df["close"] - df[f"ema_{w}"]) / df[f"ema_{w}"]
    
    # range-based momentum
    df["hl_momentum"] = df["high"].rolling(10).max() - df["low"].rolling(10).min()
    
    return df


# def add_microstructure(df):
#     # candle body size
#     df["body"] = (df["close"] - df["open"]) / df["close"]
    
#     # wick structure
#     df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
#     df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
    
#     # volatility clustering
#     df["abs_ret"] = df["ret_1"].abs()
#     df["vol_cluster"] = df["abs_ret"].shift(1).rolling(10).mean()
    
#     # jump proxy
#     df["vol_5"] = df["ret_1"].shift(1).rolling(5).std()
#     # df["jump"] = (df["ret_1"].abs() > 3 * df["vol_5"].shift(1)).astype(int)
#     ret = df["ret_1"].shift(1)
#     vol = df["vol_5"].shift(1)

#     df["jump"] = (ret.abs() > 3 * vol).astype(int)
    
#     return df


def add_microstructure(df):
    # candle body size (OK)
    df["body"] = (df["close"] - df["open"]) / df["close"]
    
    # wick structure (OK ohne shift)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
    
    # volatility clustering (clean causal)
    df["abs_ret"] = df["ret_1"].abs()
    df["vol_cluster"] = df["abs_ret"].rolling(10).mean()
    
    # jump proxy (clean version)
    df["vol_5"] = df["ret_1"].rolling(5).std()
    
    df["jump"] = (
        df["ret_1"].abs() >
        3 * df["vol_5"]
    ).astype(int)
    
    return df


def add_time_features(df):
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    # crude oil trading session proxy
    df["us_session"] = ((df["hour"] >= 13) & (df["hour"] <= 21)).astype(int)
    
    return df


# def add_regime_features(df):
#     df["vol_regime"] = pd.qcut(df["vol_close_60"], 3, labels=False)
    
#     df["trend_60"] = df["close"].diff(60)
#     df["trend_regime"] = (df["trend_60"] > 0).astype(int)
    
#     return df


def add_regime_features(df):
    # rolling volatility regime (no future info)
    # df["vol_rank"] = df["vol_close_60"].shift(1).rolling(500).rank(pct=True)
    df["vol_rank"] = (
        df["vol_close_60"]
        .shift(1)
        .rolling(500)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    df["vol_regime"] = (df["vol_rank"] > 0.66).astype(int)

    df["trend_60"] = df["close"].shift(1).diff(60)
    df["trend_regime"] = (df["trend_60"] > 0).astype(int)

    return df


def build_wti_ohlc_features(df):
    df = prepare_wti_ohlc(df)
    df = add_returns(df)
    df = add_volatility(df)
    df = add_trend_features(df)
    df = add_microstructure(df)
    df = add_time_features(df)
    df = add_regime_features(df)
    
    return df


def create_labels(df, horizons=[2,5,15,30,60,240,720,1440]):
    
    for h in horizons:
        future_ret = np.log(df["close"].shift(-h)) - np.log(df["close"])
        
        # volatility adjusted threshold
        vol = df["ret_1"].rolling(1440).std()
        thr = 0.5 * vol
        
        df[f"y_up_{h}"] = (future_ret > thr).astype(int)
        df[f"y_down_{h}"] = (future_ret < -thr).astype(int)
    
    return df