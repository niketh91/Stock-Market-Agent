import os
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from dotenv import load_dotenv
load_dotenv()  # loads .env if present, silently skips if not found

# ── Connection config from environment variables ─────────────────
WAREHOUSE_ID      = os.getenv("DATABRICKS_WAREHOUSE_ID")
GOLD_TABLE        = os.getenv("GOLD_TABLE",        "tabular.dataexpert.smc_stocks_gold")
GOLD_PAIR_SUMMARY = os.getenv("GOLD_PAIR_SUMMARY", "tabular.dataexpert.smc_stocks_gold_pair_summary")
SILVER_PAIRS      = os.getenv("SILVER_PAIRS",      "tabular.dataexpert.smc_stocks_silver_pairs")

# ── SDK client — authenticates automatically inside Databricks Apps
w = WorkspaceClient()

# ── Helper: execute SQL and return pandas DataFrame ──────────────
def run_query(sql: str) -> pd.DataFrame:
    """
    Executes a SQL statement against the warehouse and returns
    the result as a pandas DataFrame.
    """
    response = w.statement_execution.execute_statement(
        warehouse_id = WAREHOUSE_ID,
        statement    = sql,
        wait_timeout = "30s",
    )

    if response.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Query failed: {response.status.error}")

    # Handle empty result
    if (not response.result or 
        not response.result.data_array or 
        len(response.result.data_array) == 0):
        cols = []
        if response.manifest and response.manifest.schema:
            cols = [c.name for c in response.manifest.schema.columns]
        return pd.DataFrame(columns=cols)

    # Extract column names
    cols = [c.name for c in response.manifest.schema.columns]

    # data_array is a list of lists in newer SDK versions
    rows = []
    for r in response.result.data_array:
        if hasattr(r, 'values'):
            rows.append(r.values)  # older SDK
        else:
            rows.append(list(r))   # newer SDK — r is already a list

    return pd.DataFrame(rows, columns=cols)

# ── Helper: get latest trade date ───────────────────────────────
def get_latest_date() -> str:
    df = run_query(f"SELECT MAX(trade_date) as latest FROM {GOLD_TABLE}")
    return str(df["latest"].iloc[0])

# ══════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def get_latest_signals(trade_date: str) -> pd.DataFrame:
    """
    Returns a snapshot of all tickers for the given trade date.
    Includes RSI, signal regime, volatility rank, volume ratio,
    and best correlated partner for each ticker.
    """
    return run_query(f"""
        SELECT
            ticker,
            trade_date,
            day_close,
            day_return_pct,
            rsi,
            signal_regime,
            volatility_score,
            volatility_rank,
            volume_vs_average_ratio,
            strong_corr_partner_count,
            best_intraday_corr_partner,
            best_intraday_corr_score
        FROM {GOLD_TABLE}
        WHERE trade_date = '{trade_date}'
        ORDER BY ticker
    """)

def get_regime_changes(trade_date: str) -> pd.DataFrame:
    """
    Returns tickers where the signal regime changed on the given
    trade date compared to the previous day.
    """
    return run_query(f"""
        SELECT
            ticker,
            trade_date,
            prev_regime,
            signal_regime,
            prev_rsi,
            rsi,
            day_return_pct,
            volume_vs_average_ratio
        FROM {GOLD_TABLE}
        WHERE trade_date    = '{trade_date}'
          AND regime_changed = true
          AND prev_regime   IS NOT NULL
        ORDER BY ticker
    """)

def get_high_volatility_tickers(trade_date: str,
                                 threshold: float = 0.75) -> pd.DataFrame:
    """
    Returns tickers in the top percentile of volatility on the
    given trade date.
    """
    return run_query(f"""
        SELECT
            ticker,
            trade_date,
            volatility_score,
            volatility_rank,
            rsi,
            signal_regime,
            day_return_pct
        FROM {GOLD_TABLE}
        WHERE trade_date     = '{trade_date}'
          AND volatility_rank >= {threshold}
        ORDER BY volatility_rank DESC
    """)

def get_correlated_pairs(ticker: str,
                          min_consistency_pct: float = 30.0) -> pd.DataFrame:
    """
    Returns structural correlation partners for a specific ticker.
    Checks both sides of the pair since ticker can be ticker_a or ticker_b.
    """
    return run_query(f"""
        SELECT
            CASE
                WHEN ticker_a = '{ticker}' THEN ticker_b
                ELSE ticker_a
            END AS partner,
            avg_intraday_corr,
            max_intraday_corr,
            strong_corr_days,
            total_days_observed,
            corr_consistency_pct
        FROM {GOLD_PAIR_SUMMARY}
        WHERE (ticker_a = '{ticker}' OR ticker_b = '{ticker}')
          AND corr_consistency_pct >= {min_consistency_pct}
        ORDER BY avg_intraday_corr DESC
    """)

def get_divergence_alert(trade_date: str,
                          min_consistency_pct: float = 30.0,
                          divergence_threshold: float = 0.2) -> pd.DataFrame:
    """
    Finds pairs that normally move together but diverged today.
    Joins today's silver pair correlation against the historical
    gold pair summary to detect relationship breakdowns.
    """
    return run_query(f"""
        SELECT
            s.ticker_a,
            s.ticker_b,
            g.avg_intraday_corr  AS historical_avg_corr,
            s.intraday_corr      AS today_corr,
            ROUND(g.avg_intraday_corr - s.intraday_corr, 4) AS corr_drop,
            g.corr_consistency_pct
        FROM {SILVER_PAIRS} s
        JOIN {GOLD_PAIR_SUMMARY} g
          ON s.ticker_a = g.ticker_a
         AND s.ticker_b = g.ticker_b
        WHERE s.trade_date            = '{trade_date}'
          AND g.corr_consistency_pct  >= {min_consistency_pct}
          AND (g.avg_intraday_corr - s.intraday_corr) >= {divergence_threshold}
        ORDER BY corr_drop DESC
    """)