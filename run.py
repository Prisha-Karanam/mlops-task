"""
MLOps Batch Pipeline — rolling-mean signal generator.

Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps rolling-mean signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input OHLCV CSV")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler — full DEBUG output
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    # Type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"Config 'version' must be a non-empty string, got: {cfg['version']!r}")

    logger.info("Config loaded — seed=%d  window=%d  version=%s",
                cfg["seed"], cfg["window"], cfg["version"])
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("CSV parsed successfully but contains no rows")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Available columns: {list(df.columns)}"
        )

    # Coerce close to numeric; non-numeric rows become NaN
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    n_bad = df["close"].isna().sum()
    if n_bad > 0:
        logger.warning("Dropped %d row(s) with non-numeric 'close' values", n_bad)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid numeric 'close' values remain after cleaning")

    logger.info("Dataset loaded — %d rows, columns: %s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Pipeline: rolling mean + signal
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    """
    Compute rolling mean on 'close' with the given window.

    The first (window - 1) rows produce NaN because there are insufficient
    preceding observations. These rows are *excluded* from signal computation
    downstream (they are not counted in rows_processed).
    """
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    logger.debug("Rolling mean computed — window=%d, NaN prefix rows=%d", window, window - 1)
    return rolling_mean


def compute_signal(df: pd.DataFrame, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    Binary signal: 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN are excluded (signal = NaN, filtered later).
    """
    signal = pd.Series(np.nan, index=df.index)
    valid_mask = rolling_mean.notna()
    signal[valid_mask] = (df.loc[valid_mask, "close"] > rolling_mean[valid_mask]).astype(int)
    n_valid = int(valid_mask.sum())
    logger.debug("Signal computed — valid rows=%d, NaN rows excluded=%d",
                 n_valid, len(df) - n_valid)
    return signal


# ---------------------------------------------------------------------------
# Metrics writing
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict, logger: logging.Logger) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Metrics written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("Job start")
    logger.info("  input   : %s", args.input)
    logger.info("  config  : %s", args.config)
    logger.info("  output  : %s", args.output)
    logger.info("  log-file: %s", args.log_file)
    logger.info("=" * 60)

    job_start = time.perf_counter()
    version = "unknown"

    try:
        # ── 1. Config ──────────────────────────────────────────────────────
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        seed: int    = cfg["seed"]
        window: int  = cfg["window"]

        # ── 2. Seed ────────────────────────────────────────────────────────
        np.random.seed(seed)
        logger.info("NumPy random seed set to %d", seed)

        # ── 3. Dataset ─────────────────────────────────────────────────────
        df = load_dataset(args.input, logger)

        # ── 4. Rolling mean ────────────────────────────────────────────────
        logger.info("Computing rolling mean (window=%d) …", window)
        rolling_mean = compute_rolling_mean(df, window, logger)

        # ── 5. Signal ──────────────────────────────────────────────────────
        logger.info("Generating binary signal …")
        signal = compute_signal(df, rolling_mean, logger)

        # ── 6. Metrics ─────────────────────────────────────────────────────
        valid_signal = signal.dropna()
        rows_processed = int(len(valid_signal))
        signal_rate = round(float(valid_signal.mean()), 4)

        latency_ms = round((time.perf_counter() - job_start) * 1000)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          signal_rate,
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        logger.info("Metrics summary — rows_processed=%d  signal_rate=%.4f  latency_ms=%d",
                    rows_processed, signal_rate, latency_ms)

        write_metrics(args.output, metrics, logger)

        logger.info("Job completed successfully")
        logger.info("=" * 60)

        # Print final JSON to stdout (Docker visibility)
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:  # noqa: BLE001
        latency_ms = round((time.perf_counter() - job_start) * 1000)
        logger.error("Pipeline failed: %s", exc, exc_info=True)

        error_payload = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
        }
        try:
            write_metrics(args.output, error_payload, logger)
        except Exception as write_exc:
            logger.error("Additionally failed to write error metrics: %s", write_exc)

        logger.info("Job ended with errors — latency_ms=%d", latency_ms)
        logger.info("=" * 60)

        print(json.dumps(error_payload, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
