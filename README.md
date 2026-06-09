# MLOps Batch Pipeline — T0 Technical Assessment

A minimal, reproducible MLOps-style batch job that computes a rolling-mean
signal from OHLCV data, writes structured metrics, and ships in Docker.

---

## Repository layout

```
.
├── run.py            # Main pipeline script
├── config.yaml       # Experiment config (seed, window, version)
├── data.csv          # 10 000-row OHLCV dataset
├── requirements.txt  # Pinned Python dependencies
├── Dockerfile        # Single-stage Docker image
├── metrics.json      # Sample output from a successful run
├── run.log           # Sample log from a successful run
└── README.md
```

---

## Local run

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

All four flags are required — no paths are hard-coded.

---

## Docker build & run

```bash
# Build
docker build -t mlops-task .

# Run (data.csv + config.yaml are baked into the image)
docker run --rm mlops-task
```

The container prints `metrics.json` to stdout and exits `0` on success or
non-zero on failure.

To persist the outputs to your host machine:

```bash
docker run --rm -v "$(pwd)/output:/app/output" mlops-task \
  python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   output/metrics.json \
  --log-file output/run.log
```

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.5002,
  "latency_ms": 12,
  "seed": 42,
  "status": "success"
}
```

`rows_processed` is 9 996 (not 10 000) because the first `window - 1 = 4`
rows have no complete rolling-mean window and are excluded from signal
computation — this behaviour is documented in `run.py` and consistent across
every run.

### Error output shape

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

`metrics.json` is always written, even on failure.

---

## Config reference (`config.yaml`)

| Key       | Type    | Description                              |
|-----------|---------|------------------------------------------|
| `seed`    | int     | NumPy random seed for reproducibility   |
| `window`  | int     | Rolling-mean window size (rows)          |
| `version` | string  | Pipeline version tag written to output  |

---

## Signal logic

```
rolling_mean[t] = mean(close[t-window+1 … t])   # requires window full rows

signal[t] = 1  if close[t] > rolling_mean[t]
            0  otherwise
            (excluded if rolling_mean[t] is NaN)
```

---

## Reproducibility guarantee

Given identical `data.csv` and `config.yaml`, every run produces the same
`rows_processed`, `value` (signal_rate), `seed`, and `version`.
Only `latency_ms` varies between runs.
