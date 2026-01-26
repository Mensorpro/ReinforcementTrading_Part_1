# Live Data Testing

This folder contains scripts to test the trained model on current market data from MetaTrader 5.

## Files

- `fetch_mt5_data.py` - Fetches live EUR/USD data from MT5
- `test_on_live_data.py` - Tests the trained model on live data
- `README.md` - This file

## Requirements

Make sure you have MetaTrader 5 installed and the Python package:

```bash
pip install MetaTrader5
```

## Usage

### Step 1: Fetch Live Data

Run this first to download current market data:

```bash
python live_testing/fetch_mt5_data.py
```

This will:

- Connect to your MT5 terminal
- Fetch maximum available EUR/USD H1 data (up to 10,000 bars)
- Add technical indicators (same as training)
- Save to `live_testing/live_eurusd_data.csv`

**Requirements:**

- MetaTrader 5 must be running
- You must be logged into an account
- EUR/USD must be available in Market Watch

### Step 2: Test the Model

After fetching data, test your trained model:

```bash
python live_testing/test_on_live_data.py
```

This will:

- Load the live data
- Load your trained model (`model_eurusd_best.zip`)
- Run a full simulation
- Calculate performance metrics
- Generate plots
- Save results

## Output

After testing, you'll get:

1. **Console output** - Performance metrics, trade statistics
2. **Plot** - `live_testing/live_test_results.png`
   - Equity curve
   - Price chart with trade markers
   - Drawdown chart
3. **Text file** - `live_testing/test_results.txt` with detailed results

## Interpreting Results

### Good Performance Indicators:

- Positive total return
- Sharpe ratio > 1.0
- Max drawdown < 10%
- Win rate > 50%
- Profit factor > 1.0

### Warning Signs:

- Negative return (model not working on current data)
- Large drawdown (>20%)
- Very few trades (model too conservative)
- Very many trades (model overtrading)

## Troubleshooting

### "MT5 initialization failed"

- Make sure MetaTrader 5 is installed and running
- Check that you're logged into an account
- Try restarting MT5

### "Failed to select EURUSD"

- Open Market Watch in MT5
- Right-click and select "Show All"
- Find EUR/USD and add it

### "Model file not found"

- Make sure you've trained the model first
- Run `python train_agent.py` to train
- The model should be saved as `model_eurusd_best.zip`

## Next Steps

If the model performs well on live data:

1. Paper trade it for a few weeks
2. Monitor performance on new data
3. Consider live trading with small position sizes

If performance is poor:

1. Check if market conditions changed significantly
2. Consider retraining with more recent data
3. Review the model's hyperparameters
