# LSTM Stock Prediction with Attention Mechanism

**A comprehensive, honest guide to building stock price prediction models with LSTM networks.**

## Overview

This repository contains working code for stock prediction using LSTM with attention mechanisms, achieving **56.58% direction accuracy** on real October 2025 data.

Unlike most tutorials that show fake 90%+ accuracy with broken code, this project demonstrates:
- Real, validated results (56.58% vs random 50%)
- All bugs fixed from the original viral article
- Honest discussion of what works and what doesn't
- Production-ready code you can actually use

## Key Results

```
Direction Accuracy: 56.58%
Training Period: Jan 2023 - Aug 2025
Test Period: October 2025
Features: 22 technical indicators
Architecture: Bidirectional LSTM + Ensemble
Status: Better than random, potentially profitable
```

## Files

### Main Article (NEW)
- **ARTICLE_V2.md** - Complete persuasive Medium article (recommended)
- **ARTICLE.md** - Original technical article with all corrections

### Core Code
- **production_lstm_predictor.py** - Production model (56.58% accuracy)
- **lstm_attention_stock_prediction.py** - Basic corrected model
- **advanced_lstm_stock_prediction.py** - Advanced version

### Testing & Validation
- **test_production_model.py** - Training script for production model
- **validate_predictions.py** - Validation on October 2025 data

### Documentation
- **PRODUCTION_MODEL_RESULTS.md** - Complete results
- **IMPROVEMENT_PLAN.md** - Journey from 50% to 56.58%

## Quick Start

```bash
git clone https://github.com/fenago/lstm-attention-stock-prediction
cd lstm-attention-stock-prediction
pip install tensorflow keras yfinance numpy pandas matplotlib scikit-learn
python test_production_model.py
```

## What Makes This Different?

- **56.58% accuracy** (better than random 50%, realistic vs fake 90%)
- **All bugs fixed** (attention mechanism, scaler, data leakage)
- **22 technical indicators** (RSI, MACD, Bollinger Bands, etc.)
- **Honest validation** (same-regime testing, no cherry-picking)

## Performance Benchmarks

| Accuracy | Reality |
|---------|---------|
| 50-52% | Random guessing |
| **56-60%** | **Potentially profitable** ← WE ARE HERE |
| 61-65% | Professional level |
| 70%+ | Likely overfitting |

## Disclaimer

⚠️ Educational purposes only. Do NOT use for actual trading without proper risk management.

## Author

**Dr. Ernesto Lee** | [drlee.io](https://drlee.io)

---

*56% accuracy is genuinely good for stock prediction. Don't trust anyone claiming 90%+ without seeing their code and validation.*
