# Kaggle IEEE-CDF Fraud Detection Dataset

## Download

1. Install the Kaggle CLI: `pip install kaggle`
2. Set up your API token: https://www.kaggle.com/docs/api#authentication
3. Download the dataset:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/kaggle/
unzip data/kaggle/ieee-fraud-detection.zip -d data/kaggle/
```

You should end up with these files in this directory:
- `train_transaction.csv` (~590k rows)
- `train_identity.csv` (~144k rows)
- `test_transaction.csv` (optional, no labels)
- `test_identity.csv` (optional, no labels)

The notebook only needs `train_transaction.csv` and `train_identity.csv`.
