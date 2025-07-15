# 🧠 DeFi Wallet Credit Scoring System — Aave V2
📌 Objective
The goal is to build a robust machine learning pipeline that assigns a credit score (0 to 1000) to wallets interacting with the Aave V2 protocol. This score reflects how responsible or risky a wallet's behavior is, based on its historical transaction activity.

## 📂 Input Data
Raw Transaction Logs from Aave V2 protocol.

File format: JSON

Each record includes:

userWallet, action, actionData, timestamp, blockNumber, etc.

## ⚙️ Processing Flow
# 1. Preprocessing
Extract amount from nested actionData.

Convert timestamp to datetime.

Drop irrelevant columns.

# 2. Feature Engineering
Features are engineered on a per-wallet basis:

Number of actions: num_deposits, num_borrows, num_repays, num_liquidations

Transaction volume: total_borrowed, total_repaid, total_deposited

Ratios:

repay_ratio = total_repaid / total_borrowed

borrow_to_deposit_ratio = total_borrowed / total_deposited

active_days = last_txn - first_txn

num_transactions

Post-processing: Capped/normalized extreme values (e.g., repay_ratio <= 1, log-scale for total_*).

# 3. Scoring Model
Model used: RandomForestRegressor

Target score: Engineered using a custom logic:


Score = repay_ratio * 500 + borrow_to_deposit_ratio * 200 - num_liquidations * 50 + total_deposited * 10
Output is clipped between 0 and 1000

Scaled and trained to predict clean, interpretable scores

# 4. Model Export
Trained model and scaler are exported using pickle

Inference script loads the model and generates scores for any wallet dataset

## 🛠 Inference Pipeline
Just run the script:


python app.py
It will:

Load your input JSON file

Preprocess and extract features

Run through the trained model

Print wallet scores in the format:


[{"userWallet": "0xabc...123", "Score": 845}, {...}]
## 📎 Files Included
app.py: Inference script

function.py: Contains generate_features() logic

models/Score_assign_model.pkl: Trained model

models/scaler_model.pkl: Fitted scaler

