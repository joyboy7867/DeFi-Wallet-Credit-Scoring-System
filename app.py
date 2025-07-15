import json
import pandas as pd
import numpy as np
import pickle 
from function import generate_features
model=pickle.load(open("models/Score_assign_model.pkl","rb"))
scaler=pickle.load(open("models/scaler_model.pkl","rb"))
df=pd.read_json("dataset/user-wallet-transactions.json")
df['timestamp'] = pd.to_datetime(df['timestamp'])
def get_amount(x):
    return x.get("amount")
df["amount"]=df["actionData"].apply(get_amount)
feature_data=generate_features(df)
feature_data['total_borrowed'] = np.log1p(feature_data['total_borrowed'].clip(upper=1e6))
feature_data['total_repaid'] = np.log1p(feature_data['total_repaid'].clip(upper=1e6))
feature_data['total_deposited'] = np.log1p(feature_data['total_deposited'].clip(upper=1e6))
feature_data['repay_ratio'] = feature_data['repay_ratio'].clip(upper=1)
feature_data['borrow_to_deposit_ratio'] = feature_data['borrow_to_deposit_ratio'].clip(upper=5)

feature_data.replace([np.inf, -np.inf], np.nan, inplace=True)
feature_data.fillna(0, inplace=True)



feature_data = feature_data.drop(columns=['userWallet', 'first_txn', 'last_txn'], errors='ignore')
feature_data = feature_data.astype("float64")
scores = model.predict(feature_data)

feature_data['Score'] = scores
feature_data['Score'] = feature_data['Score'].round().astype(int)
feature_data['userWallet'] = df['userWallet']


print(json.dumps(feature_data[['userWallet', 'Score']].to_dict(orient='records'), indent=2))
feature_data.to_json("Tagged.json")