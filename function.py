import pandas as pd
def generate_features(df):
    grouped = df.groupby('userWallet')

    features = pd.DataFrame()
    features['num_transactions'] = grouped.size()
    features['num_deposits'] = grouped.apply(lambda x: (x['action'] == 'deposit').sum())
    features['num_borrows'] = grouped.apply(lambda x: (x['action'] == 'borrow').sum())
    features['num_repays'] = grouped.apply(lambda x: (x['action'] == 'repay').sum())
    features['num_liquidations'] = grouped.apply(lambda x: (x['action'] == 'liquidationcall').sum())

    features['total_borrowed'] = grouped.apply(lambda x: x[x['action'] == 'borrow']['amount'].sum())
    features['total_repaid'] = grouped.apply(lambda x: x[x['action'] == 'repay']['amount'].sum())
    features['total_deposited'] = grouped.apply(lambda x: x[x['action'] == 'deposit']['amount'].sum())
    features['total_repaid'] = features['total_repaid'].astype(float)
    features['total_borrowed'] = features['total_borrowed'].astype(float)
    features['total_deposited'] = features['total_deposited'].astype(float)

    features['repay_ratio'] = features['total_repaid'] / features['total_borrowed'].replace(0, 1)
    features['borrow_to_deposit_ratio'] = features['total_borrowed'] / features['total_deposited'].replace(0, 1)

    # Active time span
    features['first_txn'] = grouped['timestamp'].min()
    features['last_txn'] = grouped['timestamp'].max()
    features['active_days'] = (features['last_txn'] - features['first_txn']).dt.days + 1

    return features.reset_index()