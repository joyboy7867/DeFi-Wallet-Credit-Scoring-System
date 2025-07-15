# 📊 Credit Score Analysis Report

## 📌 Score Distribution

The following histogram shows how wallet credit scores are distributed across ranges:



## 📈 Range-wise Breakdown

| score_range   |   count |
|:--------------|--------:|
| 0-99          |     559 |
| 100-199       |       2 |
| 200-299       |       0 |
| 300-399       |       1 |
| 400-499       |       0 |
| 500-599       |       7 |
| 600-699       |       0 |
| 700-799       |       0 |
| 800-899       |       0 |
| 900-999       |       1 |

---

## 🔍 Behavioral Patterns

### 🚨 Low Score Wallets (0–300)

- Tend to have high number of **borrows** but little to no **repays**
- Frequently undergo **liquidation**
- Low or zero **total deposits**
- Often exhibit erratic or bot-like transaction patterns

Example stats:
- Average liquidations: 0.19
- Avg borrow-to-deposit ratio: 0.00

---

### ✅ High Score Wallets (700–1000)


- Regular **depositors** with a healthy borrowing pattern
- High **repay ratio**, indicating responsible debt clearance
- Rare or zero **liquidation** events
- Long wallet activity span and consistent interaction

Example stats:
- Average repay ratio: 0.16
- Average deposits: 954878.39
<img width="1238" height="612" alt="Screenshot 2025-07-15 213133" src="https://github.com/user-attachments/assets/03dfb78e-f794-4db6-836b-9d03b4e093a4" />
---
