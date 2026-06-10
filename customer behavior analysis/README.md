# 🛍️ Customer Behavior Analysis & Spending Prediction

A complete data science project analyzing customer purchasing behavior across Istanbul shopping malls, with machine learning to predict high-value customers.

## Dataset

**Customer Shopping Dataset** from Kaggle — 99,457 transactions from 94,187 unique customers across 10 Istanbul malls (January 2021 – March 2023).

| Feature | Description |
|---------|-------------|
| `invoice_no` | Unique transaction ID |
| `customer_id` | Unique customer ID |
| `gender` | Male / Female |
| `age` | Customer age (18–69) |
| `category` | 8 product categories |
| `quantity` | Items purchased |
| `price` | Unit price (USD) |
| `payment_method` | Cash / Credit Card / Debit Card |
| `invoice_date` | Transaction date |
| `shopping_mall` | One of 10 Istanbul malls |

## Project Structure

```
customer-behavior-analysis/
│
├── data/
│   └── customer_shopping_data.csv
│
├── notebooks/
│   └── customer_behavior_analysis.ipynb
│
├── images/
│   ├── 01_sales_by_category.png
│   ├── 02_monthly_revenue_trend.png
│   ├── 03_spending_distribution.png
│   ├── 04_age_vs_spending.png
│   ├── 05_payment_method_analysis.png
│   ├── 06_gender_category_heatmap.png
│   ├── 07_top_malls.png
│   ├── 08_confusion_matrices.png
│   ├── 09_feature_importance.png
│   └── 10_roc_curves.png
│
├── README.md
└── requirements.txt
```

## Analysis Parts

### Part 1 – Data Cleaning
- Date parsing, type conversion
- Missing value check, duplicate removal
- Feature engineering: `total_spending`, `age_group`, time features

### Part 2 – Exploratory Data Analysis
- Revenue by category, gender, mall, payment method
- Monthly trend analysis
- Customer spending distribution
- Age group comparison

### Part 3 – Visualizations (10 charts)
Professional charts covering all major dimensions of customer behavior.

### Part 4 – Machine Learning
**Goal:** Classify customers as High-Value (top 30% spenders) vs Low-Value

**Models compared:**

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 77.2% | 0.813 |
| **Random Forest** | **83.4%** | **0.914** |

**Key finding:** Random Forest significantly outperforms Logistic Regression, with `num_transactions`, `unique_categories`, and `avg_quantity` being the strongest predictors of high-value customers.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/customer_behavior_analysis.ipynb
```

## Key Insights

- **Clothing** is the top revenue category, followed by Cosmetics and Technology
- Revenue peaks in **Q4** — holiday shopping drives a significant spike
- **Credit Card** is the most popular payment method (40% of transactions)
- Customers who shop across multiple categories and make frequent visits are most likely to be high-value
- Random Forest achieves **91.4% AUC** in identifying high-value customers from behavioral features alone

---
*Built with Python · pandas · scikit-learn · matplotlib · seaborn*
