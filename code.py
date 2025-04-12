import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Load the Dataset
df = pd.read_csv('dataset.csv')
df.info()
df.head()
# Step 2: Data Cleaning
# Select relevant columns for the objectives
relevant_columns = [
    'ocid', 'tender/mainProcurementCategory', 'buyer/name', 'tender/value/amount',
    'tender/status', 'tender/stage', 'tender/numberOfTenderers', 'tender/datePublished'
]
df = df[relevant_columns]

# Handle missing values
df = df.dropna(subset=['tender/mainProcurementCategory', 'buyer/name'])  # Drop rows missing critical columns
df['tender/value/amount'] = df['tender/value/amount'].replace('', np.nan)  # Replace empty strings
df['tender/numberOfTenderers'] = df['tender/numberOfTenderers'].fillna(0).astype(int)  # Fill missing tenderers with 0

# Clean tender/value/amount (remove commas and convert to numeric)
df['tender/value/amount'] = df['tender/value/amount'].astype(str).str.replace(',', '')
df['tender/value/amount'] = pd.to_numeric(df['tender/value/amount'], errors='coerce')
df = df.dropna(subset=['tender/value/amount'])  # Drop rows with invalid values

# Standardize tender/mainProcurementCategory
category_mapping = {
    'Works': 'Works',
    'Goods': 'Goods',
    'Services': 'Services'
}
df['tender/mainProcurementCategory'] = df['tender/mainProcurementCategory'].map(category_mapping).fillna(df['tender/mainProcurementCategory'])

# Standardize tender/status and drop tender/stage
status_mapping = {
    'AOC': 'Awarded',
    'Cancelled': 'Cancelled',
    'To be Opened': 'Pending',
    'Financial Bid Opening': 'In Progress',
    'Technical Bid Opening': 'In Progress',
    'Technical Evaluation': 'In Progress',
    'Financial Evaluation': 'In Progress',
    'Retender': 'Retendered'
}
df['tender/status'] = df['tender/status'].map(status_mapping).fillna('Unknown')
df = df.drop(columns=['tender/stage'], errors='ignore')

# Clean date column (tender/datePublished)
df['tender/datePublished'] = pd.to_datetime(df['tender/datePublished'], errors='coerce', format='%d-%m-%y %H:%M')
df = df.dropna(subset=['tender/datePublished'])

# Create time-based features for regression
df['year'] = df['tender/datePublished'].dt.year
df['month'] = df['tender/datePublished'].dt.to_period('M')
df['month_index'] = (df['tender/datePublished'].dt.year - df['tender/datePublished'].dt.year.min()) * 12 + df['tender/datePublished'].dt.month - df['tender/datePublished'].dt.month.min()

# Drop duplicates
df = df.drop_duplicates(subset=['ocid'])

# Save cleaned dataset (optional)
df.to_csv('cleaned_dataset.csv', index=False)

# Step 3: Prepare Data for Linear Regression
# Aggregate tender counts by month for Objective 5
monthly_tenders = df.groupby('month_index').size().reset_index(name='tender_count')

# Linear Regression for Objective 5: Tender Activity Over Time
X_time = monthly_tenders[['month_index']].values
y_time = monthly_tenders['tender_count'].values
lr_time = LinearRegression()
lr_time.fit(X_time, y_time)
y_time_pred = lr_time.predict(X_time)
r2_time = r2_score(y_time, y_time_pred)
print(f"R² for Tender Activity Over Time: {r2_time:.2f}")

# Step 4: Create Dashboard Visualizations
sns.set(style='whitegrid')
plt.figure(figsize=(20, 15))

# Objective 1: Tender Distribution by Procurement Category
plt.subplot(2, 3, 1)
category_counts = df['tender/mainProcurementCategory'].value_counts()
sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
plt.title('Tender Distribution by Procurement Category')
plt.xlabel('Number of Tenders')
plt.ylabel('Category')
plt.show()

# Objective 2: Top Procuring Departments
plt.subplot(2, 3, 2)
top_buyers = df['buyer/name'].value_counts().head(10)
sns.barplot(x=top_buyers.values, y=top_buyers.index, palette='magma')
plt.title('Top 10 Procuring Departments')
plt.xlabel('Number of Tenders')
plt.ylabel('Department')
plt.show()

# Objective 3: Tender Value Distribution
plt.subplot(2, 3, 3)
df['log_value'] = np.log1p(df['tender/value/amount'])
sns.histplot(df['log_value'], bins=30, kde=True, color='teal')
plt.title('Tender Value Distribution (Log Scale)')
plt.xlabel('Log(Tender Value)')
plt.ylabel('Frequency')
plt.show()


# Objective 4: Tender Activity Over Time with Regression
plt.subplot(2, 3, 4)
plt.scatter(monthly_tenders['month_index'], monthly_tenders['tender_count'], color='purple', label='Actual')
plt.plot(monthly_tenders['month_index'], y_time_pred, color='red', linewidth=2, label='Linear Regression')
plt.title(f'Tender Activity Over Time (R² = {r2_time:.2f})')
plt.xlabel('Months Since Start')
plt.ylabel('Number of Tenders')
plt.show()

# Save the dashboard
plt.savefig('tender_dashboard.png')
