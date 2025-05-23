import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Clean column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Convert date column
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Rename for easier access
df.rename(columns={
    'estimated_unemployment_rate_(%)': 'unemployment_rate',
    'estimated_employed': 'employed',
    'estimated_labour_participation_rate_(%)': 'labour_participation_rate',
    'region.1': 'zone'
}, inplace=True)

# === Terminal Output ===

print("📊 BASIC STATISTICS:")
print(df[['unemployment_rate', 'employed', 'labour_participation_rate']].describe())
print("\n📈 Date Range:", df['date'].min().date(), "to", df['date'].max().date())
print("🧭 Total Unique States:", df['region'].nunique())
print("🗺️ Zones:", df['zone'].unique())

# Average unemployment by month (national)
monthly_avg = df.groupby(df['date'].dt.strftime('%B'))['unemployment_rate'].mean()
print("\n📅 Average Unemployment Rate by Month:")
print(monthly_avg)

# Most and least affected state (by max unemployment)
max_state = df.loc[df['unemployment_rate'].idxmax()]
min_state = df.loc[df['unemployment_rate'].idxmin()]
print(f"\n🚨 Highest Unemployment:\n{max_state['region']} - {max_state['unemployment_rate']}% on {max_state['date'].date()}")
print(f"✅ Lowest Unemployment:\n{min_state['region']} - {min_state['unemployment_rate']}% on {min_state['date'].date()}")

# === Visuals ===

# National average over time
plt.figure(figsize=(12, 6))
national_avg = df.groupby('date')['unemployment_rate'].mean()
sns.lineplot(x=national_avg.index, y=national_avg.values)
plt.title("National Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Region-wise plot
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='date', y='unemployment_rate', hue='region', legend=False)
plt.title("Unemployment Rate by Region Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap by zone
df['month'] = df['date'].dt.strftime('%b')
heatmap_data = df.pivot_table(values='unemployment_rate', index='zone', columns='month', aggfunc='mean')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".1f")
plt.title("Monthly Average Unemployment Rate by Zone")
plt.xlabel("Month")
plt.ylabel("Zone")
plt.tight_layout()
plt.show()
