import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
apps = pd.read_csv("apps.csv")
reviews = pd.read_csv("user_reviews.csv")

# Quick check
print(apps.head())
print(reviews.head())

# Merge datasets on 'App' (inner join)
df = pd.merge(apps, reviews, on="App", how="inner")

# Clean 'Installs' column (remove + , and convert to int)
df['Installs'] = df['Installs'].replace('[+,]', '', regex=True).astype(int)

# Clean 'Price' column (remove $ sign and convert to float)
df['Price'] = df['Price'].replace('[$]', '', regex=True).astype(float)

# Convert 'Size' column
def size_to_float(size):
    if isinstance(size, str):
        if 'M' in size:
            return float(size.replace('M',''))
        elif 'k' in size:
            return float(size.replace('k',''))/1024
        elif size == 'Varies with device':
            return None
    return size

df['Size'] = df['Size'].apply(size_to_float)

# ----------------------------
# 1. App Ratings Analysis
# ----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Rating'].dropna(), bins=20, kde=True, color='skyblue')
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 2. App Size vs Ratings
# ----------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x="Size", y="Rating", data=df, alpha=0.5)
plt.title("App Size vs Rating")
plt.xlabel("Size (MB)")
plt.ylabel("Rating")
plt.show()

# ----------------------------
# 3. Popularity (by Installs per Category)
# ----------------------------
plt.figure(figsize=(10,6))
top_installed = df.groupby('Category')['Installs'].sum().sort_values(ascending=False)[:10]
sns.barplot(x=top_installed.values, y=top_installed.index, palette="viridis")
plt.title("Top 10 Categories by Total Installs")
plt.xlabel("Total Installs")
plt.ylabel("Category")
plt.show()

# ----------------------------
# 4. Pricing Trends (Free vs Paid)
# ----------------------------
plt.figure(figsize=(6,5))
sns.countplot(x="Type", data=df, palette="pastel")
plt.title("Free vs Paid Apps Count")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="Type", y="Rating", data=df, palette="muted")
plt.title("App Ratings: Free vs Paid")
plt.show()

# Price distribution (Paid apps only)
paid_apps = df[df['Type'] == 'Paid']
plt.figure(figsize=(8,5))
sns.histplot(paid_apps['Price'], bins=30, color='coral')
plt.title("Price Distribution of Paid Apps")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 5. Sentiment Analysis (from reviews dataset)
# ----------------------------
plt.figure(figsize=(6,5))
sns.countplot(x="Sentiment", data=df, palette="Set2")
plt.title("User Review Sentiments")
plt.show()

# Average sentiment polarity by app type
plt.figure(figsize=(8,5))
sns.boxplot(x="Type", y="Sentiment_Polarity", data=df, palette="coolwarm")
plt.title("Sentiment Polarity: Free vs Paid Apps")
plt.show()