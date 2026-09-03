
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 110,
    "axes.titleweight": "bold",
})

print("pandas:", pd.__version__, "| numpy:", np.__version__)

DATA_PATH = "bank_transactions.csv"  # <-- EDIT IF NEEDED

assert os.path.exists(DATA_PATH), f"File not found: {DATA_PATH}"
print(f"Using dataset: {DATA_PATH} ({os.path.getsize(DATA_PATH)/1024**2:.1f} MB)")

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"TransactionAmount (INR)": "TransactionAmount"})

print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
df.head()

df.info()

miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(3)
quality = pd.DataFrame({"missing": miss, "pct": miss_pct})
quality[quality["missing"] > 0]

print(f"Full-row duplicates : {df.duplicated().sum()}")
print(f"Duplicate Transaction IDs : {df['TransactionID'].duplicated().sum()}")
print(f"Unique customers : {df['CustomerID'].nunique():,}")
print(f"Unique cities : {df['CustLocation'].nunique():,}")

missing_cols = miss[miss > 0].sort_values()
plt.figure(figsize=(8, 3.5))
missing_cols.plot(kind="barh", color="#d97e7e")
plt.title("Missing Values per Column (raw data)")
plt.xlabel("Missing rows")
plt.tight_layout()
plt.show()

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="%d/%m/%y", errors="coerce")
print(f"Date range: {df['TransactionDate'].min().date()} → {df['TransactionDate'].max().date()}")
print(f"Span : {(df['TransactionDate'].max() - df['TransactionDate'].min()).days} days")

def parse_dob(s):
    if pd.isna(s):
        return pd.NaT
    d = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    if pd.isna(d):
        return pd.NaT
    # Anything > 2016 (year of data) is wrong → shift back 100 years
    if d.year > 2016:
        d = d.replace(year=d.year - 100)
    return d

df["CustomerDOB"] = df["CustomerDOB"].apply(parse_dob)

df.loc[df["CustomerDOB"].dt.year < 1916, "CustomerDOB"] = pd.NaT
df.loc[df["CustomerDOB"].dt.year > 2010, "CustomerDOB"] = pd.NaT

print("DOB parsed successfully.")
print(f"DOB year range: {df['CustomerDOB'].dt.year.min():.0f} → {df['CustomerDOB'].dt.year.max():.0f}")

df["CustomerAge"] = ((df["TransactionDate"] - df["CustomerDOB"]).dt.days / 365.25).round(1)
print(df["CustomerAge"].describe().round(1))

def to_hour(t):
    if pd.isna(t):
        return np.nan
    s = f"{int(t):06d}"
    h = int(s[:2])
    return h if 0 <= h <= 23 else np.nan

df["TransactionHour"] = df["TransactionTime"].apply(to_hour)

df["CustGender"] = df["CustGender"].replace({"T": np.nan})  # 1 ambiguous record
df["CustLocation"] = df["CustLocation"].str.upper().str.strip()

n_zero = (df["TransactionAmount"] == 0).sum()
df = df[df["TransactionAmount"] > 0].copy()
print(f"Removed {n_zero} zero-amount transactions.")

before = len(df)
df = df.dropna(subset=["TransactionDate", "TransactionAmount", "CustomerID"])
print(f"Removed {before - len(df)} rows missing RFM-critical fields.")

q1, q3 = df["TransactionAmount"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 3 * iqr
n_high = (df["TransactionAmount"] > upper).sum()

df["TransactionAmount_capped"] = np.where(
    df["TransactionAmount"] > upper, upper, df["TransactionAmount"]
)
print(f"Cap upper bound : {upper:,.0f} INR")
print(f"Capped {n_high:,} extreme transactions ({n_high/len(df)*100:.2f}%)")

global_med = df["CustAccountBalance"].median()
df["CustAccountBalance"] = (
    df.groupby("CustLocation")["CustAccountBalance"]
      .transform(lambda x: x.fillna(x.median()))
      .fillna(global_med)
)
print(f"Account balance — remaining nulls: {df['CustAccountBalance'].isnull().sum()}")

df["TransactionWeekday"] = df["TransactionDate"].dt.day_name()
df["TransactionMonth"]   = df["TransactionDate"].dt.month
df["TransactionDay"]     = df["TransactionDate"].dt.day

print(f"\nFinal cleaned shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()

df.to_csv("bank_transactions_clean.csv", index=False)
print("Saved: bank_transactions_clean.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(df["TransactionAmount"].clip(upper=10000), bins=60, color="#5a8dee", edgecolor="white")
axes[0].set_title("Transaction Amount (clipped at 10k INR)")
axes[0].set_xlabel("INR")

axes[1].hist(np.log1p(df["TransactionAmount"]), bins=60, color="#39a48a", edgecolor="white")
axes[1].set_title("log(1 + Transaction Amount)")
axes[1].set_xlabel("log INR")
plt.tight_layout(); plt.show()

plt.figure()
df["CustomerAge"].dropna().clip(lower=15, upper=90).plot(
    kind="hist", bins=40, color="#b97abf", edgecolor="white"
)
plt.title("Customer Age Distribution")
plt.xlabel("Age (years)")
plt.tight_layout(); plt.show()

gender_counts = df["CustGender"].value_counts()
plt.figure(figsize=(6, 4))
plt.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
        colors=["#5a8dee", "#ee9b5a"])
plt.title("Transactions by Gender")
plt.tight_layout(); plt.show()
gender_counts


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
df["CustLocation"].value_counts().head(15).plot(kind="bar", ax=axes[0], color="#5a8dee")
axes[0].set_title("Top 15 cities — Transaction Count")
axes[0].tick_params(axis="x", rotation=45)

df.groupby("CustLocation")["TransactionAmount"].sum().nlargest(15).plot(
    kind="bar", ax=axes[1], color="#39a48a"
)
axes[1].set_title("Top 15 cities — Total INR")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()


daily = df.groupby("TransactionDate").agg(
    txn_count=("TransactionID", "count"),
    total_amount=("TransactionAmount", "sum"),
)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(daily.index, daily["txn_count"], color="#5a8dee")
axes[0].set_ylabel("Transactions / day")
axes[0].set_title("Daily Transaction Volume")
axes[1].plot(daily.index, daily["total_amount"] / 1e6, color="#39a48a")
axes[1].set_ylabel("Total INR (millions)")
axes[1].set_xlabel("Date")
axes[1].set_title("Daily Monetary Value")
plt.tight_layout(); plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

df["TransactionHour"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color="#b97abf"
)
axes[0].set_title("Transactions by Hour")
axes[0].set_xlabel("Hour of day")

weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df["TransactionWeekday"].value_counts().reindex(weekday_order).plot(
    kind="bar", ax=axes[1], color="#5a8dee"
)
axes[1].set_title("Transactions by Weekday")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()


sample = df[["CustAccountBalance", "TransactionAmount"]].sample(20000, random_state=42)
plt.figure(figsize=(8, 5))
plt.scatter(
    sample["CustAccountBalance"].clip(upper=2_000_000),
    sample["TransactionAmount"].clip(upper=20_000),
    alpha=0.25, s=8, color="#5a8dee",
)
plt.title("Account Balance vs Transaction Amount (sampled, clipped)")
plt.xlabel("Account Balance (INR)")
plt.ylabel("Transaction Amount (INR)")
plt.tight_layout(); plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


reference_date = df["TransactionDate"].max() + pd.Timedelta(days=1)

print("Reference date:", reference_date.date())

rfm = df.groupby("CustomerID").agg({
    "TransactionDate": lambda x: (reference_date - x.max()).days,
    "TransactionID": "count",
    "TransactionAmount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("RFM table shape:", rfm.shape)
rfm.head()
rfm.describe().round(2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(rfm["Recency"], bins=30, color="#5a8dee", edgecolor="white")
axes[0].set_title("Recency Distribution")
axes[0].set_xlabel("Days since last transaction")

axes[1].hist(rfm["Frequency"], bins=30, color="#b97abf", edgecolor="white")
axes[1].set_title("Frequency Distribution")
axes[1].set_xlabel("Number of transactions")

axes[2].hist(rfm["Monetary"], bins=30, color="#39a48a", edgecolor="white")
axes[2].set_title("Monetary Distribution")
axes[2].set_xlabel("Total transaction amount")

plt.tight_layout()
plt.show()

# Log transformation for skewed variables
rfm_model = rfm.copy()

rfm_model["Frequency"] = np.log1p(rfm_model["Frequency"])
rfm_model["Monetary"] = np.log1p(rfm_model["Monetary"])

rfm_model.head()


scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_model)

rfm_scaled[:5]

inertia = []

K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(7, 4))

plt.plot(K, inertia, marker="o", color="#5a8dee")

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")

plt.tight_layout()
plt.show()


silhouette_scores = {}

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=100
    )

    labels = kmeans.fit_predict(rfm_scaled)

    score = silhouette_score(
        rfm_scaled,
        labels,
        sample_size=10000,
        random_state=42
    )

    silhouette_scores[k] = score

    print(f"k = {k}, Silhouette Score = {score:.3f}")

plt.figure(figsize=(7, 4))

plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker="o", color="#39a48a")

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")

plt.tight_layout()
plt.show()


optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

rfm.head()


cluster_summary = rfm.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
}).round(2)

cluster_summary

cluster_size = rfm["Cluster"].value_counts().sort_index()

cluster_size

cluster_profile = cluster_summary.copy()
cluster_profile["CustomerCount"] = cluster_size

cluster_profile

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cluster_summary["Recency"].plot(kind="bar", ax=axes[0], color="#5a8dee")
axes[0].set_title("Average Recency by Cluster")
axes[0].set_ylabel("Days")

cluster_summary["Frequency"].plot(kind="bar", ax=axes[1], color="#b97abf")
axes[1].set_title("Average Frequency by Cluster")
axes[1].set_ylabel("Transactions")

cluster_summary["Monetary"].plot(kind="bar", ax=axes[2], color="#39a48a")
axes[2].set_title("Average Monetary Value by Cluster")
axes[2].set_ylabel("INR")

plt.tight_layout()
plt.show()

cluster_profile

segment_map = {
    0: "Inactive Low-Value Customers",
    1: "Regular Customers",
    2: "At-Risk High Spenders",
    3: "VIP High-Value Customers"
}

rfm["Segment"] = rfm["Cluster"].map(segment_map)

rfm.head()

pca = PCA(n_components=2)

pca_features = pca.fit_transform(rfm_scaled)

rfm["PCA1"] = pca_features[:, 0]
rfm["PCA2"] = pca_features[:, 1]

print("Explained variance ratio:", pca.explained_variance_ratio_)

import seaborn as sns

plt.figure(figsize=(12, 8))

sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Segment",
    data=rfm,
    palette="Set2",
    alpha=0.4,
    s=15,
    linewidth=0,
    legend='full'
)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(rfm_scaled)

centroids_scaled = kmeans_final.cluster_centers_
centroids_pca = pca.transform(centroids_scaled)

centroid_data = pd.DataFrame({
    'PCA1': centroids_pca[:, 0],
    'PCA2': centroids_pca[:, 1],
    'Segment': [segment_map[i] for i in range(optimal_k)] # Map cluster ID to Segment name
})

plt.scatter(
    centroid_data['PCA1'],
    centroid_data['PCA2'],
    marker='X',
    s=250,
    color='black',
    edgecolors='white',
    linewidth=1.5,
    zorder=8
)

for i, row in centroid_data.iterrows():
    plt.annotate(
        row['Segment'],
        (row['PCA1'], row['PCA2']),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        va='bottom',
        fontsize=5,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.6", fc='white', alpha=0.7, ec='gray', lw=0.5)
    )


plt.title("Customer Segments Based on RFM Features (PCA Reduced)", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.legend(title="Customer Segment", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) # Adjust legend position
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


rfm.groupby("Segment").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
}).round(2)

rfm.to_csv("customer_segments_rfm_kmeans.csv", index=True)

print("Saved: customer_segments_rfm_kmeans.csv")