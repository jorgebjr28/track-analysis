# -*- coding: utf-8 -*-
"""
Integrated Track & Field Workflow
Person 2: Athlete Analysis Lead
• Conduct correlation analysis between events
• Perform athlete clustering
• Analyze event affinity patterns
• Analyze performance trends, grade-level, top performers, performance↔points
• Avoid duplicate athlete×event errors by aggregating
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# === 1. Load dataset ===

#csv_path = r"C:\Users\rickb\Downloads\DummyDataFinal.csv"  # adjust as needed
csv_path = r"C:\Users\firer\Downloads\Temp Proj\AllMeetsExtracted.csv"  # for Windows compatibility
#csv_path = r"C:\Users\firer\Downloads\Temp Proj\text\Extracted\PortIsabel_25.csv"
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Add after initial data loading
relay_events = ['4x100M', '4x200M', '4x400M', '4x800M']

def is_relay_event(event_id):
    return any(relay in event_id for relay in relay_events)

# Add after loading data, before preprocessing
print("\nChecking for duplicates...")
total_entries = len(df)

# Check duplicates by athlete and event within same meet
duplicates = df.groupby(['meet_id', 'athlete_id', 'event_id']).size().reset_index(name='count')
duplicate_entries = duplicates[duplicates['count'] > 1]

if len(duplicate_entries) > 0:
    print(f"Found {len(duplicate_entries)} duplicate entries")
    # Keep best performance per athlete per event per meet
    df = df.sort_values('performance').groupby(
        ['meet_id', 'athlete_id', 'event_id']
    ).first().reset_index()
    print(f"Removed {total_entries - len(df)} duplicate entries")
else:
    print("No duplicate entries found")

# === 2. Preprocessing ===
# 2.1 Null handling
null_counts = df.isnull().sum()
if df.isnull().values.any():
    print("\nNull values found in:")
    print(null_counts[null_counts > 0])
    print("\nHandling null values:")
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"- Filled {col} nulls with median: {median}")
    
    # Handle categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            print(f"- Filled {col} nulls with mode: {mode}")

else:
    print("\nNo null values found in dataset")

# 2.2 Per‐event normalization
df['performance_normalized'] = (
    df
    .groupby('event_id')['performance']
    .transform(lambda x: (x - x.mean()) / x.std())
)

# Add after z-score calculation
def flag_extreme_scores(data, z_col='performance_normalized', threshold=4):
    extremes = data[abs(data[z_col]) > threshold]
    if len(extremes) > 0:
        print(f"\nFound {len(extremes)} extreme performances (|z| > {threshold}):")
        print(extremes[['athlete_id', 'event_id', 'performance', z_col]])
        
        # Visualize outliers by event
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x='event_id', y=z_col)
        plt.scatter(x=pd.Categorical(extremes['event_id']).codes, 
                   y=extremes[z_col], 
                   color='red', 
                   alpha=0.6)
        plt.xticks(rotation=45)
        plt.title('Performance Distribution with Extreme Values Highlighted')
        plt.tight_layout()
        plt.show()
    return extremes

# === 3. Exploratory Visualizations & Metrics ===
sns.set_style('whitegrid')

# Update the overall Z-score distribution (around line 109)
from scipy import stats

# 3.1 Overall Z‐score distribution with proper density axis
plt.figure(figsize=(10,4))
ax = sns.violinplot(data=df, y='performance_normalized', cut=0)

# Create the KDE to get actual density values
kde = stats.gaussian_kde(df['performance_normalized'].dropna())
y_eval = np.linspace(-3, 3, 1000)
x_density = kde(y_eval)
max_density = x_density.max()

# Calculate positions for density ticks
x_center = ax.collections[0].get_offsets()[0][0]
violin_width = ax.collections[0].get_paths()[0].get_extents().width
half_width = violin_width / 2

# Create uniform density tick marks with actual values
density_values = [0, max_density/4, max_density/2, 3*max_density/4, max_density]
density_positions = [x_center - half_width * 0.95, 
                    x_center - half_width * 0.5,
                    x_center,
                    x_center + half_width * 0.5,
                    x_center + half_width * 0.95]
density_labels = [f'{d:.2f}' for d in density_values]

plt.xticks(density_positions, density_labels)
plt.xlabel('Probability Density')
plt.ylabel('Performance (Standard Deviations)')

# Add more comprehensive explanatory note about density
plt.figtext(0.5, -0.08, 
           "Note: The width of the violin shows relative frequency of observations.\n"
           "Wider sections have more observations at that performance level.\n"
           "The x-axis values (0.00 to 0.48) represent probability density - higher values indicate\n"
           "more concentrated observations. The middle (0.24) shows peak concentration.",
           ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.9))

plt.title(f'Overall Distribution of Performance Z-Scores (n={len(df)})')
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make more room for the expanded note
plt.show()

# 3.2 Distribution by event - UPDATE THIS PART TOO
for ev in df['event_id'].unique():
    sub = df[df['event_id']==ev]
    if len(sub) >= 10:  # Only plot if enough samples
        plt.figure(figsize=(8,4))
        ax = sns.violinplot(data=sub, y='performance_normalized', cut=0)
        
        # Create the KDE for this specific event
        kde = stats.gaussian_kde(sub['performance_normalized'].dropna())
        y_eval = np.linspace(-3, 3, 1000)
        x_density = kde(y_eval)
        max_density = x_density.max()
        
        # Get positions for this specific violin
        x_center = ax.collections[0].get_offsets()[0][0]
        violin_width = ax.collections[0].get_paths()[0].get_extents().width
        half_width = violin_width / 2
        
        # Create density tick marks with actual values
        density_values = [0, max_density/4, max_density/2, 3*max_density/4, max_density]
        density_positions = [x_center - half_width * 0.95, 
                           x_center - half_width * 0.5,
                           x_center,
                           x_center + half_width * 0.5,
                           x_center + half_width * 0.95]
        density_labels = [f'{d:.2f}' for d in density_values]
        
        plt.xticks(density_positions, density_labels)
        plt.xlabel('Probability Density')
        plt.ylabel('Performance (Standard Deviations)')
        
        # Add a clearer explanatory note for each individual event plot
        plt.figtext(0.5, -0.05, 
                  "The x-axis shows probability density values where higher numbers (toward the middle)\n" 
                  "indicate more athletes achieved that performance level.",
                  ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.9))

        plt.title(f'{ev} Z-Score Distribution (n={len(sub)})')
        plt.tight_layout(rect=[0, 0.08, 1, 1])  # Make room for the note
        plt.show()

# 3.3 Performance vs Points (with zero-inflation handling)
plt.figure(figsize=(12, 4))

# Create two subplots
plt.subplot(1, 2, 1)
sns.scatterplot(x='performance_normalized', y='points', data=df, alpha=0.3)
plt.title('Performance vs Points\n(All Data)')

plt.subplot(1, 2, 2)
sns.scatterplot(x='performance_normalized', y='points', 
                data=df[df['points'] > 0], alpha=0.3)
plt.title('Performance vs Points\n(Scoring Performances Only)')

plt.tight_layout()
plt.show()

# Print zero-inflation statistics
total_entries = len(df)
zero_points = (df['points'] == 0).sum()
print(f"\nPoints Distribution Summary:")
print(f"Total Entries: {total_entries}")
print(f"Entries with 0 points: {zero_points} ({(zero_points/total_entries*100):.1f}%)")
print(f"Entries with points: {total_entries-zero_points} ({((total_entries-zero_points)/total_entries*100):.1f}%)")

# 3.4 Average points per athlete
avg_pts = df.groupby('athlete_id')['points'].mean()
print("Sample avg points per athlete:\n", avg_pts.head(), "\n")

# 3.5 Top performers per event
top5 = (df
        .sort_values(['event_id','performance'])
        .groupby('event_id')
        .head(5)
        [['event_id','athlete_id','performance']])
print("Top 5 performers per event:\n", top5, "\n")

# 3.6 Grade‐level comparison
if 'grade_level' in df.columns:
    plt.figure(figsize=(6,4))
    sns.violinplot(x='grade_level', y='performance_normalized', data=df)
    
    # Add sample size annotations
    for i, grade in enumerate(sorted(df['grade_level'].unique())):
        n = len(df[df['grade_level'] == grade])
        plt.text(i, plt.ylim()[0], f'n={n}', horizontalalignment='center', verticalalignment='top')
        
    plt.title('Z-Scores by Grade Level')
    plt.show()

# === 3.7 Event Correlation Analysis ===
print("\n=== Event Correlation Analysis ===")

# Create individual events dataframe - used for correlation analysis
individual_events_df = df[~df['event_id'].apply(is_relay_event)].copy()

# Create athlete-by-event matrix for correlation analysis
perf_mat = pd.pivot_table(
    individual_events_df,
    index='athlete_id',
    columns='event_id',
    values='performance_normalized',
    aggfunc='mean'  # Average z-score if multiple performances
)

print(f"Created event correlation matrix with {perf_mat.shape[0]} athletes and {perf_mat.shape[1]} events")

# Calculate correlation matrix with minimum number of paired observations
perf_corr = perf_mat.corr(min_periods=3)
print(f"Correlation matrix shape: {perf_corr.shape}")

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(perf_corr, annot=True, fmt=".2f", cmap='coolwarm')

plt.title('Event Performance Correlation\n(How similar are normalized performances)')
plt.xlabel('event_id')
plt.ylabel('event_id')

# Add explanatory text for the correlation matrix
plt.figtext(0.5, -0.01, 
           "This matrix shows correlations between normalized performances across different events.\n"
           "Positive values (red) indicate events where athletes tend to perform similarly.\n"
           "Negative values (blue) indicate inverse relationships between event performances.",
           ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.9))

plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Make room for explanation
plt.show()

# Find the strongest positive correlations between events
# Use a different approach that avoids the column name conflict
strong_pos_corr = []
strong_neg_corr = []

# Loop through the upper triangle of the correlation matrix
for i in range(len(perf_corr.columns)):
    for j in range(i+1, len(perf_corr.columns)):
        event1 = perf_corr.columns[i]
        event2 = perf_corr.columns[j]
        corr_val = perf_corr.iloc[i, j]
        
        if not pd.isna(corr_val):  # Skip missing correlations
            if corr_val >= 0:
                strong_pos_corr.append((event1, event2, corr_val))
            else:
                strong_neg_corr.append((event1, event2, corr_val))

# Sort by correlation strength
strong_pos_corr.sort(key=lambda x: x[2], reverse=True)
strong_neg_corr.sort(key=lambda x: x[2])

print("\nTop 10 strongest positive correlations (similar events):")
for event1, event2, corr in strong_pos_corr[:10]:
    print(f"{event1} and {event2}: {corr:.3f}")

print("\nTop 10 strongest negative correlations (dissimilar events):")
for event1, event2, corr in strong_neg_corr[:10]:
    print(f"{event1} and {event2}: {corr:.3f}")

# === 3.8 Event Co-occurrence Analysis ===
print("\n=== Event Co-occurrence Analysis ===")

# For event co-occurrence analysis, keep relays separate
individual_participation = pd.pivot_table(
    individual_events_df,
    index='athlete_id',
    columns='event_id',
    values='performance',
    aggfunc='count'
).fillna(0)

relay_participation = pd.pivot_table(
    df[df['event_id'].apply(is_relay_event)],
    index='athlete_id',
    columns='event_id',
    values='performance',
    aggfunc='count'
).fillna(0)

# Create separate correlation matrices
individual_cooccurrence = (individual_participation > 0).astype(int).corr()
relay_cooccurrence = (relay_participation > 0).astype(int).corr()

binary_participation = (individual_participation > 0).astype(int)
plt.figure(figsize=(12,8))
cooccurrence = binary_participation.corr()
sns.heatmap(cooccurrence, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Event Co-occurrence Matrix\n(How often athletes compete in both events)')
plt.tight_layout()
plt.show()

# Add explanatory text for the co-occurrence matrix
plt.figtext(0.5, -0.01, 
           "This matrix shows how frequently athletes participate in combinations of events.\n"
           "Higher values indicate event pairs that athletes commonly compete in together.",
           ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.9))

# If you want to change the colors to match your original blue colormap:
# sns.heatmap(cooccurrence, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)

# === 4. Event Affinity & Clustering Analysis ===
print("\n=== Athlete Clustering Analysis ===")

# 4.0 Data preparation for clustering
print("Preparing data for clustering:")

# First, create the individual_events_df by filtering out relay events
individual_events_df = df[~df['event_id'].apply(is_relay_event)].copy()
print(f"- Starting with {len(individual_events_df)} individual event performances")
print(f"- Excluded relay events: {', '.join(relay_events)}")

# Create athlete-by-event matrix (each row is an athlete, each column is an event)
perf_mat = pd.pivot_table(
    individual_events_df,
    index='athlete_id',
    columns='event_id',
    values='performance_normalized',
    aggfunc='mean'
)
print(f"- Created performance matrix with {perf_mat.shape[0]} athletes and {perf_mat.shape[1]} events")

# 4.1 Handle missing values (NaN)
print("\nHandling missing values:")
missing_before = perf_mat.isna().sum().sum()
print(f"- Matrix has {missing_before} missing values before imputation")
print("- Missing values represent events an athlete didn't compete in")

# Fill missing values with zeros (meaning average performance)
X_filled = perf_mat.fillna(0)
print("- Strategy: Filling NaN values with 0 (representing average performance)")
print(f"- Final data matrix shape: {X_filled.shape[0]} athletes × {X_filled.shape[1]} events")

# 4.2 Standardize features for clustering
print("\nPreprocessing for clustering:")
X = StandardScaler().fit_transform(X_filled)
print("- Applied StandardScaler to normalize feature scales")

# 4.3 Determine optimal number of clusters using Elbow Method
print("\nDetermining optimal number of clusters:")
inertia = []
ks = range(2,7)
for k in ks:
    inertia.append(KMeans(n_clusters=k, random_state=42).fit(X).inertia_)

plt.figure(figsize=(6,4))
plt.plot(list(ks), inertia, '-o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate('Elbow point indicates\noptimal number of clusters', 
             xy=(3, inertia[1]), xytext=(4, inertia[1]), 
             arrowprops=dict(arrowstyle='->'))
plt.tight_layout()
plt.show()

# 4.4 Apply K-means clustering with the optimal k value
k_final = 3  # Based on elbow method results
print(f"\nApplying K-means clustering with k={k_final}:")
kmeans = KMeans(n_clusters=k_final, random_state=42)
labels = kmeans.fit_predict(X)

# Store cluster assignments back in the original data
perf_mat['cluster'] = labels
cluster_counts = pd.Series(labels).value_counts().sort_index()
for i, count in enumerate(cluster_counts):
    print(f"- Cluster {i}: {count} athletes ({count/len(labels)*100:.1f}%)")

# 4.5 Visualize clusters using PCA for dimensionality reduction
print("\nVisualizing clusters using PCA:")
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X)
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(8,6))
scatter = plt.scatter(proj[:,0], proj[:,1], c=labels, cmap='tab10', alpha=0.7)

# Add cluster centroids to the visualization
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            marker='X', s=100, c='black', label='Centroids')

# Add legend showing cluster sizes
legend_labels = [f'Cluster {i}: {count} athletes' for i, count in enumerate(cluster_counts)]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, 
           title="Athlete Groups", loc="upper right")

# Add explanatory elements to the plot
plt.title('Athlete Clustering Based on Event Performance Patterns')
plt.xlabel(f'First Principal Component ({explained_var[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({explained_var[1]:.1%} variance)')
plt.grid(True, linestyle='--', alpha=0.3)

# Add text box explaining the clustering method
plt.figtext(0.5, -0.05, 
           "Clustering Method: Each athlete is represented by their normalized performance (z-score)\n"
           "across all events. Missing values (events not competed in) are filled with zeros.\n"
           "Athletes are grouped based on similar performance patterns across events.",
           ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for explanation
plt.show()

# 4.6 Analyze the cluster centroids to understand what each cluster represents
print("\nInterpreting cluster characteristics:")
centroids = StandardScaler().fit(X_filled).inverse_transform(kmeans.cluster_centers_)
cent_df = pd.DataFrame(centroids, columns=perf_mat.columns[:-1])

# Get the top 3 events that define each cluster
for i in range(k_final):
    top_events = cent_df.iloc[i].nlargest(3)
    bottom_events = cent_df.iloc[i].nsmallest(3)
    print(f"\nCluster {i} characteristics:")
    print(f"- Top performing events: {', '.join([f'{event} ({val:.2f})' for event, val in top_events.items()])}")
    print(f"- Lowest performing events: {', '.join([f'{event} ({val:.2f})' for event, val in bottom_events.items()])}")

# 4.7 Save cluster assignments for further analysis
perf_mat['cluster'].reset_index().to_csv('athlete_clusters.csv', index=False)
print("\nSaved cluster assignments to 'athlete_clusters.csv'")
