import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create a helper function to standardize all violin plots (add after imports):

def create_violin_plot(data, x, y, title, xlabel=None, ylabel=None):
    """Create a standardized violin plot with proper labels and annotations."""
    plt.figure(figsize=(10, 6))
    
    if x is None:
        # Single variable distribution with proper density axis
        ax = sns.violinplot(data=data, y=y, cut=0)
        
        # Create KDE to get density values
        kde = stats.gaussian_kde(data[y].dropna())
        y_eval = np.linspace(data[y].min(), data[y].max(), 1000)
        x_density = kde(y_eval)
        max_density = x_density.max()
        
        # Calculate positions for density ticks
        x_center = ax.collections[0].get_offsets()[0][0]
        violin_width = ax.collections[0].get_paths()[0].get_extents().width
        half_width = violin_width / 2
        
        # Create even density ticks with actual values
        density_values = [0, max_density/4, max_density/2, 3*max_density/4, max_density]
        density_positions = [x_center - half_width * 0.95, 
                           x_center - half_width * 0.5,
                           x_center,
                           x_center + half_width * 0.5, 
                           x_center + half_width * 0.95]
        density_labels = [f'{d:.2f}' for d in density_values]
        
        plt.xticks(density_positions, density_labels)
        plt.xlabel('Probability Density')
        
        # Add comprehensive explanation of probability density
        plt.figtext(0.5, -0.08, 
                   "About Probability Density: The x-axis values show the density of observations.\n"
                   "Higher values (in the middle) indicate more concentrated data points.\n"
                   "The width of the violin at any height corresponds to how many observations\n"
                   "occur at that performance level, with the widest point being most common.",
                   ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.9))
        plt.tight_layout(rect=[0, 0.12, 1, 0.95])  # Make room for the explanation
        
    else:
        # Multiple violins side by side (categorical)
        ax = sns.violinplot(data=data, x=x, y=y)
        plt.xlabel(xlabel or x)
        
        # Add sample size annotations for categories
        for i, cat in enumerate(data[x].unique()):
            n = len(data[data[x] == cat])
            plt.text(i, plt.ylim()[0] - 0.2, f'n={n}', 
                    horizontalalignment='center',
                    verticalalignment='bottom')
    
    plt.title(title)
    plt.ylabel(ylabel or y)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

# Load the data
df = pd.read_csv('AllMeetsExtracted.csv')

# Add after loading data, before event categories
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

# Define event categories
event_categories = {
    'Sprints': ['100M', '200M', '400M'],
    'Distance': ['800M', '1600M', '3200M'],
    'Hurdles': ['110H', '300H'],
    'Jumps': ['HJ', 'LJ', 'TJ', 'PV'],
    'Throws': ['SP', 'DT'],
    'Relays': ['4x100M', '4x200M', '4x400M', '4x800M']
}

# Add after event categories definition
relay_events = ['4x100M', '4x200M', '4x400M', '4x800M']

def is_relay_event(event_id):
    return any(relay in event_id for relay in relay_events)

# Add category column
# Modify get_event_category function
def get_event_category(event):
    if any(relay in event for relay in relay_events):
        return 'Relays'
    for category, events in event_categories.items():
        if event in events:
            return category
    return 'Other'

df['event_category'] = df['event_id'].apply(get_event_category)

# Check and handle nulls
null_counts = df.isnull().sum()
if df.isnull().values.any():
    print("\nHandling null values:")
    # Handle numeric columns first
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"- Filled {col} nulls with median: {median}")
    
    # Then handle categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            print(f"- Filled {col} nulls with mode: {mode}")
else:
    print("No null values found in dataset")

# Before z-score calculation, separate relay and individual events
individual_df = df[~df['event_id'].apply(is_relay_event)].copy()
relay_df = df[df['event_id'].apply(is_relay_event)].copy()

# Calculate Z-scores only for individual events
individual_df['z_score'] = individual_df.groupby('event_id')['performance'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# For relay events, calculate team-level z-scores
relay_df['z_score'] = relay_df.groupby('event_id')['performance'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Recombine dataframes
df = pd.concat([individual_df, relay_df])

# Add after z-score calculations
def check_outliers(df, group_col, z_col='z_score', threshold=4):
    """Check for outliers within each group."""
    outliers = df[abs(df[z_col]) > threshold].copy()
    if len(outliers) > 0:
        print(f"\nFound {len(outliers)} potential outliers (|z| > {threshold}):")
        summary = outliers.groupby(group_col).size()
        print(f"\nOutliers by {group_col}:")
        print(summary)
        
        # Visualize outliers
        plt.figure(figsize=(12, 6))
        plt.scatter(df[z_col], df['points'], alpha=0.3, label='Normal')
        plt.scatter(outliers[z_col], outliers['points'], 
                   color='red', alpha=0.8, label='Outlier')
        plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=-threshold, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Performance Distribution by {group_col} with Outliers')
        plt.xlabel('Z-Score')
        plt.ylabel('Points')
        plt.legend()
        plt.show()
        
        return outliers
    return pd.DataFrame()

# Check outliers by event category
event_outliers = check_outliers(df, 'event_category')

# Check outliers by event
event_specific_outliers = check_outliers(df, 'event_id')

# Analyze performance by category
category_performance = df.groupby(['meet_id', 'event_category']).agg({
    'z_score': ['mean', 'std', 'count'],
    'points': ['sum', 'mean']
}).round(2)

print("\nPerformance by Event Category:")
print(category_performance)

# Visualize category strengths/weaknesses
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='event_category', y='z_score')
plt.title('Performance Distribution by Event Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Update category performance visualization
create_violin_plot(df, 'event_category', 'z_score', 
                   'Performance Distribution by Event Category', 
                   'Event Category', 'Performance (Z-Score)')

# Points by category
plt.figure(figsize=(10, 5))
category_points = df.groupby('event_category')['points'].sum()
category_points.plot(kind='bar')
plt.title('Total Points by Event Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Points by category with zero-inflation handling
plt.figure(figsize=(12, 6))

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: All data including zeros
category_points_all = df.groupby('event_category')['points'].agg(['sum', 'count', 'mean'])
category_points_all['zero_percentage'] = (df.groupby('event_category')['points']
                                        .apply(lambda x: (x == 0).mean() * 100))
category_points_all['sum'].plot(kind='bar', ax=ax1)
ax1.set_title('Total Points by Category\n(Including Zeros)')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Non-zero points only
df_nonzero = df[df['points'] > 0]
category_points_nonzero = df_nonzero.groupby('event_category')['points'].mean()
category_points_nonzero.plot(kind='bar', ax=ax2)
ax2.set_title('Average Points by Category\n(Excluding Zeros)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Update points distribution plots
plt.figure(figsize=(12, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: All data including zeros with violin plot
create_violin_plot(df, 'event_category', 'points', 
                   f'Points Distribution by Category\n(Including Zeros, n={len(df)})')

# Plot 2: Non-zero points only with violin plot
create_violin_plot(df_nonzero, 'event_category', 'points', 
                   f'Points Distribution by Category\n(Excluding Zeros, n={len(df_nonzero)})')

# Print summary statistics
print("\nPoints Analysis Summary:")
print("\nCategory Statistics (Including Zeros):")
print(category_points_all.round(2))

# 2. Event Participation Analysis

# Count events per athlete
events_per_athlete = df.groupby('athlete_id')['event_id'].nunique()

plt.figure(figsize=(10, 5))
events_per_athlete.value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Events per Athlete')
plt.xlabel('Number of Events')
plt.ylabel('Number of Athletes')
plt.tight_layout()
plt.show()

# Find common event combinations
# Get top pairs
event_combinations = df.groupby('athlete_id')['event_id'].agg(list)
from itertools import combinations

pair_counts = {}
for events in event_combinations:
    if len(events) >= 2:
        for pair in combinations(sorted(set(events)), 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

# Convert to dataframe and sort
pairs_df = pd.DataFrame([(pair[0], pair[1], count) 
                        for pair, count in pair_counts.items()],
                       columns=['Event1', 'Event2', 'Count'])
pairs_df = pairs_df.sort_values('Count', ascending=False)

print("\nTop 10 Most Common Event Combinations:")
print(pairs_df.head(10))

# Create simplified co-occurrence heatmap for top N events
top_n_events = 10
top_events = df['event_id'].value_counts().head(top_n_events).index

# Create matrix for top events
top_cooccurrence = np.zeros((top_n_events, top_n_events))
for i, ev1 in enumerate(top_events):
    for j, ev2 in enumerate(top_events):
        athletes_ev1 = set(df[df['event_id'] == ev1]['athlete_id'])
        athletes_ev2 = set(df[df['event_id'] == ev2]['athlete_id'])
        top_cooccurrence[i, j] = len(athletes_ev1 & athletes_ev2)

plt.figure(figsize=(12, 8))
sns.heatmap(top_cooccurrence, 
            xticklabels=top_events,
            yticklabels=top_events,
            annot=True,
            fmt='g',
            cmap='YlOrRd')
plt.title('Co-occurrence Matrix for Top Events')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Additional insights
print("\nParticipation Statistics:")
print(f"Average events per athlete: {events_per_athlete.mean():.2f}")
print(f"Median events per athlete: {events_per_athlete.median():.2f}")
print("\nMost versatile athletes (competing in most events):")
print(events_per_athlete.nlargest(5))