# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# %%
# Load the new dataset
try:
    # *** UPDATED FILENAME HERE ***
    df = pd.read_csv("C:/Users/firer/Downloads/Temp Proj/AllMeetsExtracted.csv")
    print("Dataset 'AllMeetsExtracted.csv' loaded successfully.")
    # Display initial info
    print("\nDataset Information:")
    df.info()
except FileNotFoundError:
    print("Error: 'AllMeetsExtracted.csv' not found. Please ensure the file is uploaded/accessible.")
except Exception as e:
    print(f"An error occurred during loading: {e}")

# Add after loading data, before preprocessing
print("\nChecking for duplicates...")
total_entries = len(df)

# Check duplicates by athlete and event within same meet
duplicates = df.groupby(['meet_id', 'athlete_id', 'event_id']).size().reset_index(name='count')
duplicate_entries = duplicates[duplicates['count'] > 1]

if len(duplicate_entries) > 0:
    print(f"Found {len(duplicate_entries)} duplicate entries")
    # Keep best performance (lowest time/highest distance) per athlete per event per meet
    df = df.sort_values('performance').groupby(
        ['meet_id', 'athlete_id', 'event_id']
    ).first().reset_index()
    print(f"Removed {total_entries - len(df)} duplicate entries")
else:
    print("No duplicate entries found")

# %%
# --- Preprocessing ---
print("\n--- Preprocessing ---")

# Check for Null Values
print("\nChecking for Null Values:")
null_counts = df.isnull().sum()
print(null_counts)

if df.isnull().values.any():
    print("\nNull values found. Handling nulls:")
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"- Filled {col} nulls with median: {median}")
    
    # Handle categorical columns
    cat_cols = df.select_dtypes(include(['object']).columns)
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            print(f"- Filled {col} nulls with mode: {mode}")
else:
    print("No null values found.")

# Per-event normalization
df['performance_normalized'] = df.groupby('event_id')['performance'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print("\nPerformance normalized per event (Z-score) and added as 'performance_normalized'.")

# Add after z-score calculation
print("\nChecking for extreme Z-scores...")
z_threshold = 4  # Standard threshold for extreme values
extreme_scores = df[abs(df['performance_normalized']) > z_threshold]
if not extreme_scores.empty:
    print(f"\nFound {len(extreme_scores)} extreme performances (|z| > {z_threshold}):")
    print("\nExtreme performance details:")
    print(extreme_scores[['athlete_id', 'event_id', 'performance', 'performance_normalized']])
    
    # Create visualization of extreme scores
    plt.figure(figsize=(12, 6))
    plt.scatter(df['performance_normalized'], df['points'], alpha=0.5, label='Normal')
    plt.scatter(extreme_scores['performance_normalized'], extreme_scores['points'], 
                color='red', alpha=0.8, label='Extreme')
    plt.axvline(x=z_threshold, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=-z_threshold, color='r', linestyle='--', alpha=0.5)
    plt.title('Performance Z-Scores with Extreme Values Highlighted')
    plt.xlabel('Z-Score')
    plt.ylabel('Points')
    plt.legend()
    plt.show()
else:
    print("No extreme z-scores found.")

print("\nPreprocessing steps complete.")
print("\nFirst 5 rows with normalized data:")
print(df.head())

# %%
# --- Visualizations ---
print("\n--- Generating Visualizations ---")
sns.set_style("whitegrid")

# 1. Overall Performance Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['performance_normalized'], kde=True)
plt.title('Overall Distribution of Normalized Performance')
plt.xlabel('Normalized Performance (Z-score)')
plt.ylabel('Frequency')
plt.show()

# After overall performance distribution plot
from scipy import stats

# Function to apply proper density axis to any violin plot
def add_density_axis_to_violin(ax, data_column):
    """Add proper density axis to a violin plot."""
    # Create KDE to get density values
    kde = stats.gaussian_kde(data_column.dropna())
    y_eval = np.linspace(data_column.min(), data_column.max(), 1000)
    x_density = kde(y_eval)
    max_density = x_density.max()
    
    # Get the violin dimensions
    x_center = ax.collections[0].get_offsets()[0][0]
    violin_width = ax.collections[0].get_paths()[0].get_extents().width
    half_width = violin_width / 2
    
    # Create density ticks with actual values
    density_values = [0, max_density/4, max_density/2, 3*max_density/4, max_density]
    density_positions = [x_center - half_width * 0.95, 
                       x_center - half_width * 0.5,
                       x_center,
                       x_center + half_width * 0.5,
                       x_center + half_width * 0.95]
    density_labels = [f'{d:.2f}' for d in density_values]
    
    plt.xticks(density_positions, density_labels)
    plt.xlabel('Probability Density')
    
    return max_density  # Return for potential further use

# For the overall performance distribution plot (around line 95)
plt.figure(figsize=(12, 6))
ax = sns.violinplot(data=df, y='performance_normalized', cut=0)
max_density = add_density_axis_to_violin(ax, df['performance_normalized'])

plt.title(f'Overall Distribution of Normalized Performance (n={len(df)})')
plt.ylabel('Normalized Performance (Z-score)')
plt.show()

# %%
# Performance Distribution by Event Type (Example using top N events for clarity)
top_events = df['event_id'].value_counts().nlargest(6).index
df_top_events = df[df['event_id'].isin(top_events)]

plt.figure(figsize=(15, 10))
sns.boxplot(data=df_top_events, x='performance_normalized', y='event_id', orient='h')
plt.title('Distribution of Normalized Performance for Top 6 Events')
plt.xlabel('Normalized Performance (Z-score)')
plt.ylabel('Event ID')
plt.show()

# Update event-specific plots
plt.figure(figsize=(15, 10))
ax = sns.violinplot(data=df_top_events, x='performance_normalized', y='event_id', orient='h')
plt.title(f'Distribution of Normalized Performance for Top 6 Events')
plt.xlabel('Normalized Performance (Z-score)')
plt.ylabel('Event ID')

# Add sample size annotations with better positioning - FIX OVERLAPPING
for i, event in enumerate(top_events):
    n = len(df_top_events[df_top_events['event_id'] == event])
    # Position the sample size annotation to the left of the y-axis instead of overlapping
    plt.text(
        plt.xlim()[0] - 0.7,  # Position further left of axis
        i,                    # Y-position (event index)
        f'n={n}', 
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
    )

# Adjust figure margins to make room for annotations
plt.subplots_adjust(left=0.15)  # Increase left margin
plt.tight_layout(rect=[0.15, 0, 1, 0.95])  # Adjust layout but preserve left margin
plt.show()

# %%
# 3. Points Optimization Analysis (Performance vs. Points by Event)
num_events = df['event_id'].nunique()
print(f"\nTotal number of unique events: {num_events}")

# Decide whether to plot all events or a subset based on count
if num_events > 12: # Limit to keep the plot readable
    print(f"Plotting points optimization for the top {min(num_events, 12)} most frequent events.")
    events_to_plot = df['event_id'].value_counts().nlargest(12).index
    df_plot = df[df['event_id'].isin(events_to_plot)]
    col_wrap_val = 4
else:
    print("Plotting points optimization for all events.")
    df_plot = df
    col_wrap_val = min(4, num_events) # Adjust wrap based on number of events

# Filter out data with negative points
df_plot = df_plot[df_plot['points'] >= 0]

# Create the lmplot
g = sns.lmplot(x='performance', y='points',
               col='event_id', col_wrap=col_wrap_val,
               data=df_plot, height=4, aspect=1.2,
               line_kws={'color': 'red'}, scatter_kws={'alpha':0.5})
g.set_axis_labels("Performance", "Points Earned")
plt.subplots_adjust(top=0.9) # Adjust layout to make space for title
g.fig.suptitle('Performance vs Points by Event (Potential Optimization)', fontsize=16)
plt.show()

# %%
# Performance Distribution by Grade Level
if 'grade_level' in df.columns:
    plt.figure(figsize=(12, 7))
    # Attempt to order grade levels if they are like 9, 10, 11, 12
    try:
        sorted_grades = sorted(df['grade_level'].unique())
        sns.boxplot(data=df, x='grade_level', y='performance_normalized', order=sorted_grades)
    except: # Fallback if sorting fails
        sns.boxplot(data=df, x='grade_level', y='performance_normalized')

    plt.title('Normalized Performance Distribution by Grade Level')
    plt.xlabel('Grade Level')
    plt.ylabel('Normalized Performance (Z-score)')
    plt.show()
else:
    print("\n'grade_level' column not found, skipping grade level visualization.")

# Update grade level plots
if 'grade_level' in df.columns:
    for grade in sorted(df['grade_level'].unique()):
        grade_data = df[df['grade_level'] == grade]
        if len(grade_data) >= 10:  # Only if enough samples
            plt.figure(figsize=(8, 5))
            ax = sns.violinplot(data=grade_data, y='performance_normalized', cut=0)
            add_density_axis_to_violin(ax, grade_data['performance_normalized'])
            
            plt.title(f'Grade {grade} Performance Distribution (n={len(grade_data)})')
            plt.ylabel('Performance (Z-score)')
            plt.tight_layout()
            plt.show()

# %%
# Correlation Heatmap (between numerical features)
numerical_cols = df.select_dtypes(include=np.number)
cols_for_corr = ['performance', 'points', 'performance_normalized']

# Include grade level if it's numeric
if 'grade_level' in numerical_cols.columns:
     if pd.api.types.is_numeric_dtype(df['grade_level']):
          cols_for_corr.append('grade_level')

# Only compute correlation if there are multiple columns
if len(cols_for_corr) > 1:
    correlation_matrix = df[cols_for_corr].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()
else:
    print("\nNot enough numerical columns selected for a meaningful correlation heatmap.")

# %%
# Handle zero-inflation in points analysis
print("\nPoints Distribution Analysis:")

# Calculate zero-inflation statistics
zero_points_stats = {
    'Total Entries': len(df),
    'Zero Points': (df['points'] == 0).sum(),
    'Scoring Entries': (df['points'] > 0).sum()
}

zero_points_stats['Zero Points %'] = (zero_points_stats['Zero Points'] / 
                                    zero_points_stats['Total Entries'] * 100)

# Print statistics
print("\nZero Points Analysis:")
for key, value in zero_points_stats.items():
    if '%' in key:
        print(f"{key}: {value:.1f}%")
    else:
        print(f"{key}: {value}")

# Create visualizations for zero/non-zero points
plt.figure(figsize=(15, 5))

# Plot 1: All performances
plt.subplot(1, 3, 1)
sns.histplot(data=df, x='points', bins=30)
plt.title('Points Distribution\n(All Performances)')

# Plot 2: Non-zero points only
plt.subplot(1, 3, 2)
sns.histplot(data=df[df['points'] > 0], x='points', bins=30)
plt.title('Points Distribution\n(Scoring Performances Only)')

# Plot 3: Performance comparison
plt.subplot(1, 3, 3)
# Create boolean column for scoring status
df['scored_points'] = df['points'] > 0
sns.boxplot(data=df, x='scored_points', y='performance_normalized')
plt.title('Performance by Scoring Status')
plt.xlabel('Scored Points (True/False)')

plt.tight_layout()
plt.show()


