import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV files
events_df = pd.read_csv('events.csv')
athletes_df = pd.read_csv('athletes.csv')
performances_df = pd.read_csv('processed_performances.csv')

# Basic analysis examples:

# 1. View top performers by event
def show_top_performers(event_name, n=5):
    """Show top performers for an event"""
    # Update to handle partial matches
    event_results = performances_df[performances_df['event_name'].str.contains(event_name, case=False, na=False)]
    if event_results.empty:
        print(f"No results found for event: {event_name}")
        return pd.DataFrame()
    
    return event_results.sort_values(['result', 'place']).head(n)

# 2. Create a school performance summary
def school_summary():
    return performances_df.groupby('athlete_school')['points'].agg(['count', 'sum', 'mean'])

# 3. Visualize performance distribution
def plot_results_distribution(event_name):
    event_data = performances_df[performances_df['event_name'] == event_name]
    plt.figure(figsize=(10, 6))
    sns.histplot(data=event_data, x='result', bins=20)
    plt.title(f'Result Distribution: {event_name}')
    plt.show()

# Example usage:
print("Sample of events:")
print(events_df.head())

print("\nTop performers in 100m:")
print(show_top_performers('100 Meters'))  # Update to match exact event name

print("\nSchool performance summary:")
# Group by actual school name instead of gender
print(performances_df.groupby('athlete_school')['points'].agg(['count', 'sum', 'mean']).dropna())