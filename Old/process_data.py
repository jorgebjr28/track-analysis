import pandas as pd
from track_parser import parse_track_meet_data, transform_for_analysis, convert_age_grade

# 1. Parse raw data
raw_data = parse_track_meet_data('BISDCityMeet_25.csv')

# 2. Transform and standardize
analysis_data = transform_for_analysis(raw_data)

# 3. Save processed data
analysis_data.to_csv('processed_performances.csv', index=False)
raw_data['athletes'].to_csv('athletes.csv', index=False)
raw_data['events'].to_csv('events.csv', index=False)

print("Processing complete! Saved files:")
print("- processed_performances.csv")
print("- athletes.csv")
print("- events.csv")
