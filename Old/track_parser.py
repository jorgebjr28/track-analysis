import pandas as pd
import csv
import re

def standardize_result(result, event_type):
    """Standardize result values based on event type"""
    if not result or result in ['SCR', 'DQ', 'NT', 'ND', 'NH', 'DNS', 'DNF']:
        return None
        
    try:
        if event_type == 'T':  # Track event (time)
            # Convert MM:SS.ss format to seconds
            if ':' in result:
                mins, secs = result.split(':')
                return float(mins) * 60 + float(secs)
            return float(result)
            
        elif event_type == 'F':  # Field event (distance)
            if '-' in result:  # Convert feet-inches to inches
                feet, inches = result.split('-')
                return float(feet) * 12 + float(inches)
            return float(result)
    except ValueError:
        return None
    
    return result

def convert_age_grade(value):
    """Convert age and grade values to numeric when possible"""
    if not value or value == '0':
        return None
    try:
        return int(value)
    except ValueError:
        return value

def get_division_name(division):
    """Convert division number to name"""
    divisions = {
        '1': 'Unified',
        '2': 'Freshmen', 
        '3': 'JV',
        '4': 'Varsity'
    }
    return divisions.get(division, division)

def parse_track_meet_data(file_path):
    # Initialize dictionaries to store parsed data
    athletes = []
    events = []
    results = []
    meet_info = {}
    teams = {}
    
    # Event name mapping
    event_names = {
        '100': '100 Meters',
        '200': '200 Meters',
        '400': '400 Meters',
        '800': '800 Meters',
        '1600': '1600 Meters',
        '3200': '3200 Meters',
        '100H': '100m Hurdles',
        '110H': '110m Hurdles',
        '300H': '300m Hurdles',
        'LJ': 'Long Jump',
        'HJ': 'High Jump',
        'SP': 'Shot Put',
        'DT': 'Discus Throw'
    }

    with open(file_path, 'r') as file:
        for line in file:
            # Remove trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Split the line by semicolons
            fields = line.split(';')
            
            # Process based on record type
            record_type = fields[0]
            
            if record_type == 'H':  # Header record
                meet_info = {
                    'meet_name': fields[1],
                    'start_date': fields[2],
                    'end_date': fields[3],
                    'software': fields[4],
                    'results_date': fields[5]
                }
                
            elif record_type == 'T':  # Team record
                team_id = fields[1]
                team_name = fields[2]
                team_points = fields[3]
                teams[team_id] = {
                    'name': team_name,
                    'points': team_points
                }
                
            elif record_type == 'E':  # Individual event record
                event_type = fields[1]  # T for track, F for field
                event_number = fields[2]
                event_name = fields[3]
                distance = fields[4]
                gender = fields[5]
                division = fields[7]  # Unified, Freshmen, Junior Varsity, Varsity
                
                # Create proper event name
                if not event_name and distance:
                    event_name = event_names.get(distance, f"{distance} Meters")
                elif not event_name:
                    event_name = event_names.get(event_number, f"Event {event_number}")
                
                # Extract athlete information
                last_name = fields[18].strip()
                first_name = fields[19].strip()
                athlete_name = f"{first_name} {last_name}".strip()
                school_code = fields[24].strip()
                school_name = fields[25].strip()
                
                # Create proper athlete ID
                athlete_id = f"{athlete_name}_{school_code}".replace(' ', '_')
                
                athlete = {
                    'id': athlete_id,
                    'name': athlete_name,
                    'gender': fields[20],
                    'dob': fields[21] or None,
                    'age': convert_age_grade(fields[22]),
                    'grade': convert_age_grade(fields[23]),
                    'school_code': school_code,
                    'school_name': school_name
                }
                
                # Fix event gender - no X gender
                if gender == 'X':
                    gender = 'U'  # Mark as unified
                
                event = {
                    'id': f"{event_number}_{event_name}_{gender}_{division}",
                    'number': event_number,
                    'name': event_name,
                    'type': event_type,
                    'gender': gender,
                    'division': division
                }
                
                # Extract result information
                result = fields[10]
                place = fields[12]
                points = fields[13]
                heat = fields[14]
                
                # Add to our data structures
                event_id = f"{event_number}_{event_name}_{gender}_{division}"
                
                # Add athlete if not already in list
                if not any(a.get('id') == athlete_id for a in athletes):
                    athletes.append(athlete)
                
                # Add event if not already in list
                if not any(e.get('id') == event_id for e in events):
                    events.append(event)
                
                # Add result
                results.append({
                    'athlete_id': athlete_id,
                    'event_id': event_id,
                    'result': result,
                    'place': place,
                    'points': points,
                    'heat': heat
                })
                
            elif record_type == 'R':  # Relay event record
                team_name = fields[1]
                event_number = fields[2]
                distance = fields[4]
                gender = fields[5]
                division = fields[7]
                result = fields[10]
                team_code = fields[11]
                place = fields[12]
                points = fields[13]
                
                # Process relay members
                relay_members = []
                member_index = 16
                while member_index < len(fields) and fields[member_index]:
                    member_name = fields[member_index]
                    member_gender = fields[member_index + 2]
                    member_dob = fields[member_index + 3]
                    member_age = fields[member_index + 4]
                    member_grade = fields[member_index + 5]
                    
                    relay_members.append({
                        'name': member_name,
                        'gender': member_gender,
                        'dob': member_dob,
                        'age': member_age,
                        'grade': member_grade
                    })
                    
                    # Each member takes 7 fields
                    member_index += 7
                
                # Add relay event
                event_name = f"{distance}m Relay"
                event_id = f"R_{event_number}_{distance}_{gender}_{division}"
                if not any(e.get('id') == event_id for e in events):
                    events.append({
                        'id': event_id,
                        'number': event_number,
                        'name': event_name,
                        'type': 'R',
                        'gender': gender,
                        'division': division
                    })
                
                # Add relay result
                results.append({
                    'team_name': team_name,
                    'team_code': team_code,
                    'event_id': event_id,
                    'result': result,
                    'place': place,
                    'points': points,
                    'members': relay_members
                })
    
    # Fix athlete DataFrame columns
    athletes_df = pd.DataFrame(athletes)
    athletes_df.columns = ['id', 'name', 'gender', 'dob', 'age', 'grade', 'school_code', 'school_name']
    
    # Clean up empty values
    athletes_df = athletes_df.replace('', pd.NA)
    
    events_df = pd.DataFrame(events)
    results_df = pd.DataFrame(results)
    
    return {
        'meet_info': meet_info,
        'teams': teams,
        'athletes': athletes_df,
        'events': events_df,
        'results': results_df
    }

# Example usage
#data = parse_track_meet_data('BISDCityMeet_25.csv')

def transform_for_analysis(parsed_data):
    performances = []
    
    # Create dictionaries for lookup
    events_dict = {e['id']: e for e in parsed_data['events'].to_dict('records')}
    athletes_dict = {a['id']: a for a in parsed_data['athletes'].to_dict('records')}
    
    for result in parsed_data['results'].to_dict('records'):
        event = events_dict.get(result['event_id'])
        
        if not event:  # Skip if event not found
            continue
            
        if event['type'] == 'R':  # Relay event
            performance = {
                'event_name': event['name'],
                'event_type': 'Relay',
                'event_division': get_division_name(event['division']),
                'result': result['result'],
                'place': result.get('place'),
                'points': int(result.get('points', 0)),
                'meet_name': parsed_data['meet_info']['meet_name'],
                'meet_date': parsed_data['meet_info']['start_date'],
                'team_code': result['team_code'],
                'team_name': result['team_name'],
                'team_members': [m['name'] for m in result.get('members', [])]
            }
        else:  # Individual event
            athlete = athletes_dict.get(result['athlete_id'])
            if not athlete:  # Skip if athlete not found
                continue
                
            performance = {
                'event_name': event['name'],
                'event_type': event['type'],
                'event_division': get_division_name(event['division']),
                'result': standardize_result(result['result'], event['type']),
                'place': int(result['place']) if result['place'].isdigit() else None,
                'points': int(result.get('points', 0)),
                'meet_name': parsed_data['meet_info']['meet_name'],
                'meet_date': parsed_data['meet_info']['start_date'],
                'athlete_name': athlete['name'],
                'athlete_gender': athlete['gender'],
                'athlete_grade': athlete['grade'],
                'athlete_school': athlete['school_name']
            }
            
        performances.append(performance)
                
    return pd.DataFrame(performances)

# Create analysis-ready dataset
#analysis_data = transform_for_analysis(data)


import matplotlib.pyplot as plt
import seaborn as sns

def visualize_athlete_performances(analysis_data, athlete_name):
    """Visualize performances for a specific athlete"""
    athlete_data = analysis_data[analysis_data['athlete_name'] == athlete_name]
    
    # Plot performances by event
    plt.figure(figsize=(12, 6))
    sns.barplot(x='event_name', y='points', data=athlete_data)
    plt.title(f"Performance by Event: {athlete_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_school_strengths(analysis_data):
    """Visualize event strengths by school"""
    # Group by school and event, calculate average points
    school_event_points = analysis_data.groupby(['athlete_school', 'event_name'])['points'].mean().reset_index()
    
    # Plot heatmap
    pivot_data = school_event_points.pivot(index='athlete_school', columns='event_name', values='points')
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', linewidths=.5)
    plt.title("School Strengths by Event (Average Points)")
    plt.tight_layout()
    plt.show()
