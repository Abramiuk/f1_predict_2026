import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

print("1. Initializing V14 Engine (True Driver Impact)...")
conn = sqlite3.connect('f1_data.db')

# 1. Base Race Results + is_reg_change FLAG
df_race = pd.read_sql_query("""
SELECT res.raceId, races.circuitId, res.driverId, res.constructorId, res.grid, res.positionOrder, races.date as race_date,
       CASE WHEN races.year IN (2014, 2017, 2022) THEN 1 ELSE 0 END as is_reg_change
FROM results res JOIN races ON res.raceId = races.raceId
WHERE res.grid > 0 AND races.year >= 2012 AND res.statusId IN (1, 11, 12, 13, 14, 15, 16) 
""", conn)

# 2. Qualifying History
df_quali_history = pd.read_sql_query("""
SELECT driverId, AVG(position) as raw_quali_pace 
FROM qualifying JOIN races ON qualifying.raceId = races.raceId 
WHERE races.year >= 2020 GROUP BY driverId
""", conn)

df_drivers = pd.read_sql_query("SELECT driverId, dob FROM drivers", conn)

# 3. TRUE DRIVER IMPACT (Calculate % of team points scored by the driver, YEAR BY YEAR)
df_driver_impact = pd.read_sql_query("""
WITH DriverYearly AS (
    SELECT res.driverId, r.year, res.constructorId, SUM(res.points) as d_pts
    FROM results res JOIN races r ON res.raceId = r.raceId
    WHERE r.year >= 2014
    GROUP BY res.driverId, r.year, res.constructorId
),
TeamYearly AS (
    SELECT constructorId, year, SUM(points) as t_pts
    FROM results res JOIN races r ON res.raceId = r.raceId
    WHERE r.year >= 2014
    GROUP BY constructorId, year
)
SELECT dy.driverId,
       AVG(CAST(dy.d_pts AS FLOAT) / CASE WHEN ty.t_pts = 0 THEN 1 ELSE ty.t_pts END) as driver_impact
FROM DriverYearly dy
JOIN TeamYearly ty ON dy.constructorId = ty.constructorId AND dy.year = ty.year
GROUP BY dy.driverId
""", conn)

# 4. TEAM FORM (Absolute car speed over recent years)
df_recent_team = pd.read_sql_query("""
SELECT res.constructorId, SUM(res.points) as team_form_pts
FROM results res JOIN races r ON res.raceId = r.raceId
WHERE r.year >= 2024 GROUP BY res.constructorId
""", conn)

# 5. Adaptation Deltas
df_team_adapt = pd.read_sql_query("""
SELECT res.constructorId,
       (SUM(CASE WHEN r.year IN (2022, 2023) THEN res.points ELSE 0 END) - 
        SUM(CASE WHEN r.year IN (2020, 2021) THEN res.points ELSE 0 END)) * 1.5 + 
       (SUM(CASE WHEN r.year IN (2014, 2015) THEN res.points ELSE 0 END) - 
        SUM(CASE WHEN r.year IN (2012, 2013) THEN res.points ELSE 0 END)) * 0.2 as team_adapt_delta
FROM results res JOIN races r ON res.raceId = r.raceId GROUP BY res.constructorId
""", conn)

df_driver_adapt = pd.read_sql_query("""
SELECT res.driverId,
       (SUM(CASE WHEN r.year IN (2022, 2023) THEN res.points ELSE 0 END) - 
        SUM(CASE WHEN r.year IN (2020, 2021) THEN res.points ELSE 0 END)) * 1.2 + 
       (SUM(CASE WHEN r.year IN (2014, 2015) THEN res.points ELSE 0 END) - 
        SUM(CASE WHEN r.year IN (2012, 2013) THEN res.points ELSE 0 END)) as driver_adapt_delta
FROM results res JOIN races r ON res.raceId = r.raceId GROUP BY res.driverId
""", conn)
conn.close()

print("2. Merging and Cleaning Data...")
df = df_race.merge(df_drivers, on='driverId', how='left')
df = df.merge(df_driver_impact, on='driverId', how='left')
df = df.merge(df_recent_team, on='constructorId', how='left')
df = df.merge(df_team_adapt, on='constructorId', how='left')
df = df.merge(df_driver_adapt, on='driverId', how='left')
df = df.merge(df_quali_history, on='driverId', how='left')

df = df.replace('\\N', pd.NA)
df[['grid', 'circuitId', 'positionOrder']] = df[['grid', 'circuitId', 'positionOrder']].apply(pd.to_numeric, errors='coerce')
df['race_date'] = pd.to_datetime(df['race_date'])
df['dob'] = pd.to_datetime(df['dob'])
df['driver_age'] = (df['race_date'] - df['dob']).dt.days / 365.25

df['team_form_pts'] = df['team_form_pts'].fillna(0)
df['driver_impact'] = df['driver_impact'].fillna(0.3) 
df['raw_quali_pace'] = df['raw_quali_pace'].fillna(18.0) 
df['team_adapt_delta'] = df['team_adapt_delta'].fillna(0)
df['driver_adapt_delta'] = df['driver_adapt_delta'].fillna(0)
df = df.dropna(subset=['driver_age', 'positionOrder', 'grid', 'circuitId'])


print("3. Training AIs with Context Drift Detection...")
# AI uses driver_impact instead of raw points
X_q = df[['circuitId', 'driver_age', 'team_form_pts', 'driver_impact', 'raw_quali_pace', 'team_adapt_delta', 'is_reg_change']]
y_q = df['grid']
model_quali = RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42)
model_quali.fit(X_q, y_q)

X_r = df[['grid', 'circuitId', 'driver_age', 'team_form_pts', 'driver_impact', 'team_adapt_delta', 'driver_adapt_delta', 'is_reg_change']]
y_r = df['positionOrder']
model_race = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
model_race.fit(X_r, y_r)

print("✅ Systems Ready!\n")

# FULL F1 CALENDAR
CIRCUITS = {
    1: 'Melbourne', 3: 'Bahrain', 4: 'Barcelona', 6: 'Monaco', 7: 'Montreal', 
    9: 'Silverstone', 11: 'Hungaroring', 13: 'Spa', 14: 'Monza', 15: 'Singapore', 
    17: 'Shanghai', 18: 'Interlagos', 21: 'Imola', 22: 'Suzuka', 24: 'Abu Dhabi',
    32: 'Mexico City', 39: 'Zandvoort', 69: 'Austin', 70: 'Red Bull Ring', 
    73: 'Baku', 77: 'Jeddah', 79: 'Miami', 80: 'Las Vegas', 81: 'Qatar'
}

TEAMS = {1: 'McLaren', 3: 'Williams', 6: 'Ferrari', 9: 'Red Bull', 15: 'Sauber/Audi', 117: 'Aston Martin', 131: 'Mercedes', 210: 'Haas', 213: 'VCARB', 214: 'Alpine', 215: 'Cadillac'}
ROSTER_2026 = [
    (1, 6, "L. Hamilton"), (844, 6, "C. Leclerc"), (846, 1, "L. Norris"), (857, 1, "O. Piastri"),
    (830, 9, "M. Verstappen"), (856, 9, "L. Lawson"), (847, 131, "G. Russell"), (864, 131, "K. Antonelli"),
    (4, 117, "F. Alonso"), (840, 117, "L. Stroll"), (842, 214, "P. Gasly"), (865, 214, "J. Doohan"),
    (832, 3, "C. Sainz"), (848, 3, "A. Albon"), (852, 213, "Y. Tsunoda"), (866, 213, "I. Hadjar"),
    (807, 15, "N. Hulkenberg"), (867, 15, "G. Bortoleto"), (839, 210, "E. Ocon"), (862, 210, "O. Bearman"),
    (815, 215, "S. Perez"), (77, 215, "V. Bottas")
]

try:
    print("AVAILABLE CIRCUITS:")
    for cid, cname in sorted(CIRCUITS.items()):
        print(f"ID: {cid:<2} | {cname}")
        
    race_circuit = int(input("\nEnter Circuit ID: "))
    
    CURRENT_YEAR_REG_CHANGE = 1 
    
    quali_predictions = []
    for driver_id, team_id, name in ROSTER_2026:
        d_info = df_drivers[df_drivers['driverId'] == driver_id]
        d_impact_info = df_driver_impact[df_driver_impact['driverId'] == driver_id]
        t_form_info = df_recent_team[df_recent_team['constructorId'] == team_id]
        qp_info = df_quali_history[df_quali_history['driverId'] == driver_id]
        t_adapt = df_team_adapt[df_team_adapt['constructorId'] == team_id]
        d_adapt = df_driver_adapt[df_driver_adapt['driverId'] == driver_id]
        
        age = (datetime(2026, 3, 1) - pd.to_datetime(d_info['dob'].iloc[0])).days / 365.25 if not d_info.empty else 23.0
        impact = d_impact_info['driver_impact'].iloc[0] if not d_impact_info.empty else 0.3
        if impact < 0.30:
            impact = 0.30
        t_pts = t_form_info['team_form_pts'].iloc[0] if not t_form_info.empty else 0
        raw_q_pace = qp_info['raw_quali_pace'].iloc[0] if not qp_info.empty else 18.0
        t_ad = t_adapt['team_adapt_delta'].iloc[0] if not t_adapt.empty else 0
        d_ad = d_adapt['driver_adapt_delta'].iloc[0] if not d_adapt.empty else 0
        
        if CURRENT_YEAR_REG_CHANGE == 1:
            t_pts = t_pts * 0.7  
            
        q_score = model_quali.predict([[race_circuit, age, t_pts, impact, raw_q_pace, t_ad, CURRENT_YEAR_REG_CHANGE]])[0]
        
        quali_predictions.append({
            'id': driver_id, 'name': name, 'team': TEAMS[team_id], 
            'age': age, 't_pts': t_pts, 'impact': impact, 't_ad': t_ad, 'd_ad': d_ad,
            'q_score': q_score
        })
    
    quali_predictions.sort(key=lambda x: x['q_score'])
    
    grid_list = quali_predictions.copy()
    while True:
        print(f"\n--- AI PREDICTED STARTING GRID ({CIRCUITS.get(race_circuit, 'Track')}) ---")
        for i, driver in enumerate(grid_list):
            print(f"P{i+1:<2} | ID: {driver['id']:<4} | {driver['name']:<15} | {driver['team']:<14} | Impact: {driver['impact']*100:>3.0f}%")
            
        cmd = int(input("\nCommand (Driver ID to move, or 0 to race): "))
        if cmd == 0:
            break
            
        target_idx = next((i for i, d in enumerate(grid_list) if d['id'] == cmd), None)
        if target_idx is not None:
            new_pos = int(input(f"New starting position for {grid_list[target_idx]['name']} (1-22): "))
            driver_obj = grid_list.pop(target_idx)
            grid_list.insert(new_pos - 1, driver_obj)
        else:
            print("Driver ID not found!")

    print("\n🏎️ SIMULATING RACE...\n")
    
    race_results = []
    for index, driver in enumerate(grid_list):
        starting_grid_pos = index + 1
        r_score = model_race.predict([[starting_grid_pos, race_circuit, driver['age'], driver['t_pts'], driver['impact'], driver['t_ad'], driver['d_ad'], CURRENT_YEAR_REG_CHANGE]])[0]
        
        race_results.append({
            'start_pos': starting_grid_pos,
            'name': driver['name'],
            'team': driver['team'],
            'r_score': r_score
        })
        
    race_results.sort(key=lambda x: x['r_score'])
    
    print("=========================================================")
    print(f"🏆 FINAL RACE CLASSIFICATION 🏆")
    print("=========================================================")
    print("FINISH | DRIVER                 | TEAM             | START ")
    print("---------------------------------------------------------")
    for final_pos, p in enumerate(race_results):
        pos_change = p['start_pos'] - (final_pos + 1)
        change_symbol = f"+{pos_change}" if pos_change > 0 else str(pos_change) if pos_change < 0 else "-"
        print(f"P{final_pos+1:<5} | {p['name']:<22} | {p['team']:<16} | Grid P{p['start_pos']} ({change_symbol})")

except ValueError:
    print("Invalid input!")