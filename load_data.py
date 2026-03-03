import pandas as pd
import sqlite3

# Create connection to db
conn = sqlite3.connect('f1_data.db')

print("Connection is successful")

df_races = pd.read_csv("races.csv")
df_constructors = pd.read_csv("constructors.csv")
df_drivers = pd.read_csv("drivers.csv")
df_results = pd.read_csv("results.csv")
df_circuits = pd.read_csv("circuits.csv")
df_constructor_results = pd.read_csv("constructor_results.csv")
df_constructor_standings = pd.read_csv("constructor_standings.csv")
df_driver_standings = pd.read_csv("driver_standings.csv")
df_lap_times = pd.read_csv("lap_times.csv")
df_pit_stops = pd.read_csv("pit_stops.csv")
df_qualifying = pd.read_csv("qualifying.csv")
df_seasons = pd.read_csv("seasons.csv")
df_sprint_results = pd.read_csv("sprint_results.csv")
df_status = pd.read_csv("status.csv")





df_races.to_sql('races', conn, if_exists='replace', index=False)
df_constructors.to_sql('constructors', conn, if_exists='replace', index=False)
df_drivers.to_sql('drivers', conn, if_exists='replace', index=False)
df_results.to_sql('results', conn, if_exists='replace', index=False)
df_circuits.to_sql('circuits', conn, if_exists='replace', index=False)
df_constructor_results.to_sql('constructor_results', conn, if_exists='replace', index=False)
df_constructor_standings.to_sql('constructor_standings', conn, if_exists='replace', index=False)
df_driver_standings.to_sql('driver_standings', conn, if_exists='replace', index=False)
df_lap_times.to_sql('lap_times', conn, if_exists='replace', index=False)
df_pit_stops.to_sql('pit_stops', conn, if_exists='replace', index=False)
df_qualifying.to_sql('qualifying', conn, if_exists='replace', index=False)
df_seasons.to_sql('seasons', conn, if_exists='replace', index=False)
df_sprint_results.to_sql('sprint_results', conn, if_exists='replace', index=False)
df_status.to_sql('status', conn, if_exists='replace', index=False)


conn.close()
print("Дані успішно завантажені в SQLite!")