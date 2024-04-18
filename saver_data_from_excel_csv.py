import pandas as pd
import numpy as np

# Define the file path
file_path = 'sir.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, parse_dates=[0], dayfirst=True, dtype=int)

# Create a new column as the row-wise sum of 'dimessi_guariti' and 'deceduti' columns
df['R'] = df['dimessi_guariti'] + df['deceduti']

# Rename the 'totale_positivi' column to 'infected'
df.rename(columns={'totale_positivi': 'I'}, inplace=True)
df.rename(columns={'data.1': 'date'}, inplace=True)

# Remove the 'dimessi_guariti' column
df.drop('dimessi_guariti', axis=1, inplace=True)
df.drop('deceduti', axis=1, inplace=True)

pop_veneto= 4_869_830

df['S']= pop_veneto - df['I'] - df['R']

# Save the DataFrame to a compressed NumPy file
np.savez('veneto_data', S=df['S'], R=df['R'], I=df['I'], date=df['date'])



