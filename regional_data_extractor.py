import numpy as np
import pandas as pd

#select the region and the data range to consider for the SIR model
def sir_data_extractor(df, region='Veneto', start_date='2020-10-01', end_date='2021-10-01'):
    df = df[df['denominazione_regione'] == region]
    df = df[['data.1', 'totale_positivi', 'deceduti', 'dimessi_guariti']]
    df['data.1'] = pd.to_datetime(df['data.1'])
    start_date = start_date
    end_data = end_date
    df = df[df['data.1'] >= start_date]
    df = df[df['data.1'] <= end_date]
    return df

#create a dataframe with the appropriate varibles for the SIR model 
def create_SIR_dataframe(df):
    pop_veneto= 4_869_830
    sir_df = pd.DataFrame()
    sir_df['I'] = df['totale_positivi']
    sir_df['R'] = df['dimessi_guariti'] + df['deceduti']
    sir_df['S'] = pop_veneto - sir_df['I'] - sir_df['R']
    sir_df['date'] = df['data.1']
    return sir_df