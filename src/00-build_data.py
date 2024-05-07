# Imports 
import os
import pandas as pd

def main():
    DATA_DIR = '../data/'
    ORGINAL_DATA_FILE_NAME = 'GlobalLandTemperaturesByCity.csv'

    # Check if the data file exists
    if not os.path.exists(DATA_DIR + ORGINAL_DATA_FILE_NAME):
        os.system('unzip ' + DATA_DIR + ORGINAL_DATA_FILE_NAME + '.zip -d ' + DATA_DIR)

    # Checl if the data.csv already exists
    if not os.path.exists(DATA_DIR + 'data.csv'):
        # Load the data and clean it
        data = pd.read_csv(DATA_DIR + ORGINAL_DATA_FILE_NAME)
        data = data.dropna()
        data = data.drop_duplicates()

        # extract the columns we want 
        columns = ['dt', 'AverageTemperature', 'Latitude', 'Longitude']
        data = data[columns]

        # convert latitude and longitude and date to float
        def process_latitude(lat):
            if lat[-1] == 'N':
                return float(lat[:-1])
            else:
                return -float(lat[:-1])
        def process_longitude(lon):
            if lon[-1] == 'E':
                return float(lon[:-1])
            else:
                return -float(lon[:-1])

        def process_date(date):
            year, month, day = date.split('-')
            decimal = int(month) / 12 + int(day) / 365
            return float(year) + decimal

        data['Latitude'] = data['Latitude'].apply(process_latitude)
        data['Longitude'] = data['Longitude'].apply(process_longitude)
        data['dt_float'] = data['dt'].apply(process_date)

        # Save the whole dataset
        data.to_csv(DATA_DIR + 'data.csv', index=False)


        # save also a smaller dataset 
        data = data[data['dt_float'] >= 1960]
        data.to_csv(DATA_DIR + 'data_1960.csv', index=False)

if __name__ == '__main__':
    main()
