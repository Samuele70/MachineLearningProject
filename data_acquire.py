import openmeteo_requests
import datetime
import pandas as pd
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

names      = ["center","mantova", "parma", "cremona", "brescia", "verona"]

frames = {name: [] for name in names}

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
	"latitude": [45.2, 45.1, 44.8, 45.1, 45.5, 45.4],
	"longitude": [10.4, 10.8, 10.3, 10, 10.2, 10.9],
	"start_date": "2020-01-02",
	"end_date": datetime.datetime.today().strftime('%Y-%m-%d'),
	"hourly": ["temperature_2m", "surface_pressure", "wind_speed_100m", "wind_speed_10m", "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", "cloud_cover", "relative_humidity_2m", "rain"],
	"models": ["ecmwf_ifs"],
	"timezone": "UTC",
}

responses = openmeteo.weather_api(url, params = params)
i=0
for response in responses:
	print(f"\nCoordinates: {abs(response.Latitude())}°{'S' if response.Latitude() < 0 else 'N'} {abs(response.Longitude())}°{'W' if response.Longitude() < 0 else 'E'}")
	print(f"Elevation: {response.Elevation()} m asl")
	print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
	print(f"Model Nº: {response.Model()}")
	
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_surface_pressure = hourly.Variables(1).ValuesAsNumpy()
	hourly_wind_speed_100m = hourly.Variables(2).ValuesAsNumpy()
	hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
	hourly_wind_direction_10m = hourly.Variables(4).ValuesAsNumpy()
	hourly_wind_direction_100m = hourly.Variables(5).ValuesAsNumpy()
	hourly_wind_gusts_10m = hourly.Variables(6).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(7).ValuesAsNumpy()
	hourly_relative_humidity_2m = hourly.Variables(8).ValuesAsNumpy()
	hourly_rain = hourly.Variables(9).ValuesAsNumpy()
	
	hourly_data = {
		"date": pd.date_range(
			start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
			end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
			freq = pd.Timedelta(seconds = hourly.Interval()),
			inclusive = "left"
		)
	}
	
	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["surface_pressure"] = hourly_surface_pressure
	hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
	hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
	hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
	hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
	hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
	hourly_data["cloud_cover"] = hourly_cloud_cover
	hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
	hourly_data["rain"] = hourly_rain
	
	hourly_dataframe = pd.DataFrame(data = hourly_data)
	hourly_dataframe.to_csv(f"{names[i]}.csv", index=False)	

	print("\nHourly data\n", hourly_dataframe)
	i+=1
i-=1
merged = pd.read_csv("centre.csv", index_col="date")
while i > 0:
    print(i)
    df = pd.read_csv(f"{names[i]}.csv", index_col="date").add_suffix(f"_{names[i]}")
    merged = merged.join(df)
    i-=1

merged.to_csv("data.csv")