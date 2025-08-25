import requests
from datetime import datetime
import os

class WeatherClient:
    def __init__(self, api_key=None):
        # Try to get from parameter first, then environment variable
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Pass it directly or set OPENWEATHER_API_KEY environment variable.")
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, latitude, longitude):
        """Get current weather for farm location"""
        url = f"{self.base_url}/weather"
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'precipitation': data.get('rain', {}).get('1h', 0),
                    'wind_speed': data['wind']['speed'],
                    'conditions': data['weather'][0]['description'],
                    'location': data['name'],
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Weather API error: {e}")
        return None