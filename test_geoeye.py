import sys
import os
sys.path.insert(0, r'D:\geoeye')

from src.utils.weather_client import WeatherClient

def test_weather_api():
    """Test weather integration"""
    api_key = "dc16cac058093397dd5b505a170400bc"
    client = WeatherClient(api_key)
    
    # Test with Iowa corn farm coordinates
    weather = client.get_current_weather(41.878003, -93.097702)
    
    if weather:
        print("✅ GeoEye Weather Module Working!")
        print(f"📍 Location: {weather['location']}")
        print(f"🌡️  Temperature: {weather['temperature']}°C")
        print(f"☁️  Conditions: {weather['conditions']}")
        print(f"💧 Humidity: {weather['humidity']}%")
        print(f"💨 Wind: {weather['wind_speed']} m/s")
        return True
    else:
        print("❌ Weather API Failed")
        return False

if __name__ == "__main__":
    print("🌾 Testing GeoEye System Components")
    print("=" * 40)
    test_weather_api()