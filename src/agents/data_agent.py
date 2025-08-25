"""
Data Gathering and Preprocessing Agent
Handles all external data collection and standardization
"""

import os
import ee
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

@dataclass
class DataRequest:
    """Standardized data request"""
    field_id: str
    latitude: float
    longitude: float
    radius_m: int = 1000
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_types: List[str] = None  # ['satellite', 'weather', 'soil']

@dataclass
class DataResponse:
    """Standardized data response"""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    errors: List[str]
    timestamp: str

class DataGatheringAgent:
    """Main data collection agent"""
    
    def __init__(self):
        # Try multiple approaches to find and load .env file
        self._load_environment()
        
        self.ee_initialized = False
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")
        
        # Debug info
        print(f"🔍 Environment loading debug:")
        print(f"   Current working directory: {Path.cwd()}")
        print(f"   API key found: {'Yes' if self.weather_api_key else 'No'}")
        if self.weather_api_key:
            print(f"   API key (first 10 chars): {self.weather_api_key[:10]}...")
        
        if not self.weather_api_key:
            raise RuntimeError(
                f"OPENWEATHER_API_KEY not found in environment variables. "
                f"Please check your .env file contains: OPENWEATHER_API_KEY=your_key_here"
            )
        self._init_earth_engine()
    
    def _load_environment(self):
        """Load environment variables with multiple fallback strategies"""
        env_loaded = False
        
        # Strategy 1: Load from current working directory
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(dotenv_path=cwd_env, override=True)
            print(f"✅ Loaded .env from CWD: {cwd_env}")
            env_loaded = True
        
        # Strategy 2: Find project root from this file's location
        this_file = Path(__file__).resolve()
        
        # Try different levels up the directory tree
        for i in range(1, 4):  # Check 1, 2, 3 levels up
            potential_root = this_file.parents[i]
            env_path = potential_root / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=True)
                print(f"✅ Loaded .env from project root ({i} levels up): {env_path}")
                env_loaded = True
                break
        
        # Strategy 3: Load from any .env in current directory without specifying path
        if not env_loaded:
            load_dotenv(override=True)
            print("✅ Attempted to load .env using default strategy")
        
        # Strategy 4: Last resort - check if already in environment
        if os.getenv("OPENWEATHER_API_KEY"):
            print("✅ API key found in environment variables")
            env_loaded = True
        
        if not env_loaded:
            print("⚠️ No .env file loaded successfully")
    
    def _init_earth_engine(self):
        """Initialize Google Earth Engine"""
        try:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            
            if credentials_path and project_id:
                credentials = ee.ServiceAccountCredentials(None, credentials_path)
                ee.Initialize(credentials, project=project_id)
            else:
                ee.Initialize()
            
            self.ee_initialized = True
            print("🛰️ Earth Engine initialized")
        except Exception as e:
            print(f"❌ EE init failed: {e}")
            self.ee_initialized = False
    
    def gather_all_data(self, request: DataRequest) -> DataResponse:
        """Main entry point - gather all requested data types"""
        
        # Set defaults
        if not request.end_date:
            request.end_date = datetime.now().strftime("%Y-%m-%d")
        if not request.start_date:
            start = datetime.now() - timedelta(days=30)
            request.start_date = start.strftime("%Y-%m-%d")
        if not request.data_types:
            request.data_types = ['satellite', 'weather']
        
        collected_data = {}
        errors = []
        
        # Gather satellite data
        if 'satellite' in request.data_types:
            try:
                sat_data = self._get_satellite_data(request)
                collected_data['satellite'] = sat_data
            except Exception as e:
                errors.append(f"Satellite data failed: {e}")
        
        # Gather weather data (store only on success)
        if 'weather' in request.data_types:
            try:
                weather_data = self._get_weather_data(request)
                if isinstance(weather_data, dict) and 'error' in weather_data:
                    errors.append(f"Weather data failed: {weather_data['error']}")
                else:
                    collected_data['weather'] = weather_data
            except Exception as e:
                errors.append(f"Weather data failed: {e}")
        
        # Gather soil data (placeholder)
        if 'soil' in request.data_types:
            try:
                soil_data = self._get_soil_data(request)
                collected_data['soil'] = soil_data
            except Exception as e:
                errors.append(f"Soil data failed: {e}")
        
        return DataResponse(
            success=len(errors) == 0,
            data=collected_data,
            metadata={
                "request": request.__dict__,
                "collection_time": datetime.now().isoformat(),
                "data_sources": list(collected_data.keys())
            },
            errors=errors,
            timestamp=datetime.now().isoformat()
        )
    
    def _get_satellite_data(self, request: DataRequest) -> Dict[str, Any]:
        """Collect and preprocess satellite data"""
        if not self.ee_initialized:
            raise Exception("Earth Engine not initialized")
        
        # Create geometry
        point = ee.Geometry.Point([request.longitude, request.latitude])
        area = point.buffer(request.radius_m)
        
        # Get Sentinel-2 collection
        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")  # Updated collection
            .filterBounds(area)
            .filterDate(request.start_date, request.end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))
        
        # Get collection info
        collection_list = collection.getInfo()
        image_count = len(collection_list['features'])
        
        if image_count == 0:
            return {"error": "No satellite images found", "image_count": 0}
        
        # Get best image (least cloudy)
        best_image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
        image_info = best_image.getInfo()
        
        # Calculate indices (comprehensive set)
        indices = self._calculate_all_indices(best_image)
        
        # Get statistics
        stats = indices.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.minMax(), sharedInputs=True
            ).combine(
                ee.Reducer.stdDev(), sharedInputs=True
            ),
            geometry=area,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        # Preprocess statistics
        processed_stats = self._process_satellite_stats(stats)
        
        return {
            "source": "Sentinel-2 SR Harmonized",
            "image_count": image_count,
            "best_image_id": image_info['id'],
            "acquisition_date": image_info['properties']['system:time_start'],
            "cloud_cover": image_info['properties']['CLOUDY_PIXEL_PERCENTAGE'],
            "area_analyzed_ha": (3.14159 * (request.radius_m/1000)**2),
            "indices": processed_stats,
            "quality_flags": {
                "sufficient_images": image_count >= 1,
                "low_cloud_cover": image_info['properties']['CLOUDY_PIXEL_PERCENTAGE'] < 20,
                "recent_data": True  # Within requested timeframe
            }
        }
    
    def _calculate_all_indices(self, image):
        """Calculate comprehensive vegetation indices"""
        
        # Vegetation indices
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndre = image.normalizedDifference(["B8", "B5"]).rename("NDRE") 
        gndvi = image.normalizedDifference(["B8", "B3"]).rename("GNDVI")
        
        # Water indices
        ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
        ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
        
        # Enhanced vegetation
        savi = image.expression(
            "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
            {"NIR": image.select("B8"), "RED": image.select("B4")}
        ).rename("SAVI")
        
        evi2 = image.expression(
            "2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)",
            {"NIR": image.select("B8"), "RED": image.select("B4")}
        ).rename("EVI2")
        
        # Red-edge indices
        mcari = image.expression(
            "((RE1 - RED) - 0.2 * (RE1 - GREEN)) * (RE1 / RED)",
            {"RE1": image.select("B5"), "RED": image.select("B4"), "GREEN": image.select("B3")}
        ).rename("MCARI")
        
        # Combine all
        return ndvi.addBands([ndre, gndvi, ndwi, ndmi, savi, evi2, mcari])
    
    def _process_satellite_stats(self, stats: Dict) -> Dict[str, Dict[str, float]]:
        """Clean and organize satellite statistics"""
        indices = ["NDVI", "NDRE", "GNDVI", "NDWI", "NDMI", "SAVI", "EVI2", "MCARI"]
        
        processed = {}
        for index in indices:
            processed[index] = {
                "mean": float(stats.get(f"{index}_mean", 0) or 0),
                "min": float(stats.get(f"{index}_min", 0) or 0),
                "max": float(stats.get(f"{index}_max", 0) or 0),
                "std": float(stats.get(f"{index}_stdDev", 0) or 0),
                "range": float(stats.get(f"{index}_max", 0) or 0) - float(stats.get(f"{index}_min", 0) or 0)
            }
        
        return processed
    
    def _get_weather_data(self, request: DataRequest) -> Dict[str, Any]:
        """Collect weather data from OpenWeather API"""
        
        if not self.weather_api_key:
            return {"error": "OpenWeather API key not configured"}
        
        base_url = "https://api.openweathermap.org/data/2.5"
        
        # Current weather
        current_url = f"{base_url}/weather"
        current_params = {
            "lat": request.latitude,
            "lon": request.longitude,
            "appid": self.weather_api_key,
            "units": "metric"
        }
        
        try:
            current_response = requests.get(current_url, params=current_params)
            current_response.raise_for_status()
            current_data = current_response.json()
            
            # Process current weather
            processed_current = self._process_current_weather(current_data)
            
            # Get 5-day forecast
            forecast_url = f"{base_url}/forecast"
            forecast_params = current_params.copy()
            forecast_params["cnt"] = 40  # 5 days * 8 (3-hour intervals)
            
            forecast_response = requests.get(forecast_url, params=forecast_params)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            # Process forecast
            processed_forecast = self._process_forecast_data(forecast_data)
            
            return {
                "source": "OpenWeatherMap",
                "current": processed_current,
                "forecast": processed_forecast,
                "quality_flags": {
                    "current_available": True,
                    "forecast_available": True,
                    "api_responsive": True
                }
            }
            
        except Exception as e:
            return {"error": f"Weather API failed: {e}"}
    
    def _process_current_weather(self, data: Dict) -> Dict[str, Any]:
        """Process current weather data"""
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        
        # Calculate VPD
        temp = main.get("temp", 20)
        humidity = main.get("humidity", 50)
        vpd = self._calculate_vpd(temp, humidity)
        
        return {
            "temperature": temp,
            "humidity": humidity,
            "pressure": main.get("pressure"),
            "description": weather.get("description"),
            "wind_speed": wind.get("speed", 0),
            "wind_direction": wind.get("deg", 0),
            "vpd": vpd,
            "feels_like": main.get("feels_like"),
            "visibility": data.get("visibility", 10000) / 1000,  # km
            "timestamp": data.get("dt"),
            "location": data.get("name", "Unknown")
        }
    
    def _process_forecast_data(self, data: Dict) -> List[Dict[str, Any]]:
        """Process forecast data"""
        forecast_list = []
        
        for item in data.get("list", [])[:24]:  # Next 24 time points (3 days)
            main = item.get("main", {})
            weather = item.get("weather", [{}])[0]
            wind = item.get("wind", {})
            
            temp = main.get("temp", 20)
            humidity = main.get("humidity", 50)
            
            forecast_list.append({
                "datetime": item.get("dt_txt"),
                "timestamp": item.get("dt"),
                "temperature": temp,
                "humidity": humidity,
                "pressure": main.get("pressure"),
                "description": weather.get("description"),
                "wind_speed": wind.get("speed", 0),
                "precipitation": item.get("rain", {}).get("3h", 0) + item.get("snow", {}).get("3h", 0),
                "vpd": self._calculate_vpd(temp, humidity)
            })
        
        return forecast_list
    
    def _calculate_vpd(self, temperature: float, humidity: float) -> float:
        """Calculate Vapor Pressure Deficit"""
        svp = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
        avp = svp * (humidity / 100)
        return round(svp - avp, 3)
    
    def _get_soil_data(self, request: DataRequest) -> Dict[str, Any]:
        """Placeholder for soil data (SoilGrids, local sensors, etc.)"""
        return {
            "source": "placeholder",
            "note": "Integrate SoilGrids API or local sensor data",
            "available": False
        }

# Convenience functions for easy integration
def collect_field_data(field_id: str, lat: float, lon: float, 
                      radius_m: int = 1000, data_types: List[str] = None) -> DataResponse:
    """Simple function to collect all data for a field"""
    
    agent = DataGatheringAgent()
    request = DataRequest(
        field_id=field_id,
        latitude=lat,
        longitude=lon,
        radius_m=radius_m,
        data_types=data_types or ['satellite', 'weather']
    )
    
    return agent.gather_all_data(request)

if __name__ == "__main__":
    # Test the data agent
    result = collect_field_data(
        field_id="test_field",
        lat=41.878003,
        lon=-93.097702,
        radius_m=1000
    )
    
    print(f"✅ Success: {result.success}")
    print(f"📊 Data sources: {list(result.data.keys())}")
    print(f"⏰ Timestamp: {result.timestamp}")
    
    # Check satellite data
    if 'satellite' in result.data:
        sat_data = result.data['satellite']
        print(f"\n🛰️ Satellite: {sat_data.get('image_count', 0)} images")
        if 'indices' in sat_data and 'NDVI' in sat_data['indices']:
            print(f"   NDVI: {sat_data['indices']['NDVI'].get('mean', 'N/A')}")
    
    # Check weather data - handle potential errors
    if 'weather' in result.data:
        weather_data = result.data['weather']
        if 'error' in weather_data:
            print(f"\n🌧️ Weather Error: {weather_data['error']}")
        elif 'current' in weather_data:
            current = weather_data['current']
            print(f"\n🌤️ Weather: {current.get('temperature', 'N/A')}°C, {current.get('humidity', 'N/A')}% humidity")
        else:
            print(f"\n🌤️ Weather data structure: {list(weather_data.keys())}")
    
    if result.errors:
        print(f"\n❌ Errors: {result.errors}")