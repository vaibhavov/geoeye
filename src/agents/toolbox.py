"""
Complete Toolbox for Crop Health Monitoring Agent
All tools follow the same pattern: take AgentState, return AgentState
"""

import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Earth Engine
import ee

# Weather APIs (you'll need to install these)
import requests
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
except ImportError:
    print("⚠️ Install openmeteo for weather tools: pip install openmeteo-requests")

# =========================
# EARTH ENGINE TOOLS
# =========================

def tool_compute_indices(state) -> dict:
    """Enhanced Sentinel-2 indices with more vegetation indices"""
    try:
        lat, lon, radius = state["latitude"], state["longitude"], state["radius_m"]
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=30)

        farm_point = ee.Geometry.Point([lon, lat])
        farm_area = farm_point.buffer(radius)

        collection = (ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(farm_area)
            .filterDate(str(start_date), str(end_date))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .sort("CLOUDY_PIXEL_PERCENTAGE"))

        image_count = collection.size().getInfo()
        if image_count == 0:
            state["artifacts"]["compute_indices"] = {"error": "No suitable images"}
            return state

        img = collection.median()

        # Calculate comprehensive indices
        # Basic vegetation
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndre = img.normalizedDifference(["B8", "B5"]).rename("NDRE")
        gndvi = img.normalizedDifference(["B8", "B3"]).rename("GNDVI")
        
        # Water indices
        ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
        ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
        
        # Advanced vegetation
        savi = img.expression(
            "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
            {"NIR": img.select("B8"), "RED": img.select("B4")}
        ).rename("SAVI")
        
        evi2 = img.expression(
            "2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)",
            {"NIR": img.select("B8"), "RED": img.select("B4")}
        ).rename("EVI2")
        
        # Red-edge indices
        mcari = img.expression(
            "((RE1 - RED) - 0.2 * (RE1 - GREEN)) * (RE1 / RED)",
            {"RE1": img.select("B5"), "RED": img.select("B4"), "GREEN": img.select("B3")}
        ).rename("MCARI")
        
        cire = img.expression(
            "(RE3 / RE1) - 1",
            {"RE3": img.select("B7"), "RE1": img.select("B5")}
        ).rename("CIre")

        stack = ndvi.addBands([ndre, gndvi, ndwi, ndmi, savi, evi2, mcari, cire])

        # Stats
        reducer = (ee.Reducer.mean()
                  .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
                  .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
                  .combine(reducer2=ee.Reducer.percentile([10, 25, 75, 90]), sharedInputs=True))

        stats = stack.reduceRegion(reducer=reducer, geometry=farm_area, scale=10, maxPixels=1e9).getInfo()

        def extract_stats(index_name):
            return {
                "mean": float(stats.get(f"{index_name}_mean", 0) or 0),
                "min": float(stats.get(f"{index_name}_min", 0) or 0),
                "max": float(stats.get(f"{index_name}_max", 0) or 0),
                "std": float(stats.get(f"{index_name}_stdDev", 0) or 0),
                "p10": float(stats.get(f"{index_name}_p10", 0) or 0),
                "p25": float(stats.get(f"{index_name}_p25", 0) or 0),
                "p75": float(stats.get(f"{index_name}_p75", 0) or 0),
                "p90": float(stats.get(f"{index_name}_p90", 0) or 0),
            }

        out = {
            "period": {"start": str(start_date), "end": str(end_date)},
            "image_count": image_count,
            "area_ha": float(3.14159 * (radius/1000.0)**2),
            "NDVI": extract_stats("NDVI"),
            "NDRE": extract_stats("NDRE"),
            "GNDVI": extract_stats("GNDVI"),
            "NDWI": extract_stats("NDWI"),
            "NDMI": extract_stats("NDMI"),
            "SAVI": extract_stats("SAVI"),
            "EVI2": extract_stats("EVI2"),
            "MCARI": extract_stats("MCARI"),
            "CIre": extract_stats("CIre"),
        }
        
        state["artifacts"]["compute_indices"] = out
        return state
    except Exception as e:
        state["artifacts"]["compute_indices"] = {"error": f"indices failed: {e}"}
        return state

def tool_time_series_analysis(state) -> dict:
    """Get time series trends for key indices"""
    try:
        lat, lon, radius = state["latitude"], state["longitude"], state["radius_m"]
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=90)  # 3 months

        farm_point = ee.Geometry.Point([lon, lat])
        farm_area = farm_point.buffer(radius)

        collection = (ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(farm_area)
            .filterDate(str(start_date), str(end_date))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))

        def add_indices(image):
            ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndre = image.normalizedDifference(["B8", "B5"]).rename("NDRE")
            return image.addBands([ndvi, ndre]).set('system:time_start', image.get('system:time_start'))

        with_indices = collection.map(add_indices)
        
        # Create time series
        def get_time_series(index_name):
            def extract_value(image):
                value = image.select(index_name).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=farm_area,
                    scale=20,
                    maxPixels=1e6
                ).get(index_name)
                
                return ee.Feature(None, {
                    'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
                    'value': value,
                    'timestamp': image.get('system:time_start')
                })
            
            series = with_indices.map(extract_value)
            return series.getInfo()['features']

        ndvi_series = get_time_series('NDVI')
        ndre_series = get_time_series('NDRE')
        
        # Calculate trends (simple slope)
        def calculate_trend(series):
            values = [f['properties']['value'] for f in series if f['properties']['value'] is not None]
            if len(values) < 3:
                return {"trend": "insufficient_data", "slope": 0}
            
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
                
            return {"trend": trend, "slope": float(slope), "points": len(values)}

        out = {
            "ndvi_trend": calculate_trend(ndvi_series),
            "ndre_trend": calculate_trend(ndre_series),
            "time_series": {
                "ndvi": ndvi_series[-10:],  # Last 10 points
                "ndre": ndre_series[-10:]
            }
        }
        
        state["artifacts"]["time_series"] = out
        return state
    except Exception as e:
        state["artifacts"]["time_series"] = {"error": f"time series failed: {e}"}
        return state

def tool_spatial_analysis(state) -> dict:
    """Analyze spatial patterns and hotspots"""
    try:
        lat, lon, radius = state["latitude"], state["longitude"], state["radius_m"]
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=14)

        farm_point = ee.Geometry.Point([lon, lat])
        farm_area = farm_point.buffer(radius)

        collection = (ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(farm_area)
            .filterDate(str(start_date), str(end_date))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))

        if collection.size().getInfo() == 0:
            state["artifacts"]["spatial"] = {"error": "No recent imagery"}
            return state

        img = collection.median()
        ndvi = img.normalizedDifference(["B8", "B4"])

        # Calculate spatial statistics
        mean_ndvi = ndvi.reduceRegion(ee.Reducer.mean(), farm_area, 10).get('nd').getInfo()
        
        # Find low/high zones (simple thresholding)
        low_zone = ndvi.lt(mean_ndvi - 0.1)
        high_zone = ndvi.gt(mean_ndvi + 0.1)
        
        # Calculate areas
        pixel_area = ee.Image.pixelArea()
        low_area = low_zone.multiply(pixel_area).reduceRegion(
            ee.Reducer.sum(), farm_area, 10
        ).get('nd').getInfo()
        
        high_area = high_zone.multiply(pixel_area).reduceRegion(
            ee.Reducer.sum(), farm_area, 10
        ).get('nd').getInfo()
        
        total_area = float(3.14159 * radius**2)
        
        out = {
            "mean_ndvi": float(mean_ndvi or 0),
            "spatial_variability": {
                "low_vigor_area_ha": float(low_area or 0) / 10000,
                "high_vigor_area_ha": float(high_area or 0) / 10000,
                "low_vigor_percent": float(low_area or 0) / total_area * 100,
                "high_vigor_percent": float(high_area or 0) / total_area * 100
            },
            "uniformity_score": 1.0 - (float(low_area or 0) + float(high_area or 0)) / total_area
        }
        
        state["artifacts"]["spatial"] = out
        return state
    except Exception as e:
        state["artifacts"]["spatial"] = {"error": f"spatial analysis failed: {e}"}
        return state

# =========================
# WEATHER TOOLS
# =========================

def tool_weather_data(state) -> dict:
    """Get comprehensive weather data"""
    try:
        lat, lon = state["latitude"], state["longitude"]
        
        # Setup session with cache and retry
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # Current and forecast weather
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "et0_fao_evapotranspiration"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "et0_fao_evapotranspiration"],
            "past_days": 7,
            "forecast_days": 7
        }
        
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Current weather
        current = response.Current()
        current_data = {
            "temperature": current.Variables(0).Value(),
            "humidity": current.Variables(1).Value(),
            "precipitation": current.Variables(2).Value(),
            "wind_speed": current.Variables(3).Value(),
        }
        
        # Calculate VPD
        temp = current_data["temperature"]
        humidity = current_data["humidity"]
        svp = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        avp = svp * (humidity / 100)
        vpd = svp - avp
        current_data["vpd"] = round(vpd, 3)
        
        # Daily data for past week
        daily = response.Daily()
        daily_data = []
        for i in range(7):
            daily_data.append({
                "date": (datetime.now() - timedelta(days=7-i)).strftime("%Y-%m-%d"),
                "temp_max": daily.Variables(0).ValuesAsNumpy()[i],
                "temp_min": daily.Variables(1).ValuesAsNumpy()[i],
                "precipitation": daily.Variables(2).ValuesAsNumpy()[i],
                "et0": daily.Variables(3).ValuesAsNumpy()[i]
            })
        
        # Weather stress indicators
        stress_indicators = {
            "heat_stress": temp > 35,
            "drought_risk": sum([d["precipitation"] for d in daily_data[-7:]]) < 10,
            "high_vpd": vpd > 2.5,
            "excessive_rain": sum([d["precipitation"] for d in daily_data[-3:]]) > 50
        }
        
        out = {
            "current": current_data,
            "daily_history": daily_data,
            "stress_indicators": stress_indicators,
            "weekly_totals": {
                "precipitation": sum([d["precipitation"] for d in daily_data]),
                "avg_et0": np.mean([d["et0"] for d in daily_data if not np.isnan(d["et0"])])
            }
        }
        
        state["artifacts"]["weather"] = out
        return state
    except Exception as e:
        # Fallback to simulated data
        out = {
            "current": {
                "temperature": 25.0,
                "humidity": 60.0,
                "precipitation": 0.0,
                "wind_speed": 3.0,
                "vpd": 1.2
            },
            "error": f"weather API failed: {e}"
        }
        state["artifacts"]["weather"] = out
        return state

# =========================
# STRESS ANALYSIS TOOLS
# =========================

def tool_water_stress_advanced(state) -> dict:
    """Advanced water stress analysis using multiple indicators"""
    try:
        indices = state["artifacts"].get("compute_indices", {})
        weather = state["artifacts"].get("weather", {})
        
        if "error" in indices:
            state["artifacts"]["water_stress"] = {"error": "No indices available"}
            return state
        
        # Multiple water indicators
        ndwi_mean = indices.get("NDWI", {}).get("mean", 0)
        ndmi_mean = indices.get("NDMI", {}).get("mean", 0)
        ndvi_mean = indices.get("NDVI", {}).get("mean", 0.5)
        
        # Weather factors
        vpd = weather.get("current", {}).get("vpd", 1.0)
        recent_rain = weather.get("weekly_totals", {}).get("precipitation", 20)
        avg_et0 = weather.get("weekly_totals", {}).get("avg_et0", 4.0)
        
        # Calculate stress scores
        # NDWI stress (higher NDWI = less stress)
        ndwi_stress = max(0, min(1, 1 - (ndwi_mean + 0.2) / 0.6))
        
        # NDMI stress (similar to NDWI)
        ndmi_stress = max(0, min(1, 1 - (ndmi_mean + 0.2) / 0.6))
        
        # VPD stress
        vpd_stress = min(1, max(0, (vpd - 1.0) / 2.0))
        
        # Precipitation deficit stress
        rain_stress = max(0, min(1, (25 - recent_rain) / 25))
        
        # ET deficit (simplified)
        et_stress = max(0, min(1, (avg_et0 - 3.0) / 3.0))
        
        # Combined stress score
        combined_stress = (
            0.3 * ndwi_stress + 
            0.2 * ndmi_stress + 
            0.2 * vpd_stress + 
            0.2 * rain_stress + 
            0.1 * et_stress
        )
        
        # Determine severity
        if combined_stress > 0.7:
            severity = "critical"
        elif combined_stress > 0.5:
            severity = "high" 
        elif combined_stress > 0.3:
            severity = "moderate"
        else:
            severity = "low"
        
        out = {
            "combined_stress_score": round(combined_stress, 3),
            "severity": severity,
            "component_scores": {
                "ndwi_stress": round(ndwi_stress, 3),
                "ndmi_stress": round(ndmi_stress, 3),
                "vpd_stress": round(vpd_stress, 3),
                "precipitation_stress": round(rain_stress, 3),
                "et_stress": round(et_stress, 3)
            },
            "recommendations": []
        }
        
        # Add recommendations
        if combined_stress > 0.5:
            out["recommendations"].append("Consider immediate irrigation")
        if vpd_stress > 0.6:
            out["recommendations"].append("High VPD - irrigate early morning/evening")
        if rain_stress > 0.7:
            out["recommendations"].append("Precipitation deficit - monitor soil moisture")
            
        state["artifacts"]["water_stress"] = out
        return state
    except Exception as e:
        state["artifacts"]["water_stress"] = {"error": f"water stress analysis failed: {e}"}
        return state

def tool_nutrient_analysis(state) -> dict:
    """Comprehensive nutrient analysis"""
    try:
        indices = state["artifacts"].get("compute_indices", {})
        das = state["stage_das"]
        crop = state["crop"]
        
        if "error" in indices:
            state["artifacts"]["nutrient"] = {"error": "No indices available"}
            return state
        
        # Extract relevant indices
        ndre_mean = indices.get("NDRE", {}).get("mean", 0)
        ndvi_mean = indices.get("NDVI", {}).get("mean", 0)
        cire_mean = indices.get("CIre", {}).get("mean", 0)
        mcari_mean = indices.get("MCARI", {}).get("mean", 0)
        
        # Nitrogen assessment (primarily NDRE)
        if crop.lower() == "corn":
            if das < 30:
                n_threshold = 0.15
            elif das < 60:
                n_threshold = 0.25
            else:
                n_threshold = 0.20
        else:  # soybeans or other
            n_threshold = 0.18
        
        n_stress = max(0, (n_threshold - ndre_mean) / n_threshold)
        
        # Chlorophyll content (MCARI, CIre)
        chlorophyll_stress = 0
        if mcari_mean < 0.5:
            chlorophyll_stress += 0.3
        if cire_mean < 0.1:
            chlorophyll_stress += 0.3
        chlorophyll_stress = min(1, chlorophyll_stress)
        
        # Overall vegetation health (NDVI context)
        if ndvi_mean < 0.4:
            health_modifier = 1.2  # Increase stress if overall health is poor
        else:
            health_modifier = 1.0
        
        # Adjust for phenology
        if das < 20:
            confidence = 0.4  # Low confidence early season
            n_stress *= 0.7
        elif das > 80:
            confidence = 0.6  # Lower confidence late season
        else:
            confidence = 0.8
        
        final_n_stress = min(1, n_stress * health_modifier)
        
        # Determine severity
        if final_n_stress > 0.6:
            severity = "high"
        elif final_n_stress > 0.4:
            severity = "moderate"
        elif final_n_stress > 0.2:
            severity = "low"
        else:
            severity = "sufficient"
        
        out = {
            "nitrogen_stress_score": round(final_n_stress, 3),
            "chlorophyll_stress": round(chlorophyll_stress, 3),
            "severity": severity,
            "confidence": confidence,
            "indices_used": {
                "ndre": round(ndre_mean, 3),
                "mcari": round(mcari_mean, 3),
                "cire": round(cire_mean, 3)
            },
            "recommendations": []
        }
        
        # Add recommendations
        if final_n_stress > 0.5 and das < 70:
            if crop.lower() == "corn":
                out["recommendations"].append("Consider N application: 20-40 kg/ha")
            else:
                out["recommendations"].append("Monitor - soybeans fix N naturally")
        
        if chlorophyll_stress > 0.5:
            out["recommendations"].append("Tissue test recommended for micro-nutrients")
            
        if confidence < 0.5:
            out["recommendations"].append("Low confidence - validate with SPAD or tissue test")
        
        state["artifacts"]["nutrient"] = out
        return state
    except Exception as e:
        state["artifacts"]["nutrient"] = {"error": f"nutrient analysis failed: {e}"}
        return state

def tool_disease_risk_assessment(state) -> dict:
    """Disease risk assessment based on environmental conditions"""
    try:
        weather = state["artifacts"].get("weather", {})
        indices = state["artifacts"].get("compute_indices", {})
        crop = state["crop"]
        das = state["stage_das"]
        
        current = weather.get("current", {})
        daily_history = weather.get("daily_history", [])
        
        # Environmental risk factors
        temp = current.get("temperature", 20)
        humidity = current.get("humidity", 60)
        
        # Calculate leaf wetness proxy (humidity + recent rain)
        recent_rain = sum([d.get("precipitation", 0) for d in daily_history[-3:]])
        leaf_wetness_hours = humidity / 100 * 24 if humidity > 80 else 0
        
        # Disease-specific risk models
        risks = {}
        
        if crop.lower() == "corn":
            # Gray leaf spot risk
            if 25 <= temp <= 30 and humidity > 75:
                risks["gray_leaf_spot"] = min(1, 0.3 + (humidity - 75) / 100)
            else:
                risks["gray_leaf_spot"] = 0.1
            
            # Northern corn leaf blight
            if 18 <= temp <= 27 and leaf_wetness_hours > 6:
                risks["northern_leaf_blight"] = min(1, 0.2 + leaf_wetness_hours / 24)
            else:
                risks["northern_leaf_blight"] = 0.1
        
        elif crop.lower() == "soybeans":
            # Frogeye leaf spot
            if temp > 24 and humidity > 80:
                risks["frogeye_leaf_spot"] = min(1, 0.2 + (temp - 24) / 10)
            else:
                risks["frogeye_leaf_spot"] = 0.1
            
            # Brown spot
            if humidity > 85 and recent_rain > 10:
                risks["brown_spot"] = min(1, 0.3 + recent_rain / 50)
            else:
                risks["brown_spot"] = 0.1
        
        # General fungal risk
        fungal_risk = 0
        if humidity > 85 and temp > 20:
            fungal_risk = min(1, (humidity - 60) / 40 * (temp - 15) / 15)
        
        # NDVI drop indicator (potential disease stress)
        ndvi_mean = indices.get("NDVI", {}).get("mean", 0.6)
        expected_ndvi = 0.7 if das > 30 else 0.5
        
        health_decline = max(0, (expected_ndvi - ndvi_mean) / expected_ndvi)
        
        # Overall disease pressure
        max_specific_risk = max(risks.values()) if risks else 0
        overall_risk = min(1, max_specific_risk + 0.3 * fungal_risk + 0.2 * health_decline)
        
        if overall_risk > 0.7:
            risk_level = "high"
        elif overall_risk > 0.4:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        out = {
            "overall_disease_risk": round(overall_risk, 3),
            "risk_level": risk_level,
            "specific_diseases": {k: round(v, 3) for k, v in risks.items()},
            "environmental_factors": {
                "temperature": temp,
                "humidity": humidity,
                "recent_precipitation": recent_rain,
                "leaf_wetness_hours": leaf_wetness_hours
            },
            "recommendations": []
        }
        
        # Add recommendations
        if overall_risk > 0.6:
            out["recommendations"].append("High disease pressure - consider preventive fungicide")
            out["recommendations"].append("Increase scouting frequency")
        elif overall_risk > 0.4:
            out["recommendations"].append("Monitor closely for disease symptoms")
        
        if humidity > 85:
            out["recommendations"].append("High humidity - ensure good air circulation")
        
        state["artifacts"]["disease_risk"] = out
        return state
    except Exception as e:
        state["artifacts"]["disease_risk"] = {"error": f"disease risk assessment failed: {e}"}
        return state

# =========================
# PLACEHOLDER DETECTION TOOLS
# (Replace with actual ML models when available)
# =========================

def tool_pest_assess(state) -> dict:
    """Enhanced pest assessment with environmental factors"""
    try:
        # Use provided pest counts or estimate from environment
        context_counts = state["context"].get("pest_counts", [])
        weather = state["artifacts"].get("weather", {})
        
        # Environmental pest pressure estimation
        temp = weather.get("current", {}).get("temperature", 20)
        humidity = weather.get("current", {}).get("humidity", 60)
        
        # Simple pest pressure model
        pest_pressure = 0
        if 20 <= temp <= 30:
            pest_pressure += (temp - 15) / 15 * 0.4
        if humidity > 70:
            pest_pressure += (humidity - 50) / 50 * 0.3
        
        pest_pressure = min(1, pest_pressure)
        
        out = {
            "trap_counts": context_counts,
            "environmental_pressure": round(pest_pressure, 3),
            "risk_level": "high" if pest_pressure > 0.6 else "moderate" if pest_pressure > 0.3 else "low",
            "note": "Replace with actual trap monitoring/ML detection"
        }
        
        state["artifacts"]["pest_assess"] = out
        return state
    except Exception as e:
        state["artifacts"]["pest_assess"] = {"error": f"pest assessment failed: {e}"}
        return state

def tool_weed_detect(state) -> dict:
    """Enhanced weed detection placeholder"""
    try:
        weed_cover = state["context"].get("weed_cover_pct")
        indices = state["artifacts"].get("compute_indices", {})
        
        # Use spatial variability as proxy for weed pressure
        ndvi_std = indices.get("NDVI", {}).get("std", 0)
        
        # High variability might indicate weeds
        if weed_cover is None:
            weed_cover = min(25, ndvi_std * 100) if ndvi_std > 0.15 else 5
        
        severity = "high" if weed_cover > 20 else "moderate" if weed_cover > 10 else "low"
        
        out = {
            "weed_cover_percent": round(float(weed_cover), 1),
            "severity": severity,
            "spatial_variability": round(ndvi_std, 3),
            "mask_path": state["context"].get("weed_mask_path"),
            "note": "Replace with actual segmentation model"
        }
        
        state["artifacts"]["weed_detect"] = out
        return state
    except Exception as e:
        state["artifacts"]["weed_detect"] = {"error": f"weed detection failed: {e}"}
        return state

# =========================
# INTEGRATED ANALYSIS TOOLS
# =========================

def tool_yield_prediction(state) -> dict:
    """Simple yield prediction based on vegetation indices and stage"""
    try:
        indices = state["artifacts"].get("compute_indices", {})
        das = state["stage_das"]
        crop = state["crop"]
        
        ndvi_mean = indices.get("NDVI", {}).get("mean", 0.5)
        ndre_mean = indices.get("NDRE", {}).get("mean", 0.2)
        
        # Simple yield model (replace with actual trained model)
        if crop.lower() == "corn":
            base_yield = 10.0  # t/ha
            ndvi_factor = ndvi_mean / 0.7  # Normalize to good NDVI
            stage_factor = min(1.0, das / 80)  # Full potential after 80 DAS
            predicted_yield = base_yield * ndvi_factor * stage_factor
        else:  # soybeans
            base_yield = 3.5  # t/ha
            ndvi_factor = ndvi_mean / 0.7
            stage_factor = min(1.0, das / 70)
            predicted_yield = base_yield * ndvi_factor * stage_factor
        
        # Confidence based on stage
        if das < 30:
            confidence = 0.3
        elif das < 60:
            confidence = 0.6
        else:
            confidence = 0.8
        
        out = {
            "predicted_yield_t_ha": round(predicted_yield, 2),
            "confidence": confidence,
            "base_yield": base_yield,
            "yield_factors": {
                "vegetation_health": round(ndvi_factor, 3),
                "development_stage": round(stage_factor, 3)
            },
            "note": "Simple model - replace with calibrated ML model"
        }
        
        state["artifacts"]["yield_prediction"] = out
        return state
    except Exception as e:
        state["artifacts"]["yield_prediction"] = {"error": f"yield prediction failed: {e}"}
        return state

def tool_intervention_optimizer(state) -> dict:
    """Analyze all stress factors and prioritize interventions"""
    try:
        water_stress = state["artifacts"].get("water_stress", {})
        nutrient_stress = state["artifacts"].get("nutrient", {})
        disease_risk = state["artifacts"].get("disease_risk", {})
        weed_pressure = state["artifacts"].get("weed_detect", {})
        weather = state["artifacts"].get("weather", {})
        
        interventions = []
        
        # Water stress interventions
        water_score = water_stress.get("combined_stress_score", 0)
        if water_score > 0.6:
            interventions.append({
                "type": "irrigation",
                "priority": "critical",
                "urgency_days": 1,
                "estimated_cost_ha": 25,
                "expected_benefit": "Prevent yield loss",
                "details": "High water stress detected"
            })
        elif water_score > 0.4:
            interventions.append({
                "type": "irrigation",
                "priority": "high", 
                "urgency_days": 3,
                "estimated_cost_ha": 20,
                "expected_benefit": "Maintain plant health",
                "details": "Moderate water stress"
            })
        
        # Nutrient interventions
        n_score = nutrient_stress.get("nitrogen_stress_score", 0)
        n_confidence = nutrient_stress.get("confidence", 0)
        if n_score > 0.5 and n_confidence > 0.6:
            interventions.append({
                "type": "fertilizer",
                "priority": "high",
                "urgency_days": 7,
                "estimated_cost_ha": 45,
                "expected_benefit": "Improved yield potential",
                "details": f"Nitrogen deficiency detected (confidence: {n_confidence:.2f})"
            })
        
        # Disease management
        disease_score = disease_risk.get("overall_disease_risk", 0)
        if disease_score > 0.6:
            interventions.append({
                "type": "fungicide",
                "priority": "high",
                "urgency_days": 5,
                "estimated_cost_ha": 35,
                "expected_benefit": "Disease prevention",
                "details": "High disease pressure conditions"
            })
        
        # Weed management
        weed_cover = weed_pressure.get("weed_cover_percent", 0)
        if weed_cover > 15:
            interventions.append({
                "type": "herbicide",
                "priority": "medium",
                "urgency_days": 10,
                "estimated_cost_ha": 30,
                "expected_benefit": "Reduce competition",
                "details": f"{weed_cover}% weed coverage"
            })
        
        # Sort by priority and urgency
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        interventions.sort(key=lambda x: (priority_order[x["priority"]], x["urgency_days"]))
        
        # Calculate total costs and ROI
        total_cost = sum([i["estimated_cost_ha"] for i in interventions])
        
        out = {
            "recommended_interventions": interventions,
            "total_estimated_cost_ha": total_cost,
            "intervention_count": len(interventions),
            "most_urgent": interventions[0] if interventions else None,
            "summary": f"{len(interventions)} interventions recommended" if interventions else "No immediate interventions needed"
        }
        
        state["artifacts"]["intervention_plan"] = out
        return state
    except Exception as e:
        state["artifacts"]["intervention_plan"] = {"error": f"intervention optimization failed: {e}"}
        return state

# =========================
# TOOL REGISTRY
# =========================

AVAILABLE_TOOLS = {
    # Core satellite analysis
    "compute_indices": tool_compute_indices,
    "time_series": tool_time_series_analysis,
    "spatial_analysis": tool_spatial_analysis,
    
    # Weather and environment
    "weather_data": tool_weather_data,
    
    # Stress analysis
    "water_stress": tool_water_stress_advanced,
    "nutrient_analysis": tool_nutrient_analysis,
    "disease_risk": tool_disease_risk_assessment,
    
    # Detection (placeholders for ML models)
    "pest_assess": tool_pest_assess,
    "weed_detect": tool_weed_detect,
    
    # Integrated analysis
    "yield_prediction": tool_yield_prediction,
    "intervention_optimizer": tool_intervention_optimizer,
}

def get_tool_info():
    """Return information about available tools"""
    return {
        name: {
            "category": func.__doc__.split('\n')[0] if func.__doc__ else "No description",
            "function": func
        }
        for name, func in AVAILABLE_TOOLS.items()
    }

def run_tool(tool_name: str, state: dict) -> dict:
    """Execute a tool by name"""
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Tool '{tool_name}' not found. Available: {list(AVAILABLE_TOOLS.keys())}")
    
    return AVAILABLE_TOOLS[tool_name](state)