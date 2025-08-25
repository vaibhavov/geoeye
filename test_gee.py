import os
import ee
from dotenv import load_dotenv

load_dotenv()

def test_earth_engine():
    """Test Earth Engine initialization with service account"""
    try:
        # Get credentials from environment
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        print(f"Using credentials: {credentials_path}")
        print(f"Using project: {project_id}")
        
        # Initialize Earth Engine with service account
        credentials = ee.ServiceAccountCredentials(None, credentials_path)
        ee.Initialize(credentials, project=project_id)
        
        print("✅ Google Earth Engine initialized successfully!")
        
        # Test basic functionality
        collection = ee.ImageCollection('COPERNICUS/S2_SR')
        count = collection.limit(1).size()
        print(f"✅ Sentinel-2 collection accessible: {count.getInfo()} images")
        
        # Test with a simple image
        image = ee.Image('COPERNICUS/S2_SR/20210109T185751_20210109T185931_T10SEG')
        info = image.getInfo()
        print(f"✅ Sample image loaded: {info['type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Earth Engine error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check service account JSON file path")
        print("   2. Ensure Earth Engine API is enabled")
        print("   3. Verify service account has Earth Engine permissions")
        return False

if __name__ == "__main__":
    print("🛰️ Testing Google Earth Engine with Service Account")
    print("=" * 50)
    test_earth_engine()