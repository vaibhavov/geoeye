import sys
import os
sys.path.append('src/agents')

import ee
from toolbox import run_tool

# Initialize Earth Engine (from your original code)
def init_ee():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if credentials_path and project_id:
            credentials = ee.ServiceAccountCredentials(None, credentials_path)
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize()
        print("✅ Earth Engine initialized")
    except Exception as e:
        print(f"❌ EE init failed: {e}")

# Initialize EE first
init_ee()

# Mock state
state = {
    "field_id": "test",
    "latitude": 41.878003,
    "longitude": -93.097702, 
    "radius_m": 1000,
    "crop": "corn",
    "stage_das": 45,
    "context": {},
    "artifacts": {}
}

# Test a tool
result = run_tool("compute_indices", state)
print(result["artifacts"])