import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure we're loading the .env file properly
def setup_environment():
    """Load environment with proper debugging"""
    print("🔧 Setting up environment...")
    
    # Get current directory
    current_dir = Path.cwd()
    env_path = current_dir / ".env"
    
    print(f"   Current directory: {current_dir}")
    print(f"   Looking for .env at: {env_path}")
    print(f"   .env exists: {env_path.exists()}")
    
    if env_path.exists():
        # Load the environment file
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"   ✅ Loaded .env file")
        
        # Check if the key is loaded
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if api_key:
            print(f"   ✅ API key loaded (first 10 chars): {api_key[:10]}...")
        else:
            print(f"   ❌ API key not found in environment")
            
        # List all env vars that start with relevant prefixes
        relevant_vars = [k for k in os.environ.keys() if k.startswith(('OPENWEATHER', 'GOOGLE', 'DATABASE'))]
        print(f"   📋 Relevant environment variables found: {relevant_vars}")
        
    else:
        print(f"   ❌ .env file not found at {env_path}")
        
        # Try to find .env in parent directories
        for parent in current_dir.parents:
            parent_env = parent / ".env"
            if parent_env.exists():
                print(f"   📁 Found .env at: {parent_env}")
                load_dotenv(dotenv_path=parent_env, override=True)
                break
    
    return os.getenv("OPENWEATHER_API_KEY") is not None

# Setup environment first
env_setup_success = setup_environment()

if not env_setup_success:
    print("\n❌ Environment setup failed. Please check your .env file.")
    print("Make sure your .env file contains:")
    print("OPENWEATHER_API_KEY=your_api_key_here")
    sys.exit(1)

# Add src/agents to path
sys.path.append('src/agents')

try:
    from data_agent import collect_field_data
    print("✅ Successfully imported data_agent")
except ImportError as e:
    print(f"❌ Failed to import data_agent: {e}")
    sys.exit(1)

# Test data collection
print("\n🚀 Starting data collection test...")
try:
    result = collect_field_data(
        field_id="iowa_test",
        lat=41.878003,
        lon=-93.097702,
        radius_m=1000,
        data_types=['satellite', 'weather']
    )

    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Data sources: {list(result.data.keys())}")
    print(f"   Timestamp: {result.timestamp}")

    if 'satellite' in result.data:
        sat = result.data['satellite']
        if 'error' in sat:
            print(f"\n🛰️ Satellite Error: {sat['error']}")
        else:
            ndvi_mean = sat.get('indices', {}).get('NDVI', {}).get('mean', 'N/A')
            print(f"\n🛰️ Satellite: {sat.get('image_count')} images, NDVI: {ndvi_mean}")

    if 'weather' in result.data:
        weather = result.data['weather']
        if 'error' in weather:
            print(f"\n🌤️ Weather Error: {weather['error']}")
        elif 'current' in weather:
            current = weather['current']
            print(f"\n🌤️ Weather: {current['temperature']}°C, {current['humidity']}% humidity")

    if result.errors:
        print(f"\n❌ Errors encountered:")
        for error in result.errors:
            print(f"   - {error}")
    else:
        print(f"\n✅ All data collection successful!")

except Exception as e:
    print(f"\n💥 Exception during data collection: {e}")
    import traceback
    traceback.print_exc()