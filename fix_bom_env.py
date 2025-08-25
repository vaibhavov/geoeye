#!/usr/bin/env python3
"""
Fix the BOM issue in .env file
"""

from pathlib import Path

def fix_bom_env_file():
    """Remove BOM and recreate clean .env file"""
    env_file = Path(".env")
    
    print("🔧 Fixing BOM issue in .env file...")
    
    # Backup the old file
    if env_file.exists():
        backup_file = Path(".env.backup")
        with open(env_file, 'rb') as src:
            content = src.read()
        with open(backup_file, 'wb') as dst:
            dst.write(content)
        print(f"💾 Backed up original to {backup_file}")
    
    # Create clean content without BOM
    clean_content = """OPENWEATHER_API_KEY=dc16cac058093397dd5b505a170400bc
GOOGLE_APPLICATION_CREDENTIALS=D:\\IIT B\\MTP 1\\GeoEye\\geoeye-76fde1080762.json
DATABASE_URL=sqlite:///geoeye.db
FLASK_ENV=development
FLASK_DEBUG=True
DEFAULT_REGION=midwest_us
LOG_LEVEL=INFO
GOOGLE_CLOUD_PROJECT=geoeye"""
    
    # Write clean file without BOM
    with open(env_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(clean_content)
    
    print("✅ Created clean .env file without BOM")
    
    # Test the fix
    import os
    from dotenv import load_dotenv
    
    # Clear environment
    if 'OPENWEATHER_API_KEY' in os.environ:
        del os.environ['OPENWEATHER_API_KEY']
    
    # Load and test
    load_dotenv(override=True)
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if api_key:
        print(f"🎉 SUCCESS! API key loaded: {api_key[:10]}...")
        return True
    else:
        print("❌ Still not working...")
        return False

if __name__ == "__main__":
    success = fix_bom_env_file()
    if success:
        print("\n✅ Your .env file is now fixed!")
        print("You can run: python test_data_agent.py")
    else:
        print("\n❌ Still having issues. Please check manually.")