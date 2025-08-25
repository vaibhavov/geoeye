#!/usr/bin/env python3
"""
Simple script to debug .env file loading
Run this first to verify your environment setup
"""

import os
from pathlib import Path
from dotenv import load_dotenv

print("🔍 Environment Debugging Script")
print("=" * 50)

# Check current directory
current_dir = Path.cwd()
print(f"Current directory: {current_dir}")

# Check for .env file
env_file = current_dir / ".env"
print(f"\n📁 Checking .env file:")
print(f"   Path: {env_file}")
print(f"   Exists: {env_file.exists()}")

if env_file.exists():
    print(f"   Size: {env_file.stat().st_size} bytes")
    
    # Read and display content (hiding sensitive parts)
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        print(f"   Lines: {len(lines)}")
        print(f"\n📄 .env file content preview:")
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Hide sensitive values
                    if 'KEY' in key or 'SECRET' in key or 'TOKEN' in key:
                        display_value = f"{value[:8]}..." if len(value) > 8 else "[hidden]"
                    else:
                        display_value = value
                    print(f"   Line {i}: {key}={display_value}")
                else:
                    print(f"   Line {i}: {line}")
            elif line.startswith('#'):
                print(f"   Line {i}: [comment]")
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")

# Test loading
print(f"\n🔧 Testing environment loading:")

# Before loading
api_key_before = os.getenv("OPENWEATHER_API_KEY")
print(f"   Before load_dotenv(): {'Found' if api_key_before else 'Not found'}")

# Load environment
load_result = load_dotenv(dotenv_path=env_file, override=True)
print(f"   load_dotenv() result: {load_result}")

# After loading
api_key_after = os.getenv("OPENWEATHER_API_KEY")
print(f"   After load_dotenv(): {'Found' if api_key_after else 'Not found'}")

if api_key_after:
    print(f"   API key (first 10 chars): {api_key_after[:10]}...")
    print(f"   API key length: {len(api_key_after)}")
else:
    print("   ❌ API key still not found!")

# List all environment variables with relevant prefixes
print(f"\n📋 All relevant environment variables:")
relevant_prefixes = ['OPENWEATHER', 'GOOGLE', 'DATABASE', 'FLASK']
for key, value in os.environ.items():
    if any(key.startswith(prefix) for prefix in relevant_prefixes):
        if 'KEY' in key or 'SECRET' in key or 'TOKEN' in key:
            display_value = f"{value[:8]}..." if len(value) > 8 else "[hidden]"
        else:
            display_value = value
        print(f"   {key}={display_value}")

print(f"\n✅ Environment debugging complete!")