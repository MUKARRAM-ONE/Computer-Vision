from pathlib import Path
import os
import sys
from dotenv import load_dotenv

# Resolve relative to this script's location so the test works from any CWD
base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent

# Prefer project root .env, then src/.env
env_path = project_root / '.env'
alt = project_root / 'src' / '.env'
if not env_path.exists() and alt.exists():
    env_path = alt

print('Python executable:', sys.executable)
print('Script base dir:', base_dir)
print('Using env file:', env_path)

if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    load_dotenv()  # fallback to default search

# print the paths with all the values in the .env file
print('CAMERA_RESIZE_WIDTH =', os.getenv('CAMERA_RESIZE_WIDTH'))
print('FRAME_DELAY =', os.getenv('FRAME_DELAY'))
