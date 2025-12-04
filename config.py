from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
API_KEY = os.getenv("API_KEY")
STORAGE_DIR = "/media/discoexterno/leon"