import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Test 1: Check Google API package
print("Testing Google API import...")
try:
    import google.generativeai as genai
    print("✅ Google Generative AI imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run: pip install google-generativeai")
    exit()

# Test 2: Check if API key loaded
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("❌ API key not found in .env file!")
    print("Make sure your .env file has: GOOGLE_API_KEY=your_key_here")
    exit()
else:
    print(f"✅ API key loaded! (starts with: {api_key[:10]}...)")

# Test 3: Test API connection
print("\nTesting Google API connection...")
try:
    genai.configure(api_key=api_key)
    # Use a model that actually exists!
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = model.generate_content("Say hello in one word!")
    print(f"✅ API works! Gemini 2.5 Flash responded: '{response.text}'")
except Exception as e:
    print(f"❌ API test failed: {e}")
    exit()

print("\n🎉 GOOGLE API SETUP COMPLETE!")
print("We're using: Gemini 2.5 Flash (fast and free!)")
print("\nNext: Let's build the RAG system!")