import uvicorn
from pyngrok import ngrok
import sys

# Optional: Insert your ngrok authtoken here if you haven't configured it globally.
# You can get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
# ngrok.set_auth_token("YOUR_AUTHTOKEN")

if __name__ == "__main__":
    port = 8000
    
    try:
        # Open a ngrok tunnel to the localhost server
        public_url = ngrok.connect(port).public_url
        print("="*60)
        print(f"🚀 Your FastAPI app is running on ngrok at:")
        print(f"👉 {public_url}")
        print("="*60)
    except Exception as e:
        print(f"Failed to start ngrok tunnel: {e}")
        print("Please make sure you have installed pyngrok: 'pip install pyngrok' and configured your authtoken.")
        sys.exit(1)

    # Start the FastAPI server using Uvicorn
    # 'app' refers to app.py and 'app' refers to the FastAPI instance inside it
    uvicorn.run("app:app", host="127.0.0.1", port=port, log_level="info")
