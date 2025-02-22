import requests
import json
import time
from typing import Optional, Dict, Any
from config import config

class ApiClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": config.API_KEY,
            "Content-Type": "application/json"
        })
        
    def process_response(self, user_message: str) -> Optional[str]:
        """Process user message with retry logic and improved error handling."""
        for attempt in range(config.API_MAX_RETRIES):
            try:
                payload = {"user_message": user_message}
                
                # Send request with timeout
                response = self.session.post(
                    config.API_URL,
                    json=payload,
                    timeout=config.API_TIMEOUT
                )
                
                # Check for specific error types
                if response.status_code == 401:
                    print("❌ API authentication failed. Check your API key.")
                    return None
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    print(f"⚠️ Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                # Raise for other error status codes
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                if not data or 'synthia_response' not in data:
                    print("⚠️ Invalid response format from API")
                    return None
                
                return data['synthia_response']
                
            except requests.exceptions.Timeout:
                print(f"⚠️ Request timeout (attempt {attempt + 1}/{config.API_MAX_RETRIES})")
            except requests.exceptions.ConnectionError:
                print(f"⚠️ Connection error (attempt {attempt + 1}/{config.API_MAX_RETRIES})")
            except requests.exceptions.RequestException as e:
                print(f"❌ API Error: {str(e)}")
                return None
            
            # Wait before retry
            if attempt < config.API_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        print("❌ API request failed after all retries")
        return None
    
    def save_conversation(self, user_input: str, ai_response: str) -> None:
        """Save conversation history with error handling."""
        try:
            history = self.load_conversation_history()
            history["history"].append({
                "user": user_input,
                "synthia": ai_response,
                "timestamp": time.time()
            })
            
            # Keep only last 100 conversations
            if len(history["history"]) > 100:
                history["history"] = history["history"][-100:]
            
            with open(config.MEMORY_FILE, "w") as file:
                json.dump(history, file, indent=4)
                
        except Exception as e:
            print(f"⚠️ Error saving conversation: {str(e)}")
    
    def load_conversation_history(self) -> Dict[str, Any]:
        """Load conversation history with initialization."""
        try:
            with open(config.MEMORY_FILE, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {"history": []}
        except json.JSONDecodeError:
            print("⚠️ Invalid memory file format, creating new")
            return {"history": []}
        except Exception as e:
            print(f"⚠️ Error loading conversation history: {str(e)}")
            return {"history": []}