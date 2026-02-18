
import os
import sys

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from providers.brave_search import BraveSearchProvider
from config.settings import BRAVE_SEARCH_API_KEY

def test_brave_search():
    print("Testing Brave Search...")
    
    if not BRAVE_SEARCH_API_KEY:
        print("❌ BRAVE_SEARCH_API_KEY environment variable is NOT set.")
        return

    try:
        provider = BraveSearchProvider()
        results = provider.search("python latest version", count=1)
        
        if results:
            print(f"✅ Brave Search successful! Found {len(results)} results.")
            print(f"   First result: {results[0]['title']} ({results[0]['url']})")
        else:
            print("⚠️ Brave Search returned no results (but no error).")
            
    except Exception as e:
        print(f"❌ Brave Search failed: {e}")

if __name__ == "__main__":
    test_brave_search()
