"""
SERP API providers for LinkedIn profile search.

Supports SerpAPI and Serper providers with consistent interface.
"""
import json
from typing import Dict, List

try:
    import requests
except ImportError:
    requests = None


class SearchProvider:
    """Base class for search providers."""

    def search(self, query: str, num: int = 5, hl: str = "en", gl: str = "us") -> List[Dict[str, str]]:
        """Return list of {link, title, snippet} items."""
        raise NotImplementedError


class SerpAPIProvider(SearchProvider):
    """SerpAPI.com provider implementation."""

    BASE = "https://serpapi.com/search"

    def __init__(self, api_key: str):
        if requests is None:
            raise RuntimeError("requests module not available. Install with: pip install requests")
        if not api_key:
            raise RuntimeError("SERPAPI_API_KEY missing.")
        self.api_key = api_key

    def search(self, query: str, num: int = 5, hl: str = "en", gl: str = "us") -> List[Dict[str, str]]:
        params = {
            "engine": "google",
            "q": query,
            "num": num,
            "hl": hl,
            "gl": gl,
            "api_key": self.api_key,
            # Keep payload small; see SerpAPI json_restrictor docs
            "json_restrictor": "organic_results[].{link,title,snippet}"
        }
        resp = requests.get(self.BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in (data.get("organic_results") or []):
            link = item.get("link")
            title = item.get("title")
            snippet = item.get("snippet")
            if link and title:
                results.append({"link": link, "title": title, "snippet": snippet or ""})
        return results


class SerperProvider(SearchProvider):
    """Serper.dev provider implementation."""

    URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str):
        if requests is None:
            raise RuntimeError("requests module not available. Install with: pip install requests")
        if not api_key:
            raise RuntimeError("SERPER_API_KEY missing.")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        })

    def search(self, query: str, num: int = 5, hl: str = "en", gl: str = "us") -> List[Dict[str, str]]:
        payload = {"q": query, "num": num, "hl": hl, "gl": gl}
        resp = self.session.post(self.URL, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in (data.get("organic") or []):
            link = item.get("link")
            title = item.get("title")
            snippet = item.get("snippet", "")
            if link and title:
                results.append({"link": link, "title": title, "snippet": snippet})
        return results