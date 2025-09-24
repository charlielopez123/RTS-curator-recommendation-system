import logging, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cts_recommender.settings import get_settings, TMDBSettings
from typing import Any
import logging

cfg = get_settings()
logger = logging.getLogger(__name__)

class TMDB_APIClient:
    def __init__(self,
                total_retries: int = 3,
                backoff_factor: float = 1.0,
                status_forcelist: tuple = (429, 500, 502, 503, 504)):
        """
        Initializes a requests.Session with:
            - Authorization header
            - JSON accept header
            - HTTPAdapter for retries on connection errors and specified HTTP status codes
        """
        self.session = requests.Session()
        self.tmdb: TMDBSettings = cfg.tmdb

        # Configure retries
        retry_strategy = Retry(
            total=total_retries,
            connect=total_retries,
            read=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        # TODO: Check url scheme
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.tmdb.api_key.get_secret_value()}",
            "Accept": "application/json",
        })

    #TODO: Check response content-type
    def _handle_response(self, resp: requests.Response) -> Any:
        """
            Handle API response with proper error checking and JSON parsing.
            
            Args:
                resp: HTTP response object
                
            Returns:
                Parsed JSON data
                
            Raises:
                requests.HTTPError: For 4xx/5xx HTTP status codes
                ValueError: If response is not valid JSON
            """
        try:
        # Check for HTTP errors (4xx, 5xx)
            resp.raise_for_status()
            
            # Check if response has content
            if not resp.content:
                logger.warning(f"Empty response received for {resp.url}")
                return {}
            
            # Parse JSON response
            return resp.json()
            
        except requests.HTTPError as e:
            # Log the error with response details for debugging
            logger.error(f"HTTP {resp.status_code} error for {resp.url}: {resp.text}")
            raise
            
        except ValueError as e:
            # JSON decode error
            logger.error(f"Invalid JSON response from {resp.url}: {resp.text[:200]}...")
            raise ValueError(f"Invalid JSON response: {e}")
            
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error processing response from {resp.url}: {e}")
            raise
    
    def get(self, path: str, params=None) -> Any:
        """
        Perform a GET request to the TMDb API, returning parsed JSON.
        """
        api_base_url: str = str(self.tmdb.api_base_url)
        url = f"{api_base_url.rstrip('/')}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=10)
        return self._handle_response(resp)


