import logging, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.settings import settings, AppSettings, TMDBSettings

settings: AppSettings = AppSettings()

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
        self.tmdb: TMDBSettings = settings.tmdb

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
            "Authorization": f"Bearer {self.tmdb.api_key}",
            "Accept": "application/json",
        })

    #TODO: Check response content-type
    def _handle_response(self, resp: requests.Response):
        """
        Raise HTTPError on bad status and parse JSON response.
        """
        resp.raise_for_status()       # Throw on 4xx/5xx
        return resp.json()            # Parse JSON or error
    
    def get(self, path: str, params=None):
        """
        Perform a GET request to the TMDb API, returning parsed JSON.
        """
        api_base_url: str = self.tmdb.api_base_url
        url = f"{api_base_url.rstrip('/')}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=10)
        return self._handle_response(resp)

