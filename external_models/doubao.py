import concurrent.futures
import time
import threading
from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI

MAX_CONCURRENT_EMBEDDINGS = 4
REQUEST_DELAY_SEC = 0.5


class DoubaoMultimodalEmbeddings(Embeddings):
    """Custom LangChain embedding class for Doubao Multimodal Vision Endpoint."""

    def __init__(
        self, api_key: str, base_url: str, endpoint_id: str, dimensions: int = 2048
    ):

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.endpoint_id = endpoint_id
        self.dimensions = dimensions

        # Thread-safe rate limiter state
        self._rate_limit_lock = threading.Lock()
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self) -> None:
        """
        Thread-safe method to ensure interval pass between API calls.
        """
        with self._rate_limit_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < REQUEST_DELAY_SEC:
                time.sleep(REQUEST_DELAY_SEC - elapsed)
            # Update the last request time right before we release the lock
            self._last_request_time = time.time()

    def _embed_single(self, text: str) -> List[float]:
        """Matches exactly the payload format of your successful script."""

        # 1. Wait for the rate limit before making the HTTP request
        self._wait_for_rate_limit()

        # 2. Fire the request
        response = self.client.embeddings.create(
            model=self.endpoint_id,
            input=[text],
            # We omit `dimensions` and `encoding_format` to prevent 400 Bad Request
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Initialize an empty list of the correct size to maintain document order
        embeddings = [None] * len(texts)

        # Use ThreadPoolExecutor for clean, concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_EMBEDDINGS
        ) as executor:
            # Map futures to their original index
            futures = {
                executor.submit(self._embed_single, text): i
                for i, text in enumerate(texts)
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as e:
                    print(f"Embedding failed for chunk {idx}: {e}")
                    raise e

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)
