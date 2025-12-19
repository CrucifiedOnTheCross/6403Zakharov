import os
import time
import json
from typing import List, Dict, Optional

try:
    import requests
except Exception:
    requests = None

from io import BytesIO

from .cat_image import CatImage, ColorCatImage
from logging_config import logger


def _read_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _log_time(fn):
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        logger.debug(f"START {fn.__name__}")
        res = fn(self, *args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"END {fn.__name__} took {end - start:.3f}s")
        return res
    return wrapper


class PositiveInt:
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    def __get__(self, instance, owner):
        return getattr(instance, self.name)
    def __set__(self, instance, value):
        iv = int(value)
        if iv <= 0:
            raise ValueError("limit must be positive")
        setattr(instance, self.name, iv)


class CatImageProcessor:
    limit = PositiveInt()

    def __init__(self, provider: str = "cat", output_dir: str = "downloads", limit: int = 1):
        self.provider = provider
        self.output_dir = output_dir
        self.limit = limit

    def _api_key(self) -> str:
        env = _read_env()
        if self.provider == "cat":
            return env.get("CAT_API_KEY") or os.environ.get("CAT_API_KEY") or "DEMO-API-KEY"
        return env.get("DOG_API_KEY") or os.environ.get("DOG_API_KEY") or ""

    def _base_url(self) -> str:
        if self.provider == "cat":
            return "https://api.thecatapi.com/v1/images/search"
        return "https://api.thedogapi.com/v1/images/search"

    @_log_time
    def fetch_images(self) -> List[CatImage]:
        url = self._base_url()
        params = {
            "size": "med",
            "mime_types": "jpg,png",
            "format": "json",
            "has_breeds": "true",
            "order": "RANDOM",
            "page": "0",
            "limit": str(self.limit),
        }
        headers = {"x-api-key": self._api_key()}
        data: List[Dict] = []
        if requests is not None:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        else:
            import urllib.request
            import urllib.parse
            q = urllib.parse.urlencode(params)
            req = urllib.request.Request(url + "?" + q, headers=headers)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        images: List[CatImage] = []
        for item in data:
            breeds = item.get("breeds") or []
            if not breeds:
                continue
            breed_name = breeds[0].get("name") or "unknown"
            img_url = item.get("url")
            if not img_url:
                continue
            if requests is not None:
                ir = requests.get(img_url, timeout=60)
                ir.raise_for_status()
                content = ir.content
            else:
                import urllib.request
                with urllib.request.urlopen(img_url) as resp:
                    content = resp.read()
            cat = CatImage.from_bytes(content, img_url, breed_name)
            images.append(cat)
            logger.debug(f"Fetched {breed_name} {img_url}")
        return images

    def _sanitize(self, breed: str) -> str:
        s = breed.lower()
        s = s.replace(" ", "_")
        s = s.replace("/", "_")
        return s

    @_log_time
    def process_and_save(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        images = self.fetch_images()
        for idx, img in enumerate(images, start=1):
            b = self._sanitize(img.breed)
            base = os.path.join(self.output_dir, f"{idx}_{b}")
            orig = base + "_original.png"
            libp = base + "_edges_lib.png"
            custp = base + "_edges_custom.png"
            img.save_original(orig)
            lib = img.edge_detect_library()
            cust = img.edge_detect_custom()
            import cv2
            cv2.imwrite(libp, lib)
            cv2.imwrite(custp, cust)
            logger.debug(f"Saved {orig}, {libp}, {custp}")
