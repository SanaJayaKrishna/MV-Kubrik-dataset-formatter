"""Entry point: the Streamlit page (app.py) plus the routes that serve preview images.

Start it with ./run.sh (which runs `streamlit run serve.py`). Streamlit sees the
`app` object below and serves app.py together with the extra routes.
"""

import importlib
import os
import sys

import streamlit as st
from starlette.routing import Route

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MVK_PREVIEW_ROUTES"] = "1"   # lets app.py check that the image routes exist


def _live(module: str, name: str):
    """Look the handler up on every request, so it uses the same module objects as the
    page even after Streamlit reloads a module when its file is edited."""
    async def endpoint(request):
        return await getattr(importlib.import_module(module), name)(request)
    return endpoint


app = st.App(
    "app.py",
    routes=[
        Route("/mvk/preview/{ds}/{view:int}/{frame:int}", _live("previews", "preview_endpoint")),
        Route("/mvk/warm/{ds}", _live("previews", "warm_endpoint")),
        Route("/mvk/gt/{ds}/cameras", _live("groundtruth", "cameras_endpoint")),
        Route("/mvk/gt/{ds}/object/{index:int}", _live("groundtruth", "object_endpoint")),
    ],
)
