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


def _live(name: str):
    """Look the handler up on every request, so it uses the same previews module
    as the page even after Streamlit reloads that module when its file is edited."""
    async def endpoint(request):
        return await getattr(importlib.import_module("previews"), name)(request)
    return endpoint


app = st.App(
    "app.py",
    routes=[
        Route("/mvk/preview/{ds}/{view:int}/{frame:int}", _live("preview_endpoint")),
        Route("/mvk/warm/{ds}", _live("warm_endpoint")),
    ],
)
