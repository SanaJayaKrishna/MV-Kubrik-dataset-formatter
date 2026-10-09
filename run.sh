#!/usr/bin/env bash
# Start the MV-Kubric dataset app with the ~/.streamlit_app environment,
# then open http://localhost:8501 in a browser.
# Extra arguments go to streamlit, for example:  ./run.sh --server.port 8600
cd "$(dirname "$(readlink -f "$0")")" || exit 1
exec "$HOME/.streamlit_app/bin/streamlit" run serve.py "$@"
