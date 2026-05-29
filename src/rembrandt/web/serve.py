"""CLI entry point for the Rembrandt configurator web server."""

from __future__ import annotations

import webbrowser

import uvicorn

from rembrandt.web.app import create_app


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Run the bpy-free FastAPI app with uvicorn.

    Args:
        host: Bind address for uvicorn.
        port: Bind port for uvicorn.
        open_browser: If True, open the app URL in the default browser before serving.
    """
    url = server_url(host=host, port=port)
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(), host=host, port=port)


def server_url(*, host: str, port: int) -> str:
    """Build the URL shown to users (maps ``0.0.0.0`` to ``127.0.0.1``)."""
    browse_host = "127.0.0.1" if host == "0.0.0.0" else host
    return f"http://{browse_host}:{port}/"


def main() -> None:
    """Console script entry point for ``rembrandt-serve``."""
    serve()


if __name__ == "__main__":
    main()
