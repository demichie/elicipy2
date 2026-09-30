import subprocess
import sys
from importlib.resources import as_file, files


def run_streamlit():
    """Launch the Streamlit elicitation form from the installed package."""
    app_resource = files("elicipy").joinpath("streamlit_app.py")

    with as_file(app_resource) as app_path:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True,
        )


if __name__ == "__main__":
    run_streamlit()
