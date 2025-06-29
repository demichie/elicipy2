import subprocess
import pkg_resources


def run_streamlit():
    """Launch the Streamlit app from the installed package."""
    app_path = pkg_resources.resource_filename("elicipy", "dashboard_app.py")
    subprocess.run(["streamlit", "run", app_path], check=True)


if __name__ == "__main__":
    run_streamlit()
