import os
import pkg_resources

def run_streamlit():
    """Launch the Streamlit app from the installed package."""
    app_path = pkg_resources.resource_filename("elicipy","streamlit_app.py")
    os.system(f"streamlit run {app_path}")

if __name__ == "__main__":
    run_streamlit()
