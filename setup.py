from setuptools import setup, find_packages

setup(
    name="elicipy",
    version="2.0.0",
    packages=find_packages(),
    include_package_data=True,  # ← ADD THIS LINE
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "streamlit",
        "click",
        "altair",
        "cryptography",
        "python-dotenv",
        "bcrypt",
        "streamlit-authenticator",
        "python-pptx",
        "pygithub",
        "plotly"
    ],
    entry_points={
        "console_scripts": [
            "elicipy=elicipy.cli:main",
            "elicipy-form=elicipy.app:run_streamlit",
            "elicipy-dashboard=elicipy.app2:run_streamlit"
        ]
    },
    author="Mattia de' Michieli Vitturi",
    author_email="demichie@gmail.com",
    description="A Python package for elicitation and statistical modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/demichie/elicipy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-2.0 license",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
