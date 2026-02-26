from setuptools import setup, find_packages

setup(
    name="bonza-mragent",
    version="0.2.4",
    description="A lightweight, open-source AI Agent powered by free APIs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Bonza Insights",
    author_email="hello@achbj.com",
    url="https://github.com/bonzainsights/MRAgent",
    packages=find_packages(),
    py_modules=["main"],
    install_requires=[
        "openai>=1.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "nvidia-riva-client>=2.14.0",
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "rich>=13.0.0",
        "prompt-toolkit>=3.0.0",
        "Pillow>=10.0.0",
        "pyautogui>=0.9.54",
        "flask>=3.0.0",
        "beautifulsoup4>=4.12.0",
        "python-telegram-bot>=20.0",
        "edge-tts>=7.0.0",
        "croniter>=3.0.0",
        "schedule>=1.2.0",
        "PyPDF2>=3.0.0"
    ],
    entry_points={
        "console_scripts": [
            "mragent=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
