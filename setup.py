from setuptools import setup, find_packages

setup(
    name="zakharov_6403",
    version="0.1.0",
    author="Nikita Zakharov",
    author_email="nk.z4kharov@gmail.com",
    description="Cat Image Processing API - Group 6403",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/CrucifiedOnTheCross/6403Zakharov",
    packages=find_packages(),
    py_modules=["logging_config"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "opencv-python",
        "requests",
        "aiohttp",
        "aiofiles",
        "numba",
    ],
)
