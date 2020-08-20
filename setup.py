try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name="smart-news-query-embeddings_vkesav", # Replace with your own username
    version="0.0.1",
    author="Kesav Viswanadha",
    author_email="vkesav@google.com",
    description="Model training for Smart News Query Embeddings project",
    long_description="",
    url="https://github.com/googleinterns/smart-news-query-embeddings",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
