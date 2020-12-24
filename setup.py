import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastpredict",
    version="0.0.1",
    author="Kapil Yedidi",
    author_email="kapily.code@gmail.com",
    description="A few helper utilities to make it simpler to perform fastai inference/prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kapily/fastpredict",
    packages=setuptools.find_packages(),
    install_requires=[
        'fastai>=2.1.10',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
