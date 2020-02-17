import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simFrame", # Replace with your own username
    version="0.1.0",
    author="Marco Butz",
    author_email="marco.butz@wwu.de",
    description="A nanophotonic pixel-structure simulation environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://zivgitlab.uni-muenster.de/ag-pernice/simframe.git",
    packages=["simFrame",
                "simFrame.remoteSolver",
                "simFrame.remoteSolver.fdfd",
                "simFrame.remoteSolver.fdtd",
                "simFrame.utility"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
