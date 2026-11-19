from setuptools import setup

setup(
    name="peakcut",
    version="0.2.0",
    description="PeakCut Studio: Autonomous AI Video Clipper & Shorts Pipeline",
    py_modules=["colab_runner"],
    entry_points={
        "console_scripts": [
            "peakcut=colab_runner:cli_main",
        ],
    },
)
