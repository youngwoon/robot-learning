from setuptools import find_packages, setup
from pathlib import Path
from platform import platform


long_description = (Path(__file__).parent / "README.md").read_text()


setup(
    name="rolf",
    version="0.1",
    author="Youngwoon Lee",
    author_email="lywoon89@gmail.com",
    url="https://github.com/youngwoon/robot-learning",
    description="Robot learning framework for research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "requests",
        "numpy",
        "scipy",
        "absl-py",
        "ipdb",
        "hydra-core",
        "wandb",
        "colorlog",
        "tqdm",
        "h5py",
        "slack_sdk",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg",
        "moviepy",
        "mpi4py",
        "gym",
        "mujoco-py",
        "dm_control",
        "dmc2gym @ git+ssh://git@github.com/1nadequacy/dmc2gym.git",
        "tensorflow-macos" if "macOS" in platform() else "tensorflow",
    ],
)
