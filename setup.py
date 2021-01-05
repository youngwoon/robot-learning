from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="robot_learning",
    version="0.1",
    author="Youngwoon Lee",
    author_email="lywoon89@gmail.com",
    description="Robot learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/youngwoon/robot-learning",
    packages=find_packages(),
    install_requires=[
        "requests",
        "numpy",
        "scipy",
        "wandb",
        "colorlog",
        "tqdm",
        "h5py",
        "ipdb",
        "opencv-python",
        "imageio",
        "mpi4py",
        "gym",
        "mujoco-py",
        "absl-py",
        "dm_control @ git+ssh://git@github.com/deepmind/dm_control.git",
        "dmc2gym @ git+ssh://git@github.com/1nadequacy/dmc2gym.git",
    ],
)
