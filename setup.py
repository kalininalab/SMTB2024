from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="smtb",
    version="0.1.0",
    author="Ilya Senatorov",
    author_email="il.senatorov@protonmail.com",
    url="https://github.com/kalininalab/SMTB2024",
    packages=["smtb"],
    install_requires=requirements,
    python_requires=">=3.11",
)
