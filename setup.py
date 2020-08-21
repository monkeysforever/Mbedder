from setuptools import find_packages, setup
import re


def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html).strip()
    return cleantext


def fetch_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
        readme = clean_html(readme)
    return readme


def fetch_requirements():
    requirements_file = "requirements.txt"

    with open(requirements_file) as f:
        reqs = f.read()

    reqs = reqs.strip().split("\n")
    return reqs


DISTNAME = "Mbedder"
DESCRIPTION = "Mbedder: A pytorch powered framework for seemlessly adding contextual text embeddings \
 from pretrained models"
LONG_DESCRIPTION = fetch_long_description()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
AUTHOR = "Randeep Ahlawat"
AUTHOR_EMAIL = "ahlawat.randeep@gmail.com"
REQUIREMENTS = (fetch_requirements(),)
EXCLUDES = ("docs", "tests", "tests.*")

if __name__ == '__main__':
    setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=find_packages(exclude=EXCLUDES),
        python_requires=">=3.6",
        version='0.0.4',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url='https://github.com/monkeysforever/Mbedder',
        classifiers=[
            "Programming Language :: Python :: 3.6",            
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: Unix",
            'Intended Audience :: Science/Research',

        ]
    )
