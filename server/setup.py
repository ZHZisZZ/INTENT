"""
INTENT python package configuration.
"""

from setuptools import setup

setup(
    name='INTENT',
    version='0.1.0',
    packages=['INTENT'],
    include_package_data=True,
    install_requires=[
        'arrow',
        'bs4',
        'Flask',
        'html5validator',
        'pycodestyle',
        'pydocstyle',
        'pylint',
        'pytest',
        'requests',
        'selenium',
        'funcsigs',
        'markupsafe==2.0.1',
        'flask_cors'
    ],
    python_requires='>=3.6',
)
