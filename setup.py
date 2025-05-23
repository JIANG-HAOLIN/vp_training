from setuptools import setup, find_packages

setup(
    name='broncho_dl',              # The name of your package
    version='0.1',                  # Version number (update as needed)
    packages=find_packages(),       # Automatically find all packages and subpackages
    include_package_data=True,      # Include non-Python files if specified in MANIFEST.in (optional)
    install_requires=[              # List of dependencies
    ],
)