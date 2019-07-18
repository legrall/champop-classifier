from setuptools import find_packages, setup

setup(
    name='classifier',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask-restful==0.3.7',
        'flask-cors==3.0.8',
        'flask==1.1.0',
        'flasgger==0.9.2',
        'pytest==5.0.1',
        'gunicorn==19.9.0',
        'webargs'
    ],
)