from setuptools import find_packages, setup
from os import path


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "torchange",
                             "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


install_requires = [
    'numpy',
    'albumentations>=0.4.2',
    'tifffile',
    'scikit-image',
    'tqdm',
    'einops',
    'timm',
    'datasets[vision]',
]
setup(
    name='torchange',
    version=get_version(),
    description='pytorch-change-models',
    keywords='Remote Sensing, '
             'Earth Vision, '
             'Deep Learning, '
             'Change Detection, '
             'Change Data Generation, ',
    packages=find_packages(exclude=['projects', 'tools']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Utilities',
    ],
    url='https://github.com/Z-Zheng/pytorch-change-models',
    author='Zhuo Zheng',
    author_email='zhuozheng@cs.stanford.edu',
    license='CC-BY-NC 4.0',
    setup_requires=[],
    tests_require=[],
    install_requires=install_requires,
    zip_safe=False)
