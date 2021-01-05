from setuptools import setup, find_packages
requirements = [
    'tensorflow==1.14',
    'click',
    'stable-baselines==2.10',
    'cloudpickle==1.2.0',
    'numpy>=1.16.4,<=1.19.3',
    'gym[atari]==0.15.7',
    'mpi4py==3.0.3',
    'tqdm',
    'networkx==2.4'
]

test_requirements = [
    'flake8',
    'nose2'
]

setup(
    name='spr-rl',
    version='1.0',
    description='Distributed Online Service Coordination Using Deep Reinforcement Learning',
    url='https://github.com/RealVNF/distributed-drl-coordination',
    author='Haydar Qarawlus, Stefan Schneider',
    author_email='qarawlus@mail.upb.de',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires='>=3.6, <3.8',
    install_requires=requirements + test_requirements,
    tests_require=test_requirements,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'spr-rl=spr_rl.agent.main:main',
        ],
    },
)
