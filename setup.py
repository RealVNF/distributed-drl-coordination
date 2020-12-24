from setuptools import setup, find_packages
requirements = [
    'tensorflow==1.14',
    'click',
    'stable-baselines',
    'cloudpickle==1.2.0',
    'numpy==1.16.4',
    'gym[atari]==0.15.7',
    'mpi4py==3.0.3',
    'tqdm'
]

test_requirements = [
    'flake8',
    'nose2'
]

setup(
    name='spr-rl',
    version='0.1b',
    description='Distributed Service Scaling, Placement, and Routing Using Deep Reinforcement Learning',
    url='https://github.com/RealVNF/distributed-drl-coordination',
    author='Haydar Qarawlus, Stefan Schneider',
    author_email='qarawlus@mail.upb.de',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=requirements + test_requirements,
    tests_require=test_requirements,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'spr-rl=spr_rl.agent.main:main',
        ],
    },
)
