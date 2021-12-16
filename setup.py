"""Install midi-ddsp."""

import setuptools

setuptools.setup(
    name='midi-ddsp',
    version='0.1.0',
    description='Synthesis of MIDI with DDSP',
    author='Google Inc. & Yusong Wu',
    author_email='no-reply@google.com, wuyusongwys@gmail.com',
    url='http://github.com/magenta/midi-ddsp',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    scripts=[],
    install_requires=[
        'ddsp',
        'pretty_midi',
        'music21',
        'pandas'
    ],
    extras_require={
        'test': ['pytest', 'pylint!=2.5.0'],
    },
    entry_points={
        'console_scripts': [
            'midi_ddsp_synthesize = midi_ddsp.midi_ddsp_synthesize:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    keywords='audio MIDI MIDI-synthesizer machinelearning music',
)
