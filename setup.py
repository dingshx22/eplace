#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ePlace-MS项目安装配置
"""

from setuptools import setup, find_packages

setup(
    name="eplace_ms",
    version="0.1.0",
    description="Electrostatics-Based Placement for Mixed-Size Circuits",
    author="AI Assistant",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
    entry_points={
        'console_scripts': [
            'eplace-ms=src.main:main',
        ],
    },
)