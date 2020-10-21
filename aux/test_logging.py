#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:32:18 2018

@author: visiona
"""
import logging
from setupLogging import setupLogging

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    setupLogging()
    logger.info('LOGGING SETUP CORRECTLY')