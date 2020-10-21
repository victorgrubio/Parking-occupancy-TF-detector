# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-12-10 13:39:15
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-12-10 13:42:04
import os
import logging.config

import yaml

def setupLogging(
    default_path='log/logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
