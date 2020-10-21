#!/bin/bash
cd ..
python3 setup.py build --gatv
cd docker
docker-compose up --build detector
