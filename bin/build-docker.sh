#!/bin/bash
cd ..
python3 setup.py build --gatv
cd docker
docker-compose -p parking up -d kafka zookeeper kafdrop
sleep 5
docker-compose -p parking up --build -d detector
