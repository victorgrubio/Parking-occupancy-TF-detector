#!/bin/bash
cd ../docker
docker-compose -p parking kill detector kafka zookeeper kafdrop
