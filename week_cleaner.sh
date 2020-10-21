#!/bin/sh
cd /var/www/html/videos/
#For folder in Videos
for D in `find . -type d`
do
	cd /var/www/html/videos/$D/ #go to folder
	find -type f -name "*.mp4" -mtime +7 -delete #delete +1 week  videos
done
