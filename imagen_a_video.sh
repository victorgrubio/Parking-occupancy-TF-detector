#!/bin/sh
cd images/current
Day=$(date --date="yesterday" +'%Y%m%d')
mkdir $Day
find -name "Cam01_$Day*" -type f -exec /bin/mv {} ./$Day \;
cd $Day
ffmpeg -pattern_type glob -i "*.jpg" $Day.mp4
mv $Day.mp4 /var/www/html/videos/Cam01/
cd ..
rm $Day -r
#For folder in Videos
for D in `find . -type d`
do
	cd /var/www/html/videos/$D/ #go to folder
	find -type f -name "*.mp4" -mtime +7 -delete #delete +1 week  videos
done





