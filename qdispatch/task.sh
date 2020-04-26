#!/bin/bash

pkill python
pkill mtstensorflow

MTSTF_LOC=/home/ubuntu/mitsuba-diff
SCENE_LOC=/home/ubuntu/scenes

rm /tmp/mtsout.hds
rm /tmp/mtsgradout.hds

mkfifo /tmp/mtsout.hds
mkfifo /tmp/mtsgradout.hds

$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME.xml -o "/tmp/mtsout.hds" -l 7554 -Dweight1=0.5 -Dweight2=0.5 -Dalpha=0.1 -Ddepth=4 -DsampleCount=64 > /dev/null & export APP_PID=$!
    #MTSTF_PID=$!
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-diff.xml -o "/tmp/mtsgradout.hds" -l 7555 -Ddiff=0 -Dweight1=0.5 -Dweight2=0.5 -Dalpha=0.1 -Ddepth=4 -DsampleCount=64 > /dev/null & export APP_PID_2=$!
    #MTSTFDIFF_PID=$!
sleep 2

echo "$DEPTH $ALPHA $WEIGHT"
printf "%s\n%s\n%s" "$DEPTH" "$ALPHA" "$WEIGHT" | python $MTSTF_LOC/tf_ops/mitsuba_test.py
    
echo $APP_PID
echo $APP_PID_2

kill $APP_PID
kill $APP_PID_2

#cp /home/sassy/mtstfrun.dat "$SCENE_LOC/scenes/tfscenes/$FILENAME-a$ALPHA-w$WEIGHT-d$DEPTH.dat"
#let COUNTER=COUNTER+1 
