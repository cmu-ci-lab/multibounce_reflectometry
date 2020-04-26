#!/bin/bash

MTSTF_LOC=/home/sassy/thesis/thesis-689/temporary/mitsuba
SCENE_LOC=/home/sassy/thesis/thesis-689/scenes/tf_scenes

$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME.xml -p 1 -c $MTS_SRV -o "/tmp/mtsout.hds" -l 7554 -Dalpha=0.1 -Dweight1=0.5 -Dweight2=0.5 -Ddepth=4 -DsampleCount=64 > /dev/null & export APP_PID=$!
    #MTSTF_PID=$!
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-diff.xml -p 1 -c $MTS_SRV -o "/tmp/mtsgradout.hds" -l 7555 -Ddiff=0 -Dalpha=0.1 -Dweight1=0.5 -Dweight2=0.5 -Ddepth=4 -DsampleCount=64 > /dev/null & export APP_PID_2=$!
    #MTSTFDIFF_PID=$!
sleep 2

echo "$DEPTH $ALPHA $WEIGHT"
printf "%s\n%s\n%s" "$DEPTH" "$ALPHA" "$WEIGHT" | python mitsuba_test.py
    
echo $APP_PID
echo $APP_PID_2

kill $APP_PID
kill $APP_PID_2

#cp /home/sassy/mtstfrun.dat "$SCENE_LOC/scenes/tfscenes/$FILENAME-a$ALPHA-w$WEIGHT-d$DEPTH.dat"
#let COUNTER=COUNTER+1 
