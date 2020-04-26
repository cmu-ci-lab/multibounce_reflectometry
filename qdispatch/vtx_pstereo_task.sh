#!/bin/bash

pkill python
pkill mtstensorflow

#MTSTF_LOC=/home/sassy/thesis/thesis-689/temporary/mitsuba
#SCENE_LOC=/home/sassy/thesis/thesis-689/scenes/vtxscenes

MTSTF_LOC=/home/ubuntu/mitsuba-diff
SCENE_LOC=/home/ubuntu/vtxscenes
MESH_LOC=/home/ubuntu/vtxscenes/meshes

LOGFILE=out

rm /tmp/mtsout-0.hds
rm /tmp/mtsgradout-0.shds
rm /tmp/mtsout-1.hds
rm /tmp/mtsgradout-1.shds
rm /tmp/mtsout-2.hds
rm /tmp/mtsgradout-2.shds

mkfifo /tmp/mtsout-0.hds
mkfifo /tmp/mtsgradout-0.shds
mkfifo /tmp/mtsout-1.hds
mkfifo /tmp/mtsgradout-1.shds
mkfifo /tmp/mtsout-2.hds
mkfifo /tmp/mtsgradout-2.shds

printf "Starting stereo 0\n"
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-0.xml -o "/tmp/mtsout-0.hds" -l 7554 -Ddepth=4 -DsampleCount=64 > $LOGFILE-0.log & export APP_PID=$!
    #MTSTF_PID=$!
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-0-diff.xml -o "/tmp/mtsgradout-0.shds" -l 7555 -Ddepth=4 -DsampleCount=64 > $LOGFILE-1.log & export APP_PID_2=$!
    #MTSTFDIFF_PID=$!

printf "Starting stereo 1\n"
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-1.xml -o "/tmp/mtsout-1.hds" -l 7556 -Ddepth=4 -DsampleCount=64 > $LOGFILE-2.log & export APP_PID_3=$!
    #MTSTF_PID=$!
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-1-diff.xml -o "/tmp/mtsgradout-1.shds" -l 7557 -Ddepth=4 -DsampleCount=64 > $LOGFILE-3.log & export APP_PID_4=$!
    #MTSTFDIFF_PID=$!

printf "Starting stereo 2\n"
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-2.xml -o "/tmp/mtsout-2.hds" -l 7558 -Ddepth=4 -DsampleCount=64 > $LOGFILE-4.log & export APP_PID_5=$!
    #MTSTF_PID=$!
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-2-diff.xml -o "/tmp/mtsgradout-2.shds" -l 7559 -Ddepth=4 -DsampleCount=64 > $LOGFILE-5.log & export APP_PID_6=$!
    #MTSTFDIFF_PID=$!
sleep 2

echo "$SIZE" "$DEPTH" "$SAMPLES" "$MESH" "$GENERATOR" "3"
printf "%s\n%s\n%s\n%s\n%s\n%s" "$SIZE" "$DEPTH" "$SAMPLES" "$MESH_LOC/$MESH" "$GENERATOR" "3" | python $MTSTF_LOC/tf_ops/vtx/mitsuba_vtx_test.py
    
echo $APP_PID
echo $APP_PID_2
echo $APP_PID_3
echo $APP_PID_4
echo $APP_PID_5
echo $APP_PID_6

kill $APP_PID
kill $APP_PID_2
kill $APP_PID_3
kill $APP_PID_4
kill $APP_PID_5
kill $APP_PID_6

#cp /home/sassy/mtstfrun.dat "$SCENE_LOC/scenes/tfscenes/$FILENAME-a$ALPHA-w$WEIGHT-d$DEPTH.dat"
#let COUNTER=COUNTER+1 
