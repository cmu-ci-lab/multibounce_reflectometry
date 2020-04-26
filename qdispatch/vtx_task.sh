#!/bin/bash

pkill python
pkill mtstensorflow

#MTSTF_LOC=/home/sassy/thesis/thesis-689/temporary/mitsuba
#SCENE_LOC=/home/sassy/thesis/thesis-689/scenes/vtxscenes

MTSTF_LOC=/home/ubuntu/mitsuba-diff
SCENE_LOC=/home/ubuntu/vtxscenes
MESH_LOC=/home/ubuntu/vtxscenes/meshes

rm /tmp/mtsout.hds
rm /tmp/mtsgradout.shds

mkfifo /tmp/mtsout.hds
mkfifo /tmp/mtsgradout.shds

$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME.xml -o "/tmp/mtsout.hds" -l 7554 -Ddepth=4 -DsampleCount=64 > /dev/null & export APP_PID=$!
    #MTSTF_PID=$!
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-diff.xml -o "/tmp/mtsgradout.shds" -l 7555 -Ddepth=4 -DsampleCount=64 >> /dev/null & export APP_PID_2=$!
    #MTSTFDIFF_PID=$!
sleep 2

echo "$SIZE" "$DEPTH" "$SAMPLES" "$MESH" "$GENERATOR" "1"
printf "%s\n%s\n%s\n%s\n%s\n%s" "$SIZE" "$DEPTH" "$SAMPLES" "$MESH_LOC/$MESH" "$GENERATOR" "1" | python $MTSTF_LOC/tf_ops/vtx/mitsuba_vtx_test.py
    
echo $APP_PID
echo $APP_PID_2

kill $APP_PID
kill $APP_PID_2

#cp /home/sassy/mtstfrun.dat "$SCENE_LOC/scenes/tfscenes/$FILENAME-a$ALPHA-w$WEIGHT-d$DEPTH.dat"
#let COUNTER=COUNTER+1 
