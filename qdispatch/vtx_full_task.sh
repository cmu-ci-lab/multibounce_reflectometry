#!/bin/bash

pkill python
pkill mtstensorflow

MTSTF_LOC=/home/sassy/thesis/thesis-689/temporary/mitsuba
SCENE_LOC=/home/sassy/thesis/thesis-689/temporary/vtxscenes

#MTSTF_LOC=/home/ubuntu/mitsuba-diff
#SCENE_LOC=/home/ubuntu/vtxscenes

MESH_LOC=$(SCENE_LOC)/meshes
LIGHTS_LOC=$(SCENE_LOC)/lights

TEMP_MESH=/tmp/mts_mesh.ply
PYTHON=python

LOGFILE=out

rm /tmp/mtsout-0.hds
rm /tmp/mtsgradout-0.shds

mkfifo /tmp/mtsout-0.hds
mkfifo /tmp/mtsgradout-0.shds

printf "Starting rendering server\n"
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME.xml -o "/tmp/mtsout-0.hds" -l 7554 -Ddepth=4 -DsampleCount=64 -DlightX=0.0 -DlightY=0.0 -DlightZ=0.0 > $LOGFILE-0.log & export APP_PID=$!
    #MTSTF_PID=$!

printf "Starting differential rendering server\n"
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-diff.xml -o "/tmp/mtsgradout-0.shds" -l 7555 -Ddepth=4 -DsampleCount=64 -DlightX=0.0 -DlightY=0.0 -DlightZ=0.0 > $LOGFILE-1.log & export APP_PID_2=$!
    #MTSTFDIFF_PID=$!
sleep 2

printf "Refining normals\n"
echo "$SIZE" "$DEPTH" "$SAMPLES" "$MESH" "$GENERATOR" "1" "$LIGHTS"
printf "%s\n%s\n%s\n%s\n%s\n%s\n%s" "$SIZE" "$DEPTH" "$SAMPLES" "$MESH_LOC/$MESH" "$GENERATOR" "1" "$LIGHTS_LOC/$LIGHTS" | python $MTSTF_LOC/tf_ops/vtx/mitsuba_vtx_multi_test.py

printf "Refining mesh\n"
$PYTHON $MTSTF_LOC/remesher/sct_remesh.py $TEMP_MESH $TEMP_MESH

echo $APP_PID
echo $APP_PID_2

kill $APP_PID
kill $APP_PID_2

#cp /home/sassy/mtstfrun.dat "$SCENE_LOC/scenes/tfscenes/$FILENAME-a$ALPHA-w$WEIGHT-d$DEPTH.dat"
#let COUNTER=COUNTER+1 
