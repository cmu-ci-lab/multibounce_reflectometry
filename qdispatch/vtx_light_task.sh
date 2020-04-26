#!/bin/bash

pkill python
pkill mtstensorflow

MTSTF_LOC=/home/sassy/thesis/thesis-689/temporary/mitsuba
TESTS_LOC=/home/sassy/thesis/thesis-689/temporary/tests
SCENE_LOC=/home/sassy/thesis/thesis-689/temporary/vtxscenes

#MTSTF_LOC=/home/ubuntu/mitsuba-diff
#SCENE_LOC=/home/ubuntu/vtxscenes

MESH_LOC=$SCENE_LOC/meshes
LIGHTS_LOC=$SCENE_LOC/lights

LOGFILE=out

MTSSRV="ec2-18-207-183-170.compute-1.amazonaws.com"

rm /tmp/mtsout-0.hds
rm /tmp/mtsgradout-0.shds

mkfifo /tmp/mtsout-0.hds
mkfifo /tmp/mtsgradout-0.shds

# TODO: TEMPORARY
#DEPTH=6
#MESH=groove-large.ply
#FILENAME=tf_dir_ostereo_red
#LIGHTS=test1-inv.lt

printf "Starting stereo 0\n"
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME.xml -c "$MTSSRV" -p 2 -o "/tmp/mtsout-0.hds" -l 7554 -Ddepth=4 -DsampleCount=64 -DlightX=0.0 -DlightY=0.0 -DlightZ=0.0 > $LOGFILE-0.log & export APP_PID=$!
    #MTSTF_PID=$!
$MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/$FILENAME-diff.xml -c "$MTSSRV" -p 2 -o "/tmp/mtsgradout-0.shds" -l 7555 -Ddepth=4 -DsampleCount=64 -DlightX=0.0 -DlightY=0.0 -DlightZ=0.0 > $LOGFILE-1.log & export APP_PID_2=$!
    #MTSTFDIFF_PID=$!
sleep 2

python $MTSTF_LOC/tf_ops/vtx/mitsuba_vtx_multi_test.py $TESTS_LOC/$TESTPARAMS
 
echo $APP_PID
echo $APP_PID_2

kill $APP_PID
kill $APP_PID_2

#cp /home/sassy/mtstfrun.dat "$SCENE_LOC/scenes/tfscenes/$FILENAME-a$ALPHA-w$WEIGHT-d$DEPTH.dat"
#let COUNTER=COUNTER+1 
