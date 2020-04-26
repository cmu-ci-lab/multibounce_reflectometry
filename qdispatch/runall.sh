#!/bin/bash
COUNTER=0
#MTSTF_LOC=/home/sassy/thesis/thesis-689/temporary/mitsuba
#SCENE_LOC=/home/sassy/thesis/thesis-689

#mkfifo /tmp/mtsout.hds
#mkfifo /tmp/mtsgradout.hds
#export MTS_SRV="ec2-34-201-27-86.compute-1.amazonaws.com:7554;ec2-34-226-244-50.compute-1.amazonaws.com:7554"

#echo $MTS_SRV


filenames=(tf_sphere_test)
alphas=(0.5)
weights=(0.5)
depths=(-1)
#while [  $COUNTER -lt 1 ]; do
for FILENAME in "${filenames[@]}"
do
for ALPHA in "${alphas[@]}"
do
for WEIGHT in "${weights[@]}"
do
for DEPTH in "${depths[@]}"
do
#    $MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/scenes/tfscenes/$FILENAME.xml -p 6 -o "/tmp/mtsout.hds" -l 7554 -Dweight1=0.5 -Dweight2=0.5 -Dalpha=0.1 -Ddepth=4 -DsampleCount=64 > /dev/null & export APP_PID=$!
#MTSTF_PID=$!
#    $MTSTF_LOC/build/release/mitsuba/mtstensorflow $SCENE_LOC/scenes/tfscenes/$FILENAME-diff.xml -p 6 -o "/tmp/mtsgradout.hds" -l 7555 -Ddiff=0 -Dweight1=0.5 -Dweight2=0.5 -Dalpha=0.1 -Ddepth=4 -DsampleCount=64 > /dev/null & export APP_PID_2=$!
    #MTSTFDIFF_PID=$!
#    sleep 2

#    echo "$DEPTH $ALPHA $WEIGHT"
#    printf "%s\n%s\n%s" "$DEPTH" "$ALPHA" "$WEIGHT" | python mitsuba_test.py
#    echo $APP_PID
#    echo $APP_PID_2

#    kill $APP_PID
#    kill $APP_PID_2
#    cp /home/sassy/mtstfrun.dat "$SCENE_LOC/scenes/tfscenes/$FILENAME-a$ALPHA-w$WEIGHT-d$DEPTH.dat"
#    let COUNTER=COUNTER+1 

    FILE="${FILENAME}_a${ALPHA}_w${WEIGHT}_d${DEPTH}"
    qsub -V -o "${FILE}.o${JOB_ID}" -e "${FILE}.e${JOB_ID}" -b y -cwd -v ALPHA="${ALPHA}",WEIGHT="${WEIGHT}",DEPTH="${DEPTH}",FILENAME="${FILENAME}" "./task.sh"

done
done
done
done
