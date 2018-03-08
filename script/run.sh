#!/bin/bash

if [ $# -le 3 ] 
then
    # not enough parameters provided -> print help page
    echo "Usage: ./deal-spectrum [EXECUTABLE] [FIRST] [LAST] [INCREMENT]"
    echo "Perform automated parameter study."
else
    # print header
    echo " m     n     p"
    for (( k=$2; k<=$3; k+=$4 ))
    do
        # run program
        $1 $k
    done
fi


