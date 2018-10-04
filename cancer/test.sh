#!/bin/bash

if [ "$1" != "" ]; then
    echo "python cancer_detector.py" $1
    python cancer_detector.py $1
else
    echo "python cancer_detector.py test_config.json"
    python cancer_detector.py test_config.json
fi
