#!/bin/bash
python ./graph_updater.py > graph_updater.log 2>&1 &
exec python ./app.py