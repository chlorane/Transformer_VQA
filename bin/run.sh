#!/bin/bash
docker run --rm -it --init \
  --user="$(id -u):$(id -g)" \
  --env LOGNAME="$(whoami)" \
  --volume="$PWD:/app" \
  $@