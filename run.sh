#!/bin/zsh

drive_id="1iUReVs8L7FrtwpKG4QeluwlX8Ri1cJXh"

wget \
  --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(\
    wget \
      --quiet \
      --save-cookies /tmp/cookies.txt \
      --keep-session-cookies \
      --no-check-certificate "https://docs.google.com/uc?export=download&id=$drive_id" -O- | \
      sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p'\
  )&id=$drive_id" -O ../data.zip && rm -rf /tmp/cookies.txt

mkdir -p ../data
unzip -qq ../data.zip

python3 trainer.py