#!/bin/bash

# Script to replicate our experiments for ALIFE 2020
# If you have downloaded alife-results.tar.xz, this is only
# relevant for your own interest.

# Random seeds may behave differently on different computers
# or with different tensorflow/Python installations.

# This would take an extraordinarily long amount of time to
# actually run on a single local machine.

for ((i=0; i<20; i++));
do
  python3 -m alife.step1_eQ $i
done

for ((i=0; i<20; i++));
do
  python3 -m alife.step2_ddpg $i
done

python3 -m alife.step3_extract_critics -n 20

python3 -m alife.step4_new_actor_training -a 400300
python3 -m alife.step4_new_actor_training -a 2010
python3 -m alife.step4_new_actor_training -a 5

python3 -m alife.step5_extract_spectra
