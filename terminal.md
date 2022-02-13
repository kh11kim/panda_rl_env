# train
python train.py --algo tqc --env MyEnv-v0 --env-kwargs level:0.1 render:False --save-replay-buffer
python train.py --algo tqc --env MyEnv-v1 --env-kwargs level:0.1 render:False --save-replay-buffer
python train.py --algo tqc --env MyEnv-v1 --env-kwargs level:0.1 render:False --save-replay-buffer -i logs/tqc/MyEnv-v1_1/MyEnv-v1.zip
python train.py --algo tqc --env MyEnv-v2 --env-kwargs level:0.1 render:False --save-replay-buffer -i logs/tqc/MyEnv-v2_2/MyEnv-v2.zip

python train.py --algo tqc --env PandaReach2-v0 --env-kwargs level:0.1 render:False --save-replay-buffer
# enjoy
python train.py --algo tqc --env MyEnv-v0 --env-kwargs level:0.1 render:False --save-replay-buffer
python enjoy.py --algo tqc --env MyEnv-v1 --folder logs/tqc/MyEnv-v1_8
python enjoy.py --algo tqc --env MyEnv-v2 --folder logs/ --env-kwargs render:True level:0.1