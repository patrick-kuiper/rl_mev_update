##Run to create data set
python 02_gen_data.py 01_evt_rl_params.yaml

##Run to train model
python 03_train_05JUN24.py 01_evt_rl_params.yaml

##This will output plots, data, and models in respective folders
