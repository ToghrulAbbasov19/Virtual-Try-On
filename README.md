# Virtual-Try-On
In this project, I did diffusion based Virtual Try On. I used diffusion based approach from Guided Diffusion and then added additional cross-attention mechanism for cloth conditioning and used Autoencoder for dealing with high dimension. Both high and low resolution, and unconditional training are supported.

## Environment Setup
```bash 
conda env create -f environment.yml
conda activate vto
```

## Training
For training the main model with autoencoder and cross attention mechanism:
```bash 
python main_cond.py --data_path {PATH/TO/DATA} --save_dir {PATH/TO/SAVE/DIRECTORY} --pair_path {PATH/TO/PAIR_DATA}
```
