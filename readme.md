Aggregate of both of these anomaly detection implementations : 
https://github.com/YoungGod/DFR
https://github.com/taikiinoue45/RIAD

Dataset link : 
https://www.mvtec.com/company/research/datasets/mvtec-ad

# How to use
Change the data path and save path line 6 and 7 of main.py

Then, use the command : 

> python main.py

with the main parameters being : 

>  --mode {train,evaluation}   train or evaluation
  
> --model_name {dfr,riad,unet}  specifed model name

> --dfr_type {dfr,dfr-s}   Ponderation of anoamly map according to features ? (yes:dfr-s, no:dfr)

> --unet_size UNET_SIZE  if using model='unet', nb of features in latent space

>  --data_name {tile,grid}   data name