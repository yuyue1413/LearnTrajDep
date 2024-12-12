#!/bin/bash

python main.py --input_n 10 --output_n 25 --dct_n 35 --data_dir data/h36m/h36m

python main.py --epoch 5 --input_n 10 --output 10 --dct_n 20 --data_dir data/h36m/h36m

python demo.py --input_n 10 --output_n 10 --dct_n 20 --data_dir data/h36m/h36m