#!/bin/bash
gdown 'https://drive.google.com/uc?id=1Dp8ZjvZwdxjqs43ZNTA8gCSoravQURDg' -O data/datasets/
unzip data/datasets/dataset.zip -d data/datasets/07112023
rm -rf data/datasets/dataset.zip