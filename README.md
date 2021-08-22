# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

python train.py --save_dir checkpoint/ --gpu
python predict.py --image flowers/train/10/image_08090.jpg --checkpoint checkpoint/checkpoint.pth --category_names cat_to_name.json
