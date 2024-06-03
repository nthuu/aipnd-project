# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Train:
python train.py --arch vgg16 --learning_rate 0.001 --hidden_units 4096 --epochs 5 --gpu

Predict:
python predict.py test_data\flower.jpg --top_k 5 --category_names cat_to_name.json --gpu