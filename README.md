# nn-from-scratch

In this project, I attempt to build an auto differentiation library and use it to build a neural network trained on the MNIST Dataset.

I received significant help from Andrej Kaparthy's micrograd repository which can be found here: https://github.com/karpathy/micrograd

The purpose of this project is to demonstrate that I understand the inner workings of a neural network as well as how libraries such as Pytorch and Tensorflow use auto-differentation to train a neural network.

## Files in this repo

The following files are contained in this repo:

* **autodiff**: This folder contains the code which defines the autodiff logic and neural network building blocks
    * **core.py**: Defines the class 'Value' which stores numbers and can be automatically differentiated. You can carry out basic operations with it such as addition and subtraction.
    * **nn.py**: Defines multiple classes ('Neuron' and 'Layer') which build to the neural network class 'MLP'
* **weights**: Contains the saved weights of a neural network in pickle format
* **mnist-notebook.ipnyb**: A notebook which is designed to load data from the MNIST dataset and train a standard neural network.
