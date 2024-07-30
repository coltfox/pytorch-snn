# Spiking Neural Network Using PyTorch

Basic Spiking Neural Network (SNN) implementation using PyTorch. This is based off of the article [Building And Training Spiking Neural Networks From Scratch](https://r-gaurav.github.io/2024/01/04/Building-And-Training-Spiking-Neural-Networks-From-Scratch.html) by Ramashish
Gaurav.

It performs with a 97% test accuracy on the MNIST dataset using only dense layers.

## Network Structure
* Input Encoder (784 neurons)
    * Uses "rate encoding" to convert the input data (in this case pixels) into spikes
* Dense (1024 neurons)
* Dense (1024 neurons)
* Output Dense (10 neurons)