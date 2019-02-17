PyTorch implementation of Neural Processes (NP) by Garnelo et al https://arxiv.org/abs/1807.01622
# MNIST image completion
The task is to complete an image given some number \[1;784\] of context points (coordinates) at which we know the greyscale pixel intensity \[0;1].

# Results
The first row shows the observed greyscale context points. Unobserved pixels are in blue.
The five rows below show realisations of different samples of the global latent variable `z` given the context points above. Compare with Figure 4 in the [paper](https://arxiv.org/abs/1807.01622).
##### 10 context points
![10 context points](results/ep_300_cps_10.png?raw=true "Title")
##### 100 context points
![100 context points](results/ep_300_cps_100.png?raw=true "Title")
##### 300 context points
![300 context points](results/ep_300_cps_300.png?raw=true "Title")
##### 784 context points (full image)
![784 context points](results/ep_300_cps_784.png?raw=true "Title")
<br>



# How to run
`python main.py` produces the results above. The script saves examples of reconstructed images at the end of every epoch in `results/`.

# Requirements
 - Python 3
 - PyTorch 0.4.1 or later (tested with 1.0.1)



# Other NP implementations
R + TensorFlow - https://github.com/kasparmartens/NeuralProcesses