Pytorch implementation of Vector Quantized Variational AutoEncoder (VQ-VAE)
---------------------------------------------------------------------------

### AutoEncoder (AE)

### Variational AutoEncoder (VAE)
 - Ubiquitous AutoEncoder face two problems when it comes to the latent space representation:
   - Similar data points are not close to each other.
     - One would wish that similar points, for example points of the same class, would lie in the same region in the latent space, which is not the case with default AEs.
     - Here they can be distributed over the whole space without any structure between the classes.
   - The distribution of the data is not centered in the latent space.
     - The points can lie everywhere in the latent space and have enormous distances between each other, which is not favourable.
 - We would rather want the latent space to be well organized in terms of having regions of classes and to have "center of mass" where the majority of points lie.

## Vector Quantized Variational AutoEncoder (VQ-VAE)
### VAE vs VQ-VAE
The main difference is that VAEs learn a continuous representation of the data, whereas the VQ-VAE learns a discrete latent space, also called codebook.
This can be naturally useful in a lot of problems, which are discrete in itself. Language for example is discrete in its phonemes, words etc. 
A lot of tasks in computer vision are naturally discrete. For instance, detecting objects in an image follows discrete classes to predict.

### Basic procedure of the VQ-VAE
We already discussed the benefit of using a discrete latent space aka codebook, but how is this done in practice?
The normal VAE already provides a good starting point, from where we don't need to change a lot of things.
But first we need to get a sense what the codebook actually looks like. Simply imagine it as a long list full of vectors **e_1,....,e_N** of dimension **d**.
Converting an input **X** to a codebook vector now, you simply feed **X** through your Encoder and replace it with the closest vector, in euclidean distance, from the codebook.
The mathematical formulation for this is the following:
<insert formular here>
After converting the output from the Encoder to the codebook vector, this codebook vector is then fed into the Decoder to reconstruct the input **X**.
Having only one output vector from the Encoder would result in being only able to generate N distinct images from the Decoder.
To overcome this, normal VQ-VAE Encoder output a greater number of vectors, which are then all handled as the one vector before.
Every single vector will be replaced by its closest vector from the codebook and after converting all vectors, the resulting output can be fed through the Decoder to reconstruct **X**.
To make this more concrete: Imagine an Encoder which outputs a tensor of shape **64x32**. Then each of the **64 vectors** of **dimension=d=32** can be replaced by its closest neighbor from the codebook
and voilà you have a much higher variability in your possible outputs.

### How do we learn the codebook?
Starting off more generally: "How do we learn the model at all?". To answer this, we can take a look at the loss function of the model.

**loss = mean_squared_error(X, X_reconstructed) + mean_squared_error(sg(Encoder(x)), codebook_vectors) + beta * mean_squared_error(Encoder(x), sg(codebook_vectors))**

The first term is just the difference between the actual image and the reconstructed image. This specifically affects the learning of the Encoder and Decoder.
The second and third term are meant for the codebook learning. This very form of the two terms are taken from the Vector Quantization literature and gives our model its name.
**sg()** is the **stop gradient** operation, which simply cuts the gradient flow at this point. The result of the last two terms is that the outputs of the Encoder and the
Codebook vectors are "pulled" together, to be closer to each other. And that's it. That's how our model learns.

### Code Details
The code is specified for the MNIST dataset and can be run without any big modifications and be easily extended to other datasets. Every 1000 steps it will also save an image with how the model is currently doing: the current image and its reconstruction.
These images can be found in the "results" folder.

### Acknowledgements
A lot of code has been taken from the [official VQ-VAE implementation](https://github.com/ritheshkumar95/pytorch-vqvae) and from [lucidrains implementation](https://github.com/lucidrains/vector-quantize-pytorch).
 ```bibtex
 @misc{oord2018neural,
    title   = {Neural Discrete Representation Learning},
    author  = {Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu},
    year    = {2018},
    eprint  = {1711.00937},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
 ```
