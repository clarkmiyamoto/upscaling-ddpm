# upscaling-ddpm
My dumb idea for self-supervised upscaling.

# Idea
UNets respect locality because they're a composition of convolutional kernels, so this makes them a good "soft-inductive bias" for upscaling. So here's a dumb idea, make your UNet be a funciton $f: \mathbb R^{d \times d} \to \mathbb R^{2d \times 2d}$, and make the loss be a checkerboard pattern on the input s.t. it changes to the dimension to $\mathbb R^{2d \times 2d}$.