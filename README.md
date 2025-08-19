# upscaling-ddpm
My dumb idea for self-supervised upscaling.

# Idea
UNets respect locality because they're a composition of convolutional kernels, so this makes them a good "soft-inductive bias" for upscaling. So here's a dumb idea, make your UNet be a funciton $f: \mathbb R^{d \times d} \to \mathbb R^{2d \times 2d}$, and make the loss be a checkerboard pattern on the input s.t. it changes to the dimension to $\mathbb R^{2d \times 2d}$.

# Evaluation
Consider you have a method for evlaulating log-probability over images $p(x)$, where $x \in \mathbb R^d$. (i.e. Florent's paper https://arxiv.org/abs/2506.05310). It seems believable that if you downsampled those images $\tilde x \in \mathbb R^{\sqrt{d}}$, then upscaled them back to $\mathbb R^d$, they should have the same log-probability. A cleaner way of saying it: the log-probability is invariant under upscaling.

So we'll use this as the reconstruction metric.

# Thanks
- Bao Pahm 