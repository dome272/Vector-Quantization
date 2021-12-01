import torch
from torch.autograd import Function


class VQ(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, codebook.size(1))

            codebook_sqr = torch.sum(codebook**2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten, dim=1, keepdim=True)

            # calc distances -> (codebook - inputs)**2 = codebook**2 - 2*codebook*inputs + inputs**2
            # = codebook_sqr + inputs_sqr - 2 * (codebook_sqr @ inputs_sqr)
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
            # distances = codebook_sqr + inputs_sqr - 2 * (inputs_flatten @ codebook.t())

            indices = torch.min(distances, dim=1)[1].view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError()


class VQStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        """
        This function realizes the Codebook mapping --> Replace each vector of the input by its closest neighbor from
        the codebook. It further saves the indices of the codebook vectors, which the inputs will be replaced with along
        with the codebook in the context for the backward pass.
        :param ctx: Acts as a storing variable to communicate between forward and backward.
        :param inputs: Encoded images from the encoder.
        :param codebook: Just the codebook. num_vectors x dim_vectors
        :return: The codebook vectors which are closest to each input vector, along with their indices.
        """
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)

        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        """
        Implementation of the "straight through" gradient passing.
        X -> Encoder -> Vector Quantization -> Decoder -> X_rec
            Gradient <------------------------ Gradient
        The gradient from the Decoder will directly be passed to the Encoder.
        No gradient will be calculated in this module.
        The backward sort of acts as an identity mapping for the gradient.
        :param ctx: Acts as a storing variable to communicate between forward and backward.
        :param grad_output: Gradient of decoder.
        :param grad_indices:
        :return: Cloned gradient of decoder which will be passed to encoder.
        """
        grad_inputs, grad_cb = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            indices, codebook = ctx.saved_tensors
            emb_size = codebook.size(1)
            grad_output_flatten = grad_output.contiguous().view(-1, emb_size)
            grad_cb = torch.zeros_like(codebook)
            grad_cb.index_add_(0, indices, grad_output_flatten)

        return grad_inputs, grad_cb


vq = VQ.apply
vq_st = VQStraightThrough.apply
__all__ = [vq, vq_st]
