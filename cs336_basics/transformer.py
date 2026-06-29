import math, os

from jaxtyping import Bool, Float, Int
from torch import Tensor, device
import torch
from einops import einsum
from collections.abc import Iterable
import numpy as np
import numpy.typing as npt
from typing import IO, Any, BinaryIO

class linear(torch.nn.Module):
    """
    in_features: int  final dimension of the input
    out_features: int  final dimension of the output
    device: torch.device | None = None  Device to store the parameters on
    dtype: torch.dtype | None = None  Data type of the parameters
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = torch.nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        sigma = (2.0 / (d_in + d_out)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(
        self,
        in_features: Float[Tensor, " ... d_in"]
    ) -> Float[Tensor, " ... d_out"]:
        return einsum(in_features, self.weight, "... d_in, d_out d_in -> ... d_out")
    
class embedding(torch.nn.Module):
    """
    num_embeddings: int  Size of the vocabulary
    embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
    device: torch.device | None = None  Device to store the parameters on
    dtype: torch.dtype | None = None  Data type of the parameters
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(
        self,
        token_ids: Int[Tensor, " ..."]
    ) -> Float[Tensor, " ... embedding_dim"]:
        return self.weight[token_ids]
    
class rmsnorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(
        self,
        input: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        input_type = input.dtype
        input = input.to(torch.float32)
        ms = input.pow(2).mean(dim=-1, keepdim=True) # (B, T, 1) — reduce LAST axis
        rms = (ms + self.eps).sqrt()                 # (B, T, 1) — same shape
        output = input / rms                         # (B, T, d_model) — broadcasts
        output = output * self.weight

        return output.to(input_type)
    
class swiglu(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = linear(d_in = d_model, d_out = d_ff, device=device, dtype=dtype)
        self.w2 = linear(d_in = d_ff, d_out = d_model, device=device, dtype=dtype)
        self.w3 = linear(d_in = d_model, d_out = d_ff, device=device, dtype=dtype)
    
    def forward(
        self,
        in_features: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_out"]:
        w1_x = self.w1(in_features)
        gate = w1_x * torch.sigmoid(w1_x)    # ← SiLU: x * σ(x)
        value = self.w3(in_features)  # the actual content to pass
        h = gate * value                     # element-wise product in the wide d_ff space
        return self.w2(h)

class rope(torch.nn.Module):
    # Construct the RoPE module and create buffers if needed.
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        assert(d_k % 2 == 0)
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # precomute rotation
        k = torch.arange(d_k // 2, device=device, dtype=torch.float32)                  # shape: (1, d_k/2)
        freqs = 1.0 / (theta ** (2 * k / self.d_k)) # shape: (1, d_k/2)
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.int32)      # shape: (1, max_seq_len)
        angles = positions[:, None] * freqs [None, :]    # shape: (max_seq_len, 1) * (1, d_k/2) -> (max_seq_len, d_k/2)
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(
        self,
        in_query_or_key: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        # look up cos/sin -> (..., seq_len, d_k/2)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        
        # split the last dim into pairs
        x = in_query_or_key
        x_pairs = x.view(*x.shape[:-1], self.d_k // 2, 2)
        x_even = x_pairs[..., 0] # (..., seq_len, d_k/2)
        x_odd = x_pairs[..., 1]  # (..., seq_len, d_k/2)

        # rotate each pair
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # interleave back into d_k wide last dim
        out = torch.stack((out_even, out_odd), dim=-1) # (..., seq_len, d_k/2, 2)
        out = out.flatten(-2)                          # (..., seq_len, d_k)
        
        return out
    
def softmax(
    x: Float[Tensor, " ..."],
    dim: int
) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x_max = x.amax(dim=dim, keepdim=True)
    x_shift = x - x_max
    exp_x = x_shift.exp() # element wise
    out = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return out

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.size(-1)
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / d_k ** 0.5

    if mask is not None:
        # mask broadcasts against scores' last two dims automatically
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    out = einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return out

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    
    # "... sequence_length d_model" -> "... sequence_length d_model"
    Q = einsum(in_features, q_proj_weight, "... seq d_in, ... d_out d_in -> ... seq d_out")
    K = einsum(in_features, k_proj_weight, "... seq d_in, ... d_out d_in -> ... seq d_out")
    V = einsum(in_features, v_proj_weight, "... seq d_in, ... d_out d_in -> ... seq d_out")
    
    # split heads (d_model = num_heads x d_k)
    *batch, seq, _ = in_features.shape
    d_k = d_model // num_heads

    # split heads: (..., seq, d_model) -> (..., num_heads, seq, d_k)
    Q = Q.reshape(*batch, seq, num_heads, d_k).transpose(-3, -2)
    K = K.reshape(*batch, seq, num_heads, d_k).transpose(-3, -2)
    V = V.reshape(*batch, seq, num_heads, d_k).transpose(-3, -2)
    
    # causal mask: lower-triangular, True = keep
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=in_features.device))

    # call sdpa — handles (..., num_heads, seq, d_k) via the ... batch dims
    attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    # merge heads: (..., num_heads, seq, d_k) -> (..., seq, d_model)
    attn_out = attn_out.transpose(-3, -2).reshape(*batch, seq, d_model)
    
    # output projection
    return einsum(attn_out, o_proj_weight, "... seq d_in, d_out d_in -> ... seq d_out")

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """


    # Project
    Q = einsum(in_features, q_proj_weight, "... seq d_in, ... d_out d_in -> ... seq d_out")
    K = einsum(in_features, k_proj_weight, "... seq d_in, ... d_out d_in -> ... seq d_out")
    V = einsum(in_features, v_proj_weight, "... seq d_in, ... d_out d_in -> ... seq d_out")

    # split heads (d_model = num_heads x d_k)
    *batch, seq, _ = in_features.shape
    d_k = d_model // num_heads

    # split heads: (..., seq, d_model) -> (..., num_heads, seq, d_k)
    Q = Q.reshape(*batch, seq, num_heads, d_k).transpose(-3, -2)
    K = K.reshape(*batch, seq, num_heads, d_k).transpose(-3, -2)
    V = V.reshape(*batch, seq, num_heads, d_k).transpose(-3, -2)
    
    # 3. Default token_positions if not provided
    if token_positions is None:
        token_positions = torch.arange(seq, device=in_features.device)

    # Apply RoPE to Q and K (now they're (..., num_heads, seq, d_k)) , make number_heads at -3, so that we can apply rope reasily
    rope_module = rope(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_features.device)
    Q = rope_module(Q, token_positions)
    K = rope_module(K, token_positions)

    # causal mask: lower-triangular, True = keep
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=in_features.device))

    # call sdpa — handles (..., num_heads, seq, d_k) via the ... batch dims
    attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    # merge heads: (..., num_heads, seq, d_k) -> (..., seq, d_model)
    attn_out = attn_out.transpose(-3, -2).reshape(*batch, seq, d_model)
    
    # output projection
    return einsum(attn_out, o_proj_weight, "... seq d_in, d_out d_in -> ... seq d_out")


class transformer_block(torch.nn.Module):
    def __init__ (
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.ln1 = rmsnorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = rmsnorm(d_model=d_model, device=device, dtype=dtype)

        self.q_proj = linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.k_proj = linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.v_proj = linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.output_proj = linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)

        self.ffn = swiglu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
    
    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ):
        lay1 = in_features + multihead_self_attention_with_rope(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            o_proj_weight=self.output_proj.weight,
            in_features=self.ln1(input=in_features),
            token_positions=token_positions
        )

        lay2 = lay1 + self.ffn(
            in_features=self.ln2(input=lay1)
        )

        return lay2
    
class transformer_lm(torch.nn.Module):
    def __init__ (
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        self.token_embeddings = embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        self.layers = torch.nn.ModuleList([transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        ) for i in range(num_layers)])

        self.ln_final = rmsnorm(d_model=d_model, device=device, dtype=dtype)

        self.lm_head = linear(d_in=d_model, d_out=vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        input_embedding = self.token_embeddings(in_indices)

        layer_embedding = self.layers[0](input_embedding)

        for i in range(1, self.num_layers):
            layer_embedding = self.layers[i](layer_embedding)

        layer_embedding_norm = self.ln_final(layer_embedding)

        layer_embedding_lm_head = self.lm_head(layer_embedding_norm)

        # return softmax(layer_embedding_lm_head, -1)
        return layer_embedding_lm_head
    

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 1: compute the batch indices
    batch_size = inputs.shape[0]
    batch_indices = torch.arange(batch_size, device=inputs.device)

    # 2: select the correct class logit for each example
    correct_logits = inputs[batch_indices, targets]
    loss_per_example = inputs.logsumexp(dim=-1) - correct_logits
    
    return loss_per_example.mean()

class adamw_cls(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")

        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("adamw_cls does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]
                state["t"] += 1
                t = state["t"]

                # Decoupled weight decay (AdamW).
                p.data.mul_(1.0 - lr * weight_decay)

                # Update first and second moments.
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias-corrected Adam step size.
                step_size = lr * ((1.0 - beta2 ** t) ** 0.5) / (1.0 - beta1 ** t)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

        return loss

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if warmup_iters > 0 and it < warmup_iters:
        return it * 1.0 / warmup_iters * max_learning_rate
    elif cosine_cycle_iters > warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
    
def run_gradient_clipping(
        parameters: Iterable[torch.nn.Parameter],
        max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    if max_l2_norm <= 0:
        raise ValueError(f"max_l2_norm must be positive, got {max_l2_norm}")

    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return

    grads = [p.grad for p in params]
    total_norm = torch.sqrt(sum(torch.sum(g * g) for g in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if len(dataset) <= context_length:
        raise ValueError(
            f"dataset length ({len(dataset)}) must be greater than context_length ({context_length})"
        )

    max_start = len(dataset) - context_length
    starts = torch.randint(0, max_start, (batch_size,)).numpy()

    x_np = np.stack([dataset[start : start + context_length] for start in starts])
    y_np = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])