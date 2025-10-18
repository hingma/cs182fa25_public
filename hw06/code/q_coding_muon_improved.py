import torch
import torch.optim as optim
from torch.nn.functional import normalize

def newton_schulz_orthogonalize(X: torch.Tensor, num_iters: int, use_quintic=False):
    """
    Apply Newton-Schulz iterations to approximate orthogonalization.

    This function can apply either:
    - Cubic polynomial f(X) = (3X - X^3)/2 (default)
    - Quintic polynomial f(X) = X(15/8 - 5X^2/4 + 3X^4/8) (faster convergence)
    
    Both polynomials gradually force all singular values to 1 while preserving singular vectors.

    Args:
      X (torch.Tensor): Input matrix to orthogonalize
      num_iters (int): Number of Newton-Schulz iterations
      use_quintic (bool): Whether to use quintic polynomial (faster) or cubic polynomial

    Returns:
      torch.Tensor: Orthogonalized matrix
    """
    dtype = X.dtype
    # Use bfloat16 for potential speed/memory savings during the iterations
    X = X.bfloat16()
    # Recall from prior homeworks that we can transpose the matrix to speed up computation.
    transposed = False
    if X.size(-2) < X.size(-1):
        transposed = True
        X = X.mT

    # Ensure spectral norm is at most sqrt(3) for cubic or sqrt(3.5) for quintic
    # The quintic polynomial has a wider basin of attraction
    max_norm = 3.5**0.5 if use_quintic else 3**0.5
    norm = torch.linalg.norm(X, dim=(-2, -1), keepdim=True)
    X = torch.div(X, norm + 1e-7) * max_norm

    # Apply Newton-Schulz iterations
    for _ in range(num_iters):
        if use_quintic:
            # Apply the quintic polynomial f(X) = X(15/8 - 5X^2/4 + 3X^4/8)
            X2 = torch.matmul(X, X)  # X^2
            X4 = torch.matmul(X2, X2)  # X^4
            X = torch.matmul(X, 15/8 * torch.eye(X.shape[-1], device=X.device) - 
                             5/4 * X2 + 3/8 * X4)
        else:
            # Apply the cubic polynomial f(X) = (3X - X^3)/2
            X3 = torch.matmul(torch.matmul(X, X), X)  # X^3
            X = (3 * X - X3) / 2  # (3X - X^3)/2

    if transposed:
        X = X.mT

    return X.to(dtype)

def muon_update_quintic(grad, momentum, beta=0.95, ns_iters=3):
    """
    Apply Muon update with quintic polynomial Newton-Schulz orthogonalization.
    Using the quintic polynomial should require fewer iterations for similar convergence.
    
    Args:
        grad: The gradient tensor
        momentum: The momentum buffer
        beta: Momentum factor
        ns_iters: Number of Newton-Schulz iterations (can be lower with quintic)
        
    Returns:
        Update tensor for parameter update
    """
    momentum.lerp_(grad, 1 - beta)  # momentum = beta * momentum + (1-beta) * grad
    update = momentum.clone()

    # If the parameter is a convolutional kernel, then flatten to a 2D matrix
    original_shape = update.shape
    reshaped = False
    if update.ndim > 2:
        reshaped = True
        update = update.view(update.size(0), -1)

    # Apply Newton-Schulz orthogonalization with quintic polynomial
    update = newton_schulz_orthogonalize(update, ns_iters, use_quintic=True)
    
    # Apply muP scaling factor for consistent update magnitude
    scaling_factor = torch.sqrt(torch.tensor(max(1, update.size(-2) / update.size(-1))))
    update = update * scaling_factor

    # Restore shape if needed
    if reshaped:
        update = update.view(original_shape)

    return update

class MuonQuintic(optim.Optimizer):
    """
    Muon optimizer with quintic polynomial Newton-Schulz orthogonalization.
    
    This implementation should converge faster with fewer iterations than the
    original cubic polynomial implementation.
    """
    def __init__(self, params, lr=0.01, beta=0.95, ns_iters=3, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, ns_iters=ns_iters, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            ns_iters = group['ns_iters']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                # Get state for this parameter
                state = self.state[p]
                # Initialize momentum buffer if it doesn't exist
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(grad)

                # Apply weight decay directly to parameters (AdamW style)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Apply newton_schulz if parameter is a matrix
                if p.ndim >= 2:
                    update = muon_update_quintic(grad, state['momentum'],
                                               beta=beta, ns_iters=ns_iters)
                    # Apply update to parameters
                    p.add_(update, alpha=-lr)
                else:
                    # For non-matrix parameters, i.e. bias, use standard momentum update
                    momentum = state['momentum']
                    momentum.mul_(beta).add_(grad)
                    p.add_(momentum, alpha=-lr)

        return None
