import torch


def normalize_losses(losses: torch.Tensor,
                     losses_mask: torch.Tensor,
                     instance_normalization: str,
                     batch_normalization: str) -> torch.Tensor:
    """
    Normalizes the input losses based on the type of normalization specified
    by `instance_normalization` and `batch_normalization`.

    Parameters
    ----------
    losses: (batch_size, num_tokens)
        The loss per summary token.
    losses_mask: (batch_size, num_tokens)
        The mask which indicates which losses are valid.
    instance_normalization:
        The method of normalizing each item in the batch, either "sum" or "average",
        which will sum or average the losses per summary.
    batch_normalization:
        The method of normalizing the losses per summary, either "sum" or "average".
        After the loss for each instance is compuated via the method specified
        by `instance_normalization`, the subsequent losses are either summed
        or averaged.

    Returns
    -------
    The normalized loss.
    """
    # First, apply the loss mask to 0-out any invalid losses
    losses = losses * losses_mask.float()

    if instance_normalization == 'sum':
        # shape: (batch_size,)
        loss_per_summary = losses.sum(dim=1)
    elif instance_normalization == 'average':
        # shape: (batch_size,)
        lengths = losses_mask.float().sum(dim=1)
        # shape: (batch_size,)
        loss_per_summary = losses.sum(dim=1) / lengths
    else:
        raise Exception(f'Unknown type of instance normalization: {instance_normalization}')

    if batch_normalization == 'sum':
        loss = loss_per_summary.sum()
    elif batch_normalization == 'average':
        loss = loss_per_summary.mean()
    else:
        raise Exception(f'Unknown type of batch normalization: {batch_normalization}')

    return loss
