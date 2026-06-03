def linear_decay_lambda(epoch, epochs, decay_start_epoch):
    """Fixed LR before decay_start_epoch, then linearly decay to zero."""
    if epoch < decay_start_epoch:
        return 1.0
    denom = max(1, epochs - decay_start_epoch)
    return max(0.0, 1.0 - (epoch - decay_start_epoch) / float(denom))

