def to_device(batch_data, device):
    """Move batch data to specified device (cpu/cuda)."""
    if device == 'cuda':
        if isinstance(batch_data, dict):
            return {k: v.cuda() if hasattr(v, 'cuda') else v 
                   for k, v in batch_data.items()}
        elif hasattr(batch_data, 'cuda'):
            return batch_data.cuda()
    return batch_data