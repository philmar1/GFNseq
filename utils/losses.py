import torch

def flow_matching_loss(incoming_flows, outgoing_flows, reward):
    """Flow matching objective converted into mean squared error loss."""
    return (incoming_flows.sum() - outgoing_flows.sum() - reward).pow(2)  

def trajectory_balance_loss(logZ, log_P_F, log_P_B, reward):
    """Trajectory balance objective converted into mean squared error loss."""
    return (logZ + log_P_F - torch.log(reward) - log_P_B).pow(2)  # TODO: Complete.
