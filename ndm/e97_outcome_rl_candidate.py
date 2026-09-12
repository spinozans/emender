"""Candidate outcome-RL arithmetic only; not a qualified training integration."""
import math
import torch


def group_advantages(rewards, group_ids):
    """Caller must supply complete task groups across all actors/ranks."""
    if rewards.ndim!=1 or not rewards.numel() or group_ids.shape!=rewards.shape or group_ids.dtype!=torch.long:
        raise ValueError('reward/group shape')
    if not torch.isfinite(rewards).all():raise ValueError('nonfinite rewards')
    rewards=rewards.detach().float();out=torch.zeros_like(rewards)
    for group in torch.unique(group_ids):
        selected=group_ids==group
        if int(selected.sum())<2:raise ValueError('incomplete/singleton task group')
        values=rewards[selected];std=values.std(correction=0)
        if float(std)>0:out[selected]=(values-values.mean())/std
    return out


def policy_loss(new_logp, old_logp, reference_logp, advantages, mask,
                episode_total_tokens, episode_count, *, clip=.2, kl_beta=.01):
    """Token-clipped surrogate + k3 KL, averaged by complete episode length.

Rows may be turns, but episode_total_tokens must count ALL sampled tokens in
that row's episode, and episode_count must be the number of distinct episodes
in the complete optimizer batch. Chunk contributions must be summed, not
independently averaged. This helper does not implement DDP normalization.
Only explicitly masked model-generated predictions enter the direct loss.
"""
    if new_logp.ndim!=2 or any(t.shape!=new_logp.shape for t in (old_logp,reference_logp,mask)):
        raise ValueError('logprob/mask shape')
    if mask.dtype!=torch.bool or not bool(mask.any()):raise ValueError('empty/nonboolean assistant mask')
    batch=new_logp.shape[0]
    if advantages.shape!=(batch,) or episode_total_tokens.shape!=(batch,):raise ValueError('episode shape')
    if (not isinstance(episode_count,int) or episode_count<=0 or not 0<clip<1 or
            not math.isfinite(kl_beta) or kl_beta<0):raise ValueError('loss bounds')
    lengths=episode_total_tokens.detach().float()
    if (not torch.isfinite(lengths).all() or (lengths<=0).any() or
            (lengths<mask.sum(-1)).any() or (lengths!=lengths.round()).any()):
        raise ValueError('whole-episode token counts')
    new=new_logp.float()[mask]
    old=old_logp.detach().float()[mask];ref=reference_logp.detach().float()[mask]
    advantage=advantages.detach().float()[:,None].expand_as(new_logp)[mask]
    for value in (new,old,ref,advantage):
        if not torch.isfinite(value).all():raise ValueError('nonfinite active loss input')
    ratio=torch.exp(new-old);delta=ref-new
    kl=torch.exp(delta)-delta-1
    if not torch.isfinite(ratio).all() or not torch.isfinite(kl).all():raise ValueError('probability ratio overflow')
    objective=torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
    denom=lengths[:,None].expand_as(new_logp)[mask]*episode_count
    return ((-objective+kl_beta*kl)/denom).sum()
