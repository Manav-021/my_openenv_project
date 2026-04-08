def compute_score(total_reward: float, max_reward: float) -> float:
    if max_reward == 0:
        return 0.0
    score = total_reward / max_reward
    return max(0.0, min(score, 1.0))