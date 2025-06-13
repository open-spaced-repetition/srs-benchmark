from config import Config


def sm2(r_history: str, config: Config) -> float:
    """
    Implementation of the SM-2 algorithm.

    Args:
        r_history: A comma-separated string of ratings
        config: Configuration object containing s_max parameter

    Returns:
        float: The calculated interval
    """
    ivl = 0.0
    ef = 2.5
    reps = 0
    for rating_str in r_history.split(","):
        rating = int(rating_str) + 1
        if rating > 2:
            if reps == 0:
                ivl = 1
                reps = 1
            elif reps == 1:
                ivl = 6
                reps = 2
            else:
                ivl = ivl * ef
                reps += 1
        else:
            ivl = 1
            reps = 0
        ef = max(1.3, ef + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)))
        ivl = min(max(1, round(ivl + 0.01)), config.s_max)
    return ivl
