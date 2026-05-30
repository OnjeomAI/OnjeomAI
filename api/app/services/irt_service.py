"""IRT 3PL (또는 1PL Rasch 폴백) 능력 추정 서비스.

per-item a/b/c 파라미터가 있으면 3PL, 없으면 difficulty→b 매핑 + a=1/c=0 고정(1PL).
EAP(Expected A Posteriori) 알고리즘으로 theta를 추정한다.
"""

import math

import numpy as np

# theta 탐색 격자: -4.0 ~ 4.0, 간격 0.05 (총 161점)
_THETA_GRID: np.ndarray = np.linspace(-4.0, 4.0, 161)

# N(0, 1) 사전 분포 (log scale)
_PRIOR_LOG: np.ndarray = -0.5 * _THETA_GRID ** 2 - math.log(math.sqrt(2 * math.pi))

# difficulty(1~5) → IRT b 파라미터 기본 매핑 (3PL b_param 없을 때 사용)
_B: dict[int, float] = {1: -2.0, 2: -1.0, 3: 0.0, 4: 1.0, 5: 2.0}

# theta → 능력 수준 레이블 (cutoff 오름차순)
_LEVELS: list[tuple[float, str]] = [
    (-1.5, "하"),
    (-0.5, "중하"),
    (0.5, "중"),
    (1.5, "중상"),
]


def _irt_log_prob(b: float, correct: int, a: float = 1.0, c: float = 0.0) -> np.ndarray:
    """log P(u=correct | θ, a, b, c) — 3PL 모델, 전체 theta 격자에 대해 계산."""
    logit = a * (_THETA_GRID - b)
    p = c + (1.0 - c) / (1.0 + np.exp(-logit))
    if correct:
        return np.log(p + 1e-300)
    else:
        return np.log(1.0 - p + 1e-300)


def estimate_theta(responses: list[dict]) -> dict:
    """EAP 능력 추정.

    Args:
        responses: [{"difficulty": 1~5, "score": 0~100,
                     "a_param": float|None, "b_param": float|None, "c_param": float|None}, ...]

    Returns:
        {"theta": float, "se": float, "ability_level": str, "next_difficulty": int}
    """
    log_post = _PRIOR_LOG.copy()

    for resp in responses:
        b = resp.get("b_param") if resp.get("b_param") is not None else _B.get(int(resp["difficulty"]), 0.0)
        a = resp.get("a_param") if resp.get("a_param") is not None else 1.0
        c = resp.get("c_param") if resp.get("c_param") is not None else 0.0
        correct = 1 if int(resp["score"]) >= 50 else 0

        log_post += _irt_log_prob(b, correct, a, c)

    log_post -= log_post.max()
    posterior = np.exp(log_post)
    posterior /= posterior.sum()

    theta_eap = float(np.dot(_THETA_GRID, posterior))
    se = float(np.sqrt(np.dot((_THETA_GRID - theta_eap) ** 2, posterior)))

    return {
        "theta": round(theta_eap, 4),
        "se": round(se, 4),
        "ability_level": _theta_to_level(theta_eap),
        "next_difficulty": _theta_to_next_difficulty(theta_eap),
    }


def _theta_to_level(theta: float) -> str:
    for cutoff, label in _LEVELS:
        if theta < cutoff:
            return label
    return "상"


def _theta_to_next_difficulty(theta: float) -> int:
    if theta < -1.5:
        return 1
    if theta < -0.5:
        return 2
    if theta < 0.5:
        return 3
    if theta < 1.5:
        return 4
    return 5
