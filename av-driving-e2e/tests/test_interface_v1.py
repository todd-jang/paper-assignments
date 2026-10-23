"""Interface Contract v1.0 shape 테스트 — mock 모듈 기반 (numpy 전용, torch 불필요)

계약 원칙: 실 구현(torch)은 이 테스트의 mock을 교체하는 것만으로 자격 검증된다.
각 mock은 계약 shape을 그대로 이행하는 최소 로직이다.
"""
import numpy as np

# --- 전역 상수 (00_INTERFACE_CONTRACT.md §0 그대로) ---
H = 896            # [라벨-a] Qwen2.5-0.5B
Dv = 1024          # [라벨-b]
Da = 1024          # [라벨-b]
V_PAT = 24 * 24    # 576 = (336/14)^2
V_WIN = 32         # frames
A_TOK = 250        # 5 s @ 50 Hz
N_Q = 32           # resampler latent queries
K = 256            # action bins


# ---------- mock: 에이전트 A — 비전 ----------
def vision_encode(frames: np.ndarray):
    """frames (B, T, 336, 336, 3) -> long tokens (B, T, 576, Dv), projected 지나지도 별도 제공"""
    B, T = frames.shape[0], frames.shape[1]
    # (B,T,336,336,3) → (B,T,576,588): 24×24 패치 각 14×14×3 content
    tok_long = (frames.reshape(B, T, 24, 14, 24, 14, 3)
                       .transpose(0, 1, 2, 4, 3, 5, 6)
                       .reshape(B, T, V_PAT, 14 * 14 * 3))
    W = np.random.default_rng(0).normal(size=(tok_long.shape[-1] + 1, Dv))
    x = np.pad(tok_long, ((0, 0), (0, 0), (0, 0), (0, 1)))
    proj = x @ W                                   # (B,T,576,Dv)
    return proj


def vision_sliding_window(tok, win=V_WIN):
    """40프레임 → 최근 win프레임만 (Sliding Window): (B,win,576,Dv)"""
    return tok[:, -win:]


# ---------- mock: 에이전트 B — 오디오 ----------
def audio_encode(wave: np.ndarray, sr=16_000, win_s=5.0):
    """(B, sr*win_s) -> (B, A_TOK, Da) 고정 길이"""
    B = wave.shape[0]
    assert wave.shape[1] == int(sr * win_s)
    return np.zeros((B, A_TOK, Da))                # shape 계약만 검증


def audio_sync(audio_t0, video_t0):
    """frame t ↔ 세그먼트 [t-0.05, t+0.05]: 타임스탬프 오차 계약 ≤0.1 s"""
    return abs(audio_t0 - video_t0) <= 0.1


# ---------- mock: 에이전트 C — Perceiver Resampler ----------
def resampler(v_tok: np.ndarray, a_tok: np.ndarray, n_q=N_Q, out_dim=H):
    """축소 mock: 계약 검증 대상은 '출력 shape이 입력 길이와 무관 고정'뿐 — 실 cross-attention은 agent C 구현"""
    B = v_tok.shape[0]
    kv = np.concatenate(
        [v_tok.reshape(B, -1, v_tok.shape[-1]).mean(axis=1), a_tok.mean(axis=1)], axis=1
    )                                              # (B, 2*out_in) — 입력 개수 불변 요약
    scores = kv @ np.random.default_rng(3).normal(size=(kv.shape[1], n_q))
    return np.repeat(scores[..., None], out_dim, axis=2)   # (B, n_q, out_dim)


# ---------- mock: 에이전트 D — 액션 ----------
def action_tokenize(steer, throttle, lo=(-1.0, 0.0), hi=(1.0, 1.0), K=K):
    i_st = int(np.floor(np.clip((steer - lo[0]) / (hi[0] - lo[0]), 0, 1) * (K - 1)))
    i_th = int(np.floor(np.clip((throttle - lo[1]) / (hi[1] - lo[1]), 0, 1) * (K - 1)))
    return f"{i_st} {i_th}"


def action_detokenize_midbin(tok: str, lo=(-1.0, 0.0), hi=(1.0, 1.0), K=K):
    """bin 중점 복원 (데모 D 계약 — floor 복원 계-드리프트 방지)"""
    i_st, i_th = map(int, tok.split())
    st = lo[0] + (i_st + 0.5) * (hi[0] - lo[0]) / (K - 1)
    th = lo[1] + (i_th + 0.5) * (hi[1] - lo[1]) / (K - 1)
    return np.clip(st, lo[0], hi[0]), np.clip(th, lo[1], hi[1])


def ema_filter(x: np.ndarray, alpha=0.3):
    y = np.empty_like(x); acc = 0.0
    for i, xi in enumerate(x):
        acc = alpha * xi + (1 - alpha) * acc
        y[i] = acc
    return y


# ================= 테스트 =================
def test_vision_shape():
    frames = np.random.rand(2, V_WIN, 336, 336, 3)
    tok = vision_encode(frames)
    assert tok.shape == (2, V_WIN, V_PAT, Dv)


def test_vision_sliding():
    tok = vision_encode(np.random.rand(2, 40, 336, 336, 3))
    win = vision_sliding_window(tok, win=V_WIN)
    assert win.shape == (2, V_WIN, V_PAT, Dv), "40프레임 → 최근 32만"


def test_audio_fixed():
    wave = np.random.rand(2, 80_000)     # 5 s @ 16 kHz
    out = audio_encode(wave)
    assert out.shape == (2, A_TOK, Da), "항상 고정 (250, Da)"


def test_audio_sync():
    assert audio_sync(video_t0=10.0, audio_t0=10.08) is True
    assert audio_sync(video_t0=10.0, audio_t0=10.15) is False


def test_resampler_fixed_output():
    a = resampler(np.random.randn(2, V_WIN, V_PAT, H), np.random.randn(2, A_TOK, H))
    b = resampler(np.random.randn(2, 5, V_PAT, H), np.random.randn(2, A_TOK, H))
    assert a.shape == (2, N_Q, H), "입력 길이와 무관 항상 (32, H)"
    assert a.shape == b.shape, "변하지 않음 = O(N²) 차단"


def test_action_roundtrip_midbin():
    bin_w = 2.0 / (K - 1)                  # W=2 스파이어링
    for x in (-0.97, -0.31, 0.0, 0.5, 0.76):
        tok = action_tokenize(x, 0.7)
        s, t = action_detokenize_midbin(tok)
        assert abs(s - x) <= bin_w, f"중점 복원 max error = bin폭, 관측 {abs(s-x):.4f}"


def test_ema_jitter():
    rng = np.random.default_rng(7)
    x = 0.5 * np.tanh(2 * rng.standard_normal(200)) + 0.01 * rng.standard_normal(200)
    jx = np.mean(np.abs(np.diff(x)))
    jy = np.mean(np.abs(np.diff(ema_filter(x, 0.3))))
    assert jy < jx, f"EMA 지터 감소 계약: {jy:.4f} < {jx:.4f}"


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn(); print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{len(fns)-fails}/{len(fns)} 통과 — Interface Contract v1.0 자격")
    raise SystemExit(1 if fails else 0)
