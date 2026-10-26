# Interface Contract v1.0 — Tensor Shape 고정 (변경 불가 / 변경은 v1.1 절차만)

> 성격: 에이전트 A~D 공동 계약. 예시 모양의 원 출처는 teacher(2026-10-23). 베이스라인 수치 중 라벨 있음 = 학생 확정(teacher 승인 대기).
> 검증 독립성: 본 규격의 shape은 `tests/test_interface_v1.py`가 mock 모듈로 선별하며, 실제 구현은 이 테스트를 그대로 통과해야 한다.

## 0. 전역 상수

| 상수 | 값 | 출처 |
|---|---|---|
| BATCH | B (변수, 테스트 기본 2) | — |
| LLM_HIDDEN (H) | **896** | 백본 선택(Qwen2.5-0.5B-Instruct) = [라벨-a] 학생 확정 · hidden 수치 자체 = [라벨-b] 공식 config 본검증 전 |
| VISION_PATCH_TOKENS | **576** (24² = (336/14)²) | CLIP ViT-L/14@336 — LLaVA-1.5와 동일 계열 조건([라벨-a]) |
| VISION_DIM (Dv) | 1024 | CLIP ViT-L 마지막 블록 차원 — [라벨-b] config 본검증 필요 |
| VIDEO_FPS | 10 | [라벨-a] 실용 기본치 (teacher 미지정) |
| VIDEO_WINDOW | 32 frames (=3.2 s) | [라벨-a] |
| AUDIO_TOKEN_RATE | 50 Hz (20 ms/토큰) | WavLM Base+ 출력 간격 — [라벨-b] 논문 본검증 필요 |
| AUDIO_DIM (Da) | 1024 | WavLM Base+ hidden — [라벨-b] |
| AUDIO_WINDOW | 250 tokens (=5.0 s) | [라벨-a] |
| RESAMPLER_LATENT_Q | **32** | teacher 지정 (예시 문구 그대로) |
| ACTION_BINS (K) | 256 | RT-2 §3.2 직접 조회 — 고정 |
| CONTROL_MIN__MAX | steer [−1,1] · throttle [0,1] | [라벨-a] |
| LATENCY_BUDGET | 액션 출력 <50 ms (teacher 지정) · 제어 루프 10 Hz | [라벨-a] 루프 계산구 |

## 1. 모듈 계약 (순서대로 연결 — 뒤 모듈은 앞 모듈 shape에만 의존)

```
[Frame (336,336,3) → float32, RGB, 0..1]                        (A 진입)
A_OUT   vision/frame tokens : (T, 576, Dv)  T≤VIDEO_WINDOW     (A 출구 · per-frame 유지)
A_PROJ  (T, 576, H)           — Dv→H 투영 후 LLM쪽 통일 차원

[Waveform (16_000 Hz float32) → Log-Mel → AST/WavLM]            (B 진입)
B_OUT   audio/frame tokens    : (250, Da)   고정 길이·타임스탬프 부착  (B 출구)
B_PROJ  (250, H)

[공간 싱크] frame t ↔ audio 세그먼트 [t−0.05, t+0.05] wall-clock   (B 책임)

C_IN    union tokens          : (T·576 + 250, 1024) + dtype 타입특성 (0=v,1=a)   (C 진입)
C_OUT   resampler latent      : (32, H) — **입력 길이와 무관 항상 고정**        (C 출구)

D_IN    prompt = text tokens + C_OUT (인터리빙)                                 (D 진입)
D_ACT   "st thr" 2 출력 토큰 → 256 bin 역산 → (steer, throttle) 지속
        복원 = bin 중점 (RT-2 floor 대비 데모 D 기준) + EMA α=0.3 저역통과   (D 출구)
```

## 2. Shape 계약 표 (통합 시 단 하나의 진실)

| 경계 텐서 | 모양 | 소유 | 소비 |
|---|---|---|---|
| vision/frame tokens (long) | (B, T, 576, Dv) | A | C |
| vision/frame tokens (projected) | (B, T, 576, H) | A | C |
| audio tokens (long) | (B, 250, Da) | B | C |
| audio tokens (projected) | (B, 250, H) | B | C |
| type token dtype | (B, T·576+250) 0/1 | C | C |
| resampler latent | (B, **32**, H) ← 항상 | C | D |
| action tokens (사전 직렬화) | (B, 2) 문자열 목록 "st thr" | D | log |
| decoded control | (B,) float ×2 (steer, throttle) | D | EMA → 물리 |

## 3. 단위 테스트 요구 (각 모듈 독립 — teacher 원칙 2)

| 테스트 | 대상 | 합격 조건 |
|---|---|---|
| `test_vision_skeleton` | A | 32프레임 주입 → 유지 (32,576,Dv); 40프레임 → 창 초과분 날림 후 (32,576,Dv) |
| `test_audio_skeleton` | B | 5 s 파형 → (250, Da) 고정; frame t↔audio t 타임스탬프 오차 ≤ 0.1 s |
| `test_resampler_compression` | C | 입력 극한값 T=5 → 3,130 토큰 vs T=32 → 18,682 토큰 (T·576+250) → 출력 (32, H) 둘 다; **그 변하지 않음 = O(N²) 차단 증명** |
| `test_action_roundtrip` | D | 중점 복원: max\|x−detok(x)\| ≤ 1/510 (bin폭/2, 이론 상한 — 테스트·데모 D는 bin폭 2/255 기준 진술이며 관측 3.3e-3 ≤ 0.0078 정합); EMA α 출력 지터 **입력 지터 미만 감소** (테스트는 jy<jx; 실측 ≈0.6× @α=0.3 — 데모 C) |

## 4. 라벨 요약 (정직 선언)

- [라벨-a] 학생 확정 기본값 — teacher 승인 대기: Qwen2.5-0.5B·CLIP ViT-L/14@336·10 FPS·창 32/5s·루프 10 Hz·제어 범위
- [라벨-b] 외부 모델 사양 — 사용 전 공식 config/논문 본검증 필수: Qwen hidden 896·CLIP out 1024·WavLM 50Hz/1024
- 실제 구현은 **mock 증명 테스트(test_interface_v1.py, numpy 불필요, torch 없이도 동캐스팅 가능)**를 계약 삼아 본모형 교체만 수행
