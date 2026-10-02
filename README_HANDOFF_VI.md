# BÀN GIAO THỰC NGHIỆM — BEST-ARM IDENTIFICATION UNDER ONE-BIT FEEDBACK

Phiên bản bàn giao: 2026-10-02  
Code thực nghiệm chính: v39.5 FAST-PARALLEL  
Mục tiêu: người nhận có thể (i) hiểu đúng từng experiment, (ii) kiểm tra raw data,
(iii) regenerate toàn bộ figure, và (iv) rerun các experiment khi cần.

---

## 1. Nội dung package

```text
README_HANDOFF_VI.md              <- tài liệu bàn giao này
README.md                         <- hướng dẫn reproduction ngắn
requirements.txt
REPRODUCE.sh
DATA_INVENTORY.csv
experiment_manifest.json

onebit_bai/                       <- source thuật toán / oracle / profile likelihood
scripts/                          <- các runner thực nghiệm
tests/                            <- regression + math/implementation tests
configs/                          <- cấu hình chính
data/                             <- raw CSV đã dùng cho paper
reproduce/
    plot_all_figures.py           <- regenerate các figure hiện tại
    audit_handoff.py              <- kiểm tra integrity/data accounting
figures/                          <- figure PNG/PDF đã generate
paper/
    main_through_experiments.tex  <- phần main mới nhất đến hết Experiments
    preview_full.pdf              <- preview paper v46
```

Package này dùng dữ liệu synthetic; không có private dataset.

---

## 2. Bài toán và hai observation models

### Model E — exponential-family / Gaussian benchmark

Trong thực nghiệm hiện tại, Model E dùng Gaussian reward với known
standard deviation `sigma = 0.2`.

One-bit learner không quan sát trực tiếp reward `X`.  Nó chọn threshold `q`
và quan sát bit dạng `Y = 1{X >= q}`.  Thiết kế threshold của best arm phải
là **shared** giữa các challenger trong một K-arm experiment khả thi.

Các method name trong raw data:

- `e_tt`: one-bit Top-Two dùng shared adaptive threshold design.
- `e_full`: full-feedback comparator.
- `e_companion`: finite-confidence identification-schedule companion, là
  construction riêng cho non-asymptotic diagnostic.

### Model U — arbitrary bounded distributions

Reward nằm trong `[0,1]`.  Trong simulation, Beta laws chỉ được dùng làm
generator; thuật toán/lower bound của Model U không giả biết Beta family.

- `u_raw`: giữ raw current-status pair `(Q,Y)`, `Q ~ Uniform[0,1]`.
- `u_full`: full-feedback bounded nonparametric comparator.
- `u_marginal`: marginal-bit diagnostic dùng trong correctness checks.

Không được pool `u_raw` practical với certified-U sieve run: hai stopping
rules khác nhau.

---

## 3. Fixed / shared / pairwise trong Model E

Ba khái niệm này là **design spaces lý thuyết**, không phải ba kiểu bootstrap.

- **Fixed**: tất cả arm dùng một common deterministic threshold `q0`.
  Đây là feasible subclass nhưng bị hạn chế.
- **Shared**: một feasible joint arm-threshold occupation measure cho toàn bộ
  K-arm experiment.  Best-arm threshold law là cùng một law phục vụ mọi
  challenger.  Đây là object đúng của lower bound và algorithm.
- **Pairwise relaxation**: mỗi pair `(best, challenger j)` được phép có một
  best-arm threshold law riêng.  Đây là relaxation; nói chung không thể
  hiện thực hóa đồng thời thành một physical K-arm observation stream.

Quan hệ:
```text
Gamma_fixed <= Gamma_shared <= Gamma_pair
```
không claim strictness.

Không bắt buộc phải có Monte-Carlo fixed-vs-shared-vs-pairwise để theorem
đúng; nếu thêm thì chỉ nên là supplementary numerical ablation.

**Lưu ý khác:** "paired" trong paired bootstrap là Monte-Carlo coupling,
hoàn toàn khác "pairwise relaxation" ở trên.

---

## 4. Instance và configuration chính

### 4.1 Clustered five-arm instance dùng Q1/Q3/correctness

```text
means = [0.70, 0.64, 0.63, 0.62, 0.61]
K = 5
beta = 0.5
```

Model E:
```text
Gaussian sigma = 0.2
```

Model U:
```text
Beta generator
concentration = 12
support = [0,1]
```

### 4.2 Legacy representative paired benchmark

Gaussian:
```text
CLUSTER5 = [0.70,0.64,0.63,0.62,0.61]
beta = 0.5
```

Bounded:
```text
HARD5 = [0.70,0.60,0.50,0.40,0.30]
beta = 0.5
Beta concentration = 12
```

Population ratios used for the legacy relative-efficiency plot:

```text
T*_1bit / T*_full, E-CLUSTER5  = 1.5719756221598549
T*_1bit / T*_full, U-HARD5     = 3.4370394005896188
```

### 4.3 Certified Model-U stress instance

```text
means = [0.90,0.50,0.30,0.20,0.10]
beta = 0.5
deltas = [0.10,0.05,0.02,0.01]
reps = 200
cap = 200000
refresh = 50
decide = 50
boundary = certified
```

Final result: mọi trajectory đều censor trước khi có recommendation.
Aggregate nằm ở `data/U_certified_easy5_summary.csv`.

---

## 5. Raw data: mapping và ý nghĩa

### `data/E_Q1.csv`
Direct first-order Model-E diagnostic.

```text
method = e_tt
reps = 200
deltas = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-10]
rows = 1200
all stopped
observed errors = 0
```

### `data/U_Q1.csv`
Direct first-order Model-U raw current-status diagnostic.

```text
method = u_raw
reps = 200
same 6 deltas as E_Q1
rows = 1200
all stopped
observed errors = 0
```

### `data/E_Q3.csv`
Deep-confidence full-vs-one-bit comparison.

```text
methods = e_full, e_tt
reps = 200 per method/delta
deltas = [1e-4,1e-6,1e-8,1e-10]
rows = 1600
all stopped
observed errors = 0
```

### `data/U_Q3.csv`
Deep-confidence full-vs-one-bit comparison.

```text
methods = u_full, u_raw
reps = 200 per method/delta
same 4 deltas
rows = 1600
all stopped
observed errors = 0
```

**Quan trọng:** `u_raw` trong U_Q3 reuse đúng U_Q1 trajectories ở bốn delta
chung. Không được tính Q1 và Q3-U như hai independent datasets.

### `data/E_correctness.csv`

```text
method = e_tt
deltas = [0.10,0.05,0.02,0.01]
600 reps / delta
rows = 2400
stop fraction = 1
observed errors = 0
```

### `data/U_correctness.csv`

```text
methods = u_raw, u_marginal
deltas = [0.10,0.05,0.02,0.01]
600 reps / method / delta
rows = 4800
stop fraction = 1
observed errors = 0
```

Zero observed error là implementation diagnostic, **không** phải empirical
proof của rare-delta PAC guarantee.

### `data/E_nonasym_companion.csv`

```text
methods = e_tt, e_companion
deltas = [1e-2,1e-3,1e-4]
400 reps / method / delta
rows = 2400
all stopped
observed errors = 0
```

### Legacy paired files

`legacy_E_CLUSTER5_b05.csv`:
```text
algorithms = full_asym, full_finite, onebit_asym, onebit_finite
200 rep IDs
5 deltas 1e-4 ... 1e-8
rows = 4000
```

`legacy_U_HARD5_b05.csv`:
```text
algorithms = full, onebit
200 rep IDs
5 deltas 1e-4 ... 1e-8
rows = 2000
```

Hai file này là nguồn raw của representative paired relative-efficiency plot.

---

## 6. Schema CSV mới

Các file Q1/Q3/correctness/companion có schema:

```text
rep
model
method
delta
tau
censor_at
stopped
error
recommended
true_best
stop_stat
stop_boundary
stop_stat_over_boundary
lambda_at_stop
boundary_over_L
run_time_s
seed
```

Ý nghĩa quan trọng:

- `tau`: stopping time; blank nếu trajectory censor.
- `censor_at`: cap.
- `stopped=1`: algorithm thực sự stop.
- `recommended=-1`: không có recommendation; đây là sentinel, không phải arm -1.
- `error=0` khi `stopped=0` không được đọc là "recommendation đúng".
- `stop_stat/stop_boundary`: evidence và boundary lúc stop khi available.
- `seed`: seed của replication.

Khi có censoring, không được lọc `stopped=1` rồi gọi mean đó là `E[tau]`.

---

## 7. Q1 — direct first-order diagnostic

Statistic:

```text
R_FO(delta) = mean(tau_delta) * Gamma_star / log(1/delta)
            = mean(tau_delta) / [T_star log(1/delta)]
```

Population rates dùng trong plotting package:

```text
Gamma_E_Q1 = 0.004080994480837258
Gamma_U_Q1 = 0.0033946213856749087
```

Horizontal reference `1` là first-order asymptotic reference, không phải
finite-delta upper bound.

Q1 là direct test gần nhất với theorem dạng
`E[tau] ~ T_star log(1/delta)`.

---

## 8. Legacy relative-efficiency experiment

Statistic đúng:

```text
R_rel(delta) =
    [mean(tau_1bit) / mean(tau_full)]
    /
    [Tstar_1bit / Tstar_full]
```

Không dùng:
```text
mean(tau_1bit / tau_full)
```

Bootstrap:
1. resample **whole replication IDs**;
2. lấy full/one-bit pair tương ứng;
3. recompute ratio of means;
4. lấy percentile 2.5% / 97.5%.

Mốc `R_rel=1` chỉ có nghĩa relative sample penalty khớp population theory.
Nó không chứng minh từng method đã đạt direct first-order regime.

---

## 9. Practical finite-confidence calibration

Practical boundary trong legacy implementation:

```text
b_prac(n, delta) =
log((K-1) * pi^2 * max(n,1)^2 / (6 delta))
```

Fixed-point predictor:

```text
n_pred(delta) =
min { n in stopping-check schedule :
      n * Gamma_pop >= b_prac(n,delta) }
```

Observed-vs-predictor figure là practical implementation diagnostic.

Nó:
- không phải direct Q1 first-order plot;
- không phải certified non-asymptotic theorem bound.

---

## 10. Q3 — full vs one-bit

Q3 chỉ so empirical stopping-time scale của two methods ở deep confidence.

Không gọi bootstrap là "paired" nếu simulation record không chứng minh một
common-random-number coupling.  Cùng `rep` number tự nó không tạo pairing.

---

## 11. Certified-U stress test và lý do all-censor

Proof-certified U sieve penalty rất lớn ở practical `n`.

Final stress run giữ 5 arms nhưng dùng separated means để giảm instance
difficulty.  Dù vậy 200/200 trajectories vẫn không cross certified boundary
trước cap `200000`.

Cách report đúng:
- computational conservatism diagnostic;
- stop fraction 0;
- không plot mean stopping time;
- không biến `error=0` thành correctness evidence;
- không trộn với practical `u_raw`.

---

## 12. Code architecture

### `onebit_bai/algorithms.py`
Sampling/stopping algorithms, boundaries, full/one-bit experiment logic.

### `onebit_bai/oracles.py`
Population/numerical oracles, characteristic values, Model-E shared design.

### `onebit_bai/u_profile.py`
Raw current-status profile likelihood and generalized PAVA logic.

v39.5 optimization:
- old per-block bisection replaced by exact closed-form root where applicable;
- computational optimization only, not a statistical approximation.

### `onebit_bai/experiment.py`
Core simulation runner and multiprocessing.

### `onebit_bai/report.py`, `focused_report.py`, `combined_report.py`
Aggregations and report generation.

### `scripts/run_custom_suite.py`
Recommended high-level suite runner.

### `scripts/create_custom_case.py`
Create custom JSON cases.

### `tests/`
Regression tests for math, U finite-proof components, and parallel/resume logic.

---

## 13. Reproduce figures without rerunning expensive Monte Carlo

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Audit data first:

```bash
python reproduce/audit_handoff.py
```

Regenerate current figures:

```bash
python reproduce/plot_all_figures.py
```

Output goes to:

```text
figures/
```

The plotting script uses fixed bootstrap seed and percentile bootstrap CIs.

---

## 14. Run tests

```bash
pytest -q tests
```

The code package was built so that expensive experiment reruns are not needed
just to validate the plotting/data path.

---

## 15. Rerun the main cluster suite

Configuration:

```text
configs/cluster5_main.json
```

Example:

```bash
python scripts/run_custom_suite.py \
  --config configs/cluster5_main.json \
  --workers 8 \
  --run \
  --output-root runs/cluster5
```

For a multi-core Linux server, set BLAS/OpenMP threads to 1 before using many
worker processes:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Changing `workers` changes runtime only; it must not change the statistical
configuration.

---

## 16. Certified-U rerun configuration

The exact final configuration is recorded in:

```text
configs/certified_u_easy5_final.json
```

Equivalent command pattern:

```bash
python -m onebit_bai.experiment \
  --model U \
  --u-generator beta \
  --means 0.90 0.50 0.30 0.20 0.10 \
  --concentration 12.0 \
  --beta 0.5 \
  --xgrid 61 \
  --qquad 301 \
  --refresh 50 \
  --decide 50 \
  --deltas 0.10 0.05 0.02 0.01 \
  --skip-safe-bit \
  --u-boundary certified \
  --reps 200 \
  --cap 200000 \
  --workers <N>
```

Trong final run thực tế, `N=16` vì giới hạn RAM available lúc đó.

---

## 17. Quy tắc ghép data

Được concatenate hai batch chỉ khi **tất cả** các item sau trùng:

```text
model
instance / means
beta
method / algorithm
boundary
delta grid
refresh schedule
decision schedule
cap
oracle/numerical settings
implementation semantics
```

và replication seeds không trùng.

Không được pool:
- practical U với certified U;
- E_Q1 với E_Q3;
- U_Q1 với U_Q3 như independent evidence;
- stopped-only rows với uncensored mean interpretation.

Có thể redraw mọi dataset về cùng một visual style; đó chỉ là presentation.

---

## 18. Figures và vị trí trong paper v46

Main paper hiện ưu tiên:
1. combined relative-efficiency figure;
2. practical calibration E/U.

Supplement chứa:
- Q1 direct first-order;
- Q3 full-vs-one-bit;
- E finite-confidence companion;
- correctness details;
- certified-U stress summary.

Source paper liên quan nằm ở:

```text
paper/main_through_experiments.tex
paper/preview_full.pdf
```

---

## 19. Integrity / handoff checklist

Người nhận nên chạy theo thứ tự:

```bash
python reproduce/audit_handoff.py
pytest -q tests
python reproduce/plot_all_figures.py
```

Sau đó kiểm tra:

- raw row counts khớp `DATA_INVENTORY.csv`;
- U_Q3/U_Q1 reuse relation được báo `[OK]`;
- không có censored row bị drop trong mean calculation;
- figure names khớp source LaTeX;
- population constants không bị lấy nhầm giữa instance cũ và mới.

---

## 20. Một thiếu hụt duy nhất trong archive hiện tại

Raw `trials.csv` của final certified-U stress run chưa được upload vào phiên
chat dùng để tạo package này.  Do đó package chỉ có exact aggregate summary.

Nếu còn quyền truy cập machine chạy experiment, hãy copy original raw file vào:

```text
data/U_certified_easy5_final.csv
```

và giữ nguyên seed/runtime columns.  Không reconstruct/synthesize raw rows.

---

## 21. Tóm tắt cho người tiếp nhận

Nếu chỉ cần hiểu nhanh project:

- Main scientific object: instance-dependent BAI under one-bit feedback.
- Model E: feasible **shared** threshold design; fixed is restrictive,
  pairwise is a theoretical relaxation.
- Model U: current-status observation with raw threshold retained.
- Q1: direct asymptotic normalization.
- Legacy paired benchmark: relative one-bit/full information penalty.
- Q3: empirical full-vs-one-bit stopping comparison.
- Correctness: implementation check, not rare-event certification.
- Certified U: mathematically valid but computationally extremely conservative.
- Never conflate "pairwise design" with "paired Monte Carlo".
- Never pool practical-U and certified-U data.
