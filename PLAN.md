# RT-DETR on Raspberry Pi 4B：仓库现状分析与下一步开发计划

## 1. 背景与目标

本项目仓库：

```text
https://github.com/shitong-G/ECE4730J-Degree-Project
```

当前项目主要目标是在 Raspberry Pi 4B 上部署 RT-DETR 模型，并根据设备温度与图像复杂度实时调控推理策略，从而在边缘设备上实现更稳定的长时间目标检测。

目前已完成的工作包括：

1. 原始 RT-DETR 模型已在 Raspberry Pi 4B 上完成部署。
2. 已进行初步性能与温度测试。
3. 测试发现当设备温度达到约 85°C 时，CPU 会出现降频，进而导致推理性能下降。
4. 模型耗时分布中，CNN backbone 占比最高，约 50%；encoder 和 decoder 各占约 25%。
5. 已将原始 CNN backbone 从 R18 替换为 R18-lite。
6. 下一步计划是在视频推理过程中动态切换运行策略，并使用设备温度控制策略切换行为。

基于仓库内容，目前项目已经不只是“在树莓派上跑 RT-DETR”，而是已经发展成一个 **scene-thermal co-adaptive runtime system**，即结合场景复杂度和设备热状态的自适应推理系统。

---

## 2. 仓库当前状态总结

从仓库结构和已有代码来看，项目已经具备比较完整的实验框架。

仓库中已经存在以下关键组成部分：

```text
configs/
  raspberry_pi4.yaml

scripts/
  run_experiment.py
  plot_results.py

src/
  scene_runtime/
    controller/
    scene/
    device/
    inference/
    loop/
```

其中，`scripts/run_experiment.py` 已经支持多种运行策略，例如：

```text
native_rtdetr
fixed_low_power
fixed_frame_skip
thermal_only
scene_only
scene_thermal_coadaptive
```

这说明项目已经搭好了基本实验入口，可以直接运行不同 baseline 和 adaptive policy。

主运行循环中也已经包含了以下流程：

```text
读取视频帧
→ 提取场景特征
→ 读取设备状态
→ 分类 runtime state
→ controller 生成 RuntimeAction
→ 根据 action 决定是否推理或跳帧
→ 执行 RT-DETR 推理
→ 写入日志
```

因此，下一步不需要重新设计一套 runtime 框架，而是应该基于现有结构补全缺失逻辑。

---

## 3. 当前最关键的问题

当前仓库的整体架构是合理的，但是有几个核心逻辑仍然是 placeholder 或者尚未真正作用到推理过程。

最重要的问题有三个：

1. `SceneWorkloadEstimator.classify_workload()` 目前没有真正分类场景复杂度。
2. `RuntimeDecisionController._rule_based_action()` 目前没有真正实现 scene-thermal 自适应策略。
3. 部分 `RuntimeAction` 字段目前可能只被记录到日志中，但未必真正影响 ONNX Runtime 推理。

下面分别说明。

---

## 4. 问题一：场景复杂度分类尚未真正实现

### 4.1 当前状态

仓库中的 `SceneWorkloadEstimator` 已经提取了多种轻量级图像复杂度特征，包括：

```text
frame_diff
motion_intensity
edge_density
entropy
prev_detection_count
```

这些特征本身是合理的，适合 Raspberry Pi 4B 这种资源有限的设备。

其中：

- `edge_density` 可以反映画面纹理和边缘复杂度；
- `motion_intensity` 可以反映视频中的运动强度；
- `frame_diff` 可以反映相邻帧变化；
- `entropy` 可以反映画面信息量；
- `prev_detection_count` 可以反映上一帧检测到的目标数量。

但是，当前 `classify_workload()` 并没有真正使用这些特征，而是固定返回：

```python
return "medium"
```

这意味着当前即使运行 `scene_only` 或 `scene_thermal_coadaptive` 策略，场景复杂度部分也不会真正生效。

### 4.2 影响

当前问题会导致：

```text
light / medium / heavy workload 不会真实变化
```

因此：

1. `scene_only` 实验不能体现画面复杂度控制；
2. `scene_thermal_coadaptive` 实验实际退化为 thermal-only 或固定策略；
3. 后续画 workload 曲线时，曲线可能长期保持 `medium`；
4. 报告中很难证明系统真正做到了 scene-aware adaptation。

### 4.3 建议实现

建议先实现一个简单、可解释、低开销的 rule-based workload classifier。

示例代码如下：

```python
def classify_workload(self, features: dict[str, Any]) -> str:
    edge = float(features["edge_density"])
    motion = float(features["motion_intensity"])
    det = int(features["prev_detection_count"])

    if (
        edge <= self._light_edge_max
        and motion <= self._light_motion_max
        and det < self._heavy_det_min // 2
    ):
        return "light"

    if (
        edge >= self._heavy_edge_min
        or motion >= self._heavy_motion_min
        or det >= self._heavy_det_min
    ):
        return "heavy"

    return "medium"
```

这个版本的优点是：

1. 计算开销低；
2. 行为容易解释；
3. 很适合写进实验报告；
4. 后续可以逐步替换成更复杂的 learned policy 或统计模型。

---

## 5. 问题二：RuntimeDecisionController 的 adaptive policy 仍是 placeholder

### 5.1 当前状态

当前 controller 已经有 `RuntimeState` 的概念，也会融合：

```text
workload
thermal_state
```

例如：

```text
workload: light / medium / heavy
thermal_state: normal / warm / hot
```

但是实际生成 action 的 `_rule_based_action()` 目前仍然比较简化，没有真正把 workload 和 thermal state 组合起来使用。

### 5.2 建议设计

建议把 controller 实现成一个 3×3 的 rule table：

| Thermal State | Light Workload | Medium Workload | Heavy Workload |
|---|---|---|---|
| normal | 低功耗运行 | 平衡运行 | 高精度运行 |
| warm | 降低频率和分辨率 | 平衡偏节能 | 保留检测能力但降低开销 |
| hot | 强制降载 | 冷却优先 | 保留最低检测能力 |

具体策略可以设计为：

| Thermal | Workload | Mode | Resolution | Inference Interval | CPU Threads | Governor | Query Budget |
|---|---|---|---:|---:|---:|---|---:|
| normal | light | eco_scene_light | 480 | 2 | 3 | ondemand | 100 |
| normal | medium | balanced | 480 | 1 | 3 | ondemand | 200 |
| normal | heavy | accuracy_heavy | 640 | 1 | 4 | performance | 300 |
| warm | light | warm_eco | 480 | 3 | 2 | ondemand | 100 |
| warm | medium | warm_balanced | 480 | 2 | 3 | ondemand | 150 |
| warm | heavy | warm_heavy | 480 | 1 | 3 | ondemand | 200 |
| hot | light | hot_cooldown_light | 320 | 4 | 2 | powersave | 80 |
| hot | medium | hot_cooldown | 320 | 3 | 2 | powersave | 100 |
| hot | heavy | hot_preserve_detection | 480 | 2 | 2 | powersave | 150 |

### 5.3 建议实现代码

可以在 `RuntimeDecisionController` 中实现如下逻辑：

```python
def _rule_based_action(self, workload: str, thermal: str) -> RuntimeAction:
    if thermal == "unknown":
        return self._balanced("balanced_unknown")

    table = {
        ("normal", "light"):  ("eco_scene_light", 480, 2, 3, "ondemand", 100),
        ("normal", "medium"): ("balanced", 480, 1, 3, "ondemand", 200),
        ("normal", "heavy"):  ("accuracy_heavy", 640, 1, 4, "performance", 300),

        ("warm", "light"):    ("warm_eco", 480, 3, 2, "ondemand", 100),
        ("warm", "medium"):   ("warm_balanced", 480, 2, 3, "ondemand", 150),
        ("warm", "heavy"):    ("warm_heavy", 480, 1, 3, "ondemand", 200),

        ("hot", "light"):     ("hot_cooldown_light", 320, 4, 2, "powersave", 80),
        ("hot", "medium"):    ("hot_cooldown", 320, 3, 2, "powersave", 100),
        ("hot", "heavy"):     ("hot_preserve_detection", 480, 2, 2, "powersave", 150),
    }

    mode, res, interval, threads, governor, q = table.get(
        (thermal, workload),
        (
            "fallback_balanced",
            self._default_res,
            self._default_interval,
            self._default_threads,
            "ondemand",
            200,
        ),
    )

    return RuntimeAction(
        mode=mode,
        input_resolution=res,
        inference_interval=interval,
        cpu_threads=threads,
        governor=governor,
        decoder_layers=None,
        query_budget=q,
    )
```

这个策略的设计原则是：

1. 温度优先级高于场景复杂度；
2. 当设备处于 hot 状态时，优先避免继续升温；
3. 当温度 normal 且画面复杂时，允许使用更高分辨率和更密集推理；
4. 当画面简单时，即使温度正常，也不需要每帧高精度推理；
5. 当温度过高时，即使画面复杂，也要降低开销，只保留基本检测能力。

---

## 6. 问题三：部分 RuntimeAction 字段可能尚未真正生效

当前 `RuntimeAction` 中包含多个字段，例如：

```text
input_resolution
inference_interval
cpu_threads
governor
decoder_layers
query_budget
```

但是需要区分两类字段：

1. 已经真实影响运行过程的字段；
2. 目前只是被写入日志或作为 policy hint 的字段。

---

### 6.1 inference_interval 已经真实生效

`inference_interval` 是当前最可靠的 runtime knob。

主循环中已经根据 `action.inference_interval` 决定是否执行推理：

```text
如果当前帧满足 interval 条件 → 执行 self._engine.infer()
否则 → 跳过推理
```

因此，frame skipping 已经可以真实减少推理次数，降低 CPU 负载和温度。

这是下一阶段实验中最应该优先使用的控制手段。

---

### 6.2 input_resolution 是否生效取决于 ONNX 输入 shape

`input_resolution` 可能存在一个重要风险：

如果当前 ONNX 模型是固定输入尺寸，例如：

```text
[1, 3, 640, 640]
```

那么即使 controller 输出：

```text
input_resolution = 480
```

或者：

```text
input_resolution = 320
```

实际 ONNX 推理也可能仍然使用固定的 640×640 输入。

建议先在 Raspberry Pi 上运行以下脚本检查 ONNX 输入 shape：

```bash
python - <<'PY'
import onnxruntime as ort

sess = ort.InferenceSession("models/rtdetr_r18_lite_pi4.onnx")
for i in sess.get_inputs():
    print(i.name, i.shape)
PY
```

如果输出类似：

```text
images [1, 3, 640, 640]
```

说明模型输入是固定尺寸。

如果输出类似：

```text
images [1, 3, 'height', 'width']
```

或者：

```text
images ['batch', 3, 'height', 'width']
```

说明可能支持动态输入尺寸。

### 6.3 多分辨率 ONNX 比 dynamic shape 更稳妥

对于 Raspberry Pi 4B，建议优先使用多个固定尺寸 ONNX 模型，而不是一开始就使用 dynamic-shape ONNX。

例如：

```text
models/
  rtdetr_r18_lite_320.onnx
  rtdetr_r18_lite_480.onnx
  rtdetr_r18_lite_640.onnx
```

运行时可以提前加载多个 session：

```text
320 session
480 session
640 session
```

然后根据 `RuntimeAction.input_resolution` 选择对应 session。

这种方式的优点是：

1. 每个 ONNX 模型 shape 固定，ONNX Runtime 优化更稳定；
2. 避免 dynamic shape 在 Pi 上引入额外 overhead；
3. 实验报告更容易解释；
4. 不同 resolution 的 latency 差异更清楚。

---

### 6.4 cpu_threads 当前可能尚未动态生效

当前 `RuntimeAction` 中包含：

```text
cpu_threads
```

但是 ONNX Runtime 的线程数通常需要在创建 `InferenceSession` 时通过 `SessionOptions` 设置。

示例：

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = cpu_threads
so.inter_op_num_threads = 1

session = ort.InferenceSession(
    model_path,
    sess_options=so,
    providers=["CPUExecutionProvider"],
)
```

如果当前代码只在日志中记录了 `cpu_threads`，但没有用它创建 session，那么这个字段不会真正改变推理线程数。

同时还要注意：如果 session 在程序启动时只创建一次，那么运行过程中动态切换 `cpu_threads` 并不简单。

更稳妥的做法是：

1. 第一阶段实验只切换 `inference_interval` 和 `input_resolution`；
2. 第二阶段再考虑提前创建多个不同线程数的 ONNX session；
3. 或者固定线程数，只把线程数作为实验配置而不是每帧动态控制项。

---

### 6.5 governor / affinity 当前更适合作为后续扩展

`governor` 和 CPU affinity 也类似。

如果代码中只是生成：

```text
governor = "powersave"
```

或者：

```text
governor = "performance"
```

但没有实际调用系统命令或写入 sysfs，那么它们只是 policy hint，不会真正改变系统状态。

在报告中应避免直接声称：

```text
系统已经动态切换 CPU governor
```

除非已经实现并验证。

更严谨的表述应该是：

```text
The controller emits runtime actions including resolution, frame skipping,
CPU thread hints, and governor hints. In the current implementation,
frame skipping is directly enforced; resolution is enforced when the ONNX
model supports the requested input shape. CPU/governor controls are logged
as policy outputs and can be applied by a future OS-control module.
```

---

## 7. 温度阈值设计建议

当前项目已观测到：

```text
约 85°C 时 Raspberry Pi 4B 出现 CPU 降频
```

因此策略控制不应该等到 85°C 才开始降载，而应该提前进入保护模式。

### 7.1 当前配置

当前 `raspberry_pi4.yaml` 中的 thermal threshold 大致是：

```yaml
thermal:
  normal_max_c: 60.0
  warm_max_c: 72.0
```

这意味着：

```text
temperature <= 60°C       → normal
60°C < temperature <= 72°C → warm
temperature > 72°C         → hot
```

这个配置比较保守。它的优点是可以提前降载，尽量避免接近 85°C；缺点是可能较早牺牲推理精度和 FPS。

### 7.2 建议使用两套 thermal profile

建议实验中使用两套不同阈值配置进行对比。

#### Conservative profile

```yaml
thermal:
  normal_max_c: 60.0
  warm_max_c: 72.0
```

特点：

```text
更早进入 warm/hot
更保守
更容易避免 throttling
可能牺牲部分检测质量和 FPS
```

#### Performance-preserving profile

```yaml
thermal:
  normal_max_c: 68.0
  warm_max_c: 80.0
```

特点：

```text
更晚进入 warm/hot
更重视性能
更接近 85°C throttling 临界点
需要观察是否仍能避免降频
```

建议比较以下指标：

```text
max_temperature_c
mean_temperature_c
time_above_80c
currently_throttled_ratio
arm_clock_mhz
fps
latency_ms
detection_count
confidence
```

这样可以证明：

1. adaptive policy 是否能避免 85°C 降频；
2. 控制策略是否过于保守；
3. 不同温度阈值对 FPS 和检测质量的影响。

---

## 8. 实验设计

建议直接使用仓库已有的策略文件和实验入口，不需要重新写一套实验系统。

### 8.1 实验一：固定策略 baseline

目的：

```text
建立不同固定运行策略下的性能、温度和检测质量基线
```

建议测试：

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy native_rtdetr \
  --video data/sample.mp4 \
  --duration-min 15
```

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy fixed_frame_skip \
  --video data/sample.mp4 \
  --duration-min 15
```

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 15
```

关注指标：

```text
temperature
fps
latency
arm_clock_mhz
currently_throttled
detection_count
```

---

### 8.2 实验二：thermal-only 控制

目的：

```text
验证只使用设备温度是否可以避免 85°C 降频
```

运行：

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy thermal_only \
  --video data/sample.mp4 \
  --duration-min 15
```

预期现象：

```text
温度升高 → controller 降低推理频率或分辨率
温度趋于稳定 → CPU clock 更稳定
相比 native_rtdetr，更少出现 throttling
```

---

### 8.3 实验三：scene-only 控制

目的：

```text
验证图像复杂度能否影响运行策略
```

运行：

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_only \
  --video data/sample.mp4 \
  --duration-min 15
```

注意：这个实验必须先实现 `classify_workload()`，否则 workload 会一直是 `medium`，无法证明 scene-aware adaptation。

建议准备三类视频：

| 视频类型 | 特征 | 预期 workload |
|---|---|---|
| Low complexity | 背景简单、目标少、运动少 | light |
| Medium complexity | 普通道路/室内场景 | medium |
| High complexity | 目标多、运动明显、纹理复杂 | heavy |

---

### 8.4 实验四：scene-thermal coadaptive 控制

目的：

```text
验证同时使用图像复杂度和设备温度，是否能在稳定温度的同时保留必要检测质量
```

运行：

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_coadaptive \
  --video data/sample.mp4 \
  --duration-min 15
```

重点观察：

```text
workload 随视频内容变化
temperature 上升时 action mode 是否切换
hot 状态下 inference_interval 是否增大
FPS 是否比 native_rtdetr 更稳定
ARM clock 是否比 native_rtdetr 更少下降
```

---

## 9. 日志字段建议

当前日志中已经记录了很多关键字段，例如：

```text
frame_id
timestamp
temperature
fps
latency
workload
thermal_state
input_resolution
inference_interval
cpu_threads
governor
query_budget
```

但是建议新增一个非常重要的字段：

```text
action_mode
```

原因是：

```text
strategy 表示实验整体策略，例如 scene_thermal_coadaptive
action_mode 表示 controller 当前帧实际选择的运行模式，例如 hot_cooldown 或 accuracy_heavy
```

如果没有 `action_mode`，那么后续画图时只能看到整段视频都属于：

```text
scene_thermal_coadaptive
```

但看不到每一帧实际切到了什么运行模式。

建议在 log record 中加入：

```python
action_mode=action.mode
```

最终 CSV 中应包含类似字段：

```text
frame_id
timestamp_s
strategy
action_mode
workload
thermal_state
temperature_c
arm_clock_mhz
currently_throttled
input_resolution
inference_interval
cpu_threads
governor
latency_ms
fps
detection_count
```

这样后续图表可以展示：

```text
温度升高 → thermal_state 从 normal 变成 warm/hot
controller action 从 accuracy_heavy 变成 hot_cooldown
inference_interval 增大
FPS 稳定
ARM clock 不再明显下降
```

这会是报告中最有说服力的一组结果。

---

## 10. 画图建议

仓库已有 `scripts/plot_results.py`，建议继续使用它，同时增加一张 action mode 时间线图。

建议最终报告中至少包含以下图表：

### 10.1 Temperature over time

展示不同策略下设备温度变化：

```text
native_rtdetr 是否达到 85°C
thermal_only 是否提前降载
scene_thermal_coadaptive 是否稳定在较低温度
```

### 10.2 FPS over time

展示长期视频推理稳定性：

```text
native_rtdetr 是否随着温度上升而 FPS 下降
adaptive policy 是否维持更稳定 FPS
```

### 10.3 ARM clock over time

展示是否发生真实降频：

```text
arm_clock_mhz 是否在高温时下降
adaptive policy 是否减少 clock drop
```

### 10.4 Runtime action timeline

展示策略切换行为：

```text
normal → warm → hot
balanced → warm_balanced → hot_cooldown
```

### 10.5 Workload timeline

展示场景复杂度分类是否生效：

```text
light / medium / heavy
```

---

## 11. 与已有 profiling 结论的关系

你们之前已经测得：

```text
CNN backbone: 约 50%
encoder: 约 25%
decoder: 约 25%
```

这个结论非常重要，因为它解释了为什么要把 R18 替换成 R18-lite。

但是需要注意：当前 runtime profiler 可能只能分到：

```text
preprocess_ms
build_feed_ms
onnx_run_ms
postprocess_ms
infer_total_ms
```

它未必能直接分出：

```text
backbone_ms
encoder_ms
decoder_ms
```

因此报告中建议分开描述：

1. **Model component profiling**  
   用于说明 RT-DETR 的主要计算瓶颈来自 CNN backbone。

2. **Runtime profiling**  
   用于说明在视频推理时，adaptive policy 如何影响 latency、FPS、温度和降频。

3. **Backbone replacement**  
   R18-lite 是针对 backbone bottleneck 的模型结构优化。

4. **Runtime adaptation**  
   scene-thermal controller 是针对长时间运行稳定性的系统级优化。

这样逻辑会更清楚：

```text
先通过 profiling 找到 backbone 是主要瓶颈
→ 用 R18-lite 降低单帧推理开销
→ 再用 thermal/scene-aware runtime 控制长期运行过程
→ 避免高温降频导致性能崩溃
```

---

## 12. 推荐下一步开发优先级

### Priority 1：实现真正的 workload classification

修改文件：

```text
src/scene_runtime/scene/workload_estimator.py
```

目标：

```text
classify_workload() 不再固定返回 medium
能够根据 edge_density / motion_intensity / detection_count 返回 light / medium / heavy
```

验收标准：

```text
运行不同复杂度视频时，日志中的 workload 会发生变化
plot_results.py 能画出 workload 随时间变化
```

---

### Priority 2：实现 scene-thermal rule-based policy

修改文件：

```text
src/scene_runtime/controller/runtime_controller.py
```

目标：

```text
_rule_based_action() 根据 workload 和 thermal_state 输出不同 RuntimeAction
```

验收标准：

```text
normal/heavy → 更高 resolution、更小 interval
hot/light → 更低 resolution、更大 interval
hot/medium → cooldown policy
```

---

### Priority 3：新增 action_mode 日志字段

修改文件可能包括：

```text
src/scene_runtime/loop/runtime_loop.py
src/scene_runtime/logging/logger.py
```

目标：

```text
CSV 中记录每一帧实际选择的 action.mode
```

验收标准：

```text
CSV 中出现 action_mode 列
可以画出 action mode timeline
```

---

### Priority 4：验证 input_resolution 是否真实生效

在 Raspberry Pi 上运行：

```bash
python - <<'PY'
import onnxruntime as ort

sess = ort.InferenceSession("models/rtdetr_r18_lite_pi4.onnx")
for i in sess.get_inputs():
    print(i.name, i.shape)
PY
```

根据结果决定：

```text
固定 shape → 导出多个 ONNX
动态 shape → 确认不同 input_resolution 下 latency 是否变化
```

---

### Priority 5：完成六组实验

建议固定使用同一段视频、同一环境温度、同一运行时长，测试：

```text
native_rtdetr
fixed_frame_skip
fixed_low_power
thermal_only
scene_only
scene_thermal_coadaptive
```

每组至少运行：

```text
15 minutes
```

如果时间允许，可以运行：

```text
30 minutes
```

这样更容易观察温度稳定性和 throttling 行为。

---

## 13. 建议的下一次 commit

推荐 commit 标题：

```text
Implement scene-thermal adaptive runtime policy
```

建议 commit 内容：

```text
1. Implement workload classification using edge density, motion intensity, and detection count.
2. Implement 3×3 scene-thermal rule-based runtime policy.
3. Add action_mode to runtime CSV logs.
4. Add experiment instructions for native, fixed, thermal-only, scene-only, and coadaptive runs.
5. Add ONNX input shape verification note for multi-resolution inference.
```

---

## 14. 建议的报告贡献点表述

可以在论文或项目报告中将贡献总结为：

```text
This project deploys RT-DETR on Raspberry Pi 4B and investigates its
long-duration runtime behavior under thermal constraints. Initial profiling
shows that the CNN backbone dominates inference latency, motivating the
replacement of the original R18 backbone with an R18-lite variant.

Beyond model-level optimization, the project introduces a scene-thermal
co-adaptive runtime controller. The controller monitors device temperature
and lightweight scene-complexity features, then dynamically adjusts runtime
actions such as input resolution and frame skipping. The goal is to prevent
thermal throttling around 85°C while maintaining acceptable detection
performance for complex scenes.
```

中文版本可以写成：

```text
本项目在 Raspberry Pi 4B 上部署 RT-DETR，并分析其在长时间运行中的热稳定性问题。初步 profiling 显示，CNN backbone 是主要推理瓶颈，约占总耗时的 50%，因此项目将原始 R18 backbone 替换为 R18-lite，以降低单帧推理开销。

在模型结构优化之外，本项目进一步设计了结合图像复杂度和设备温度的自适应运行时控制器。该控制器实时监测设备温度和轻量级场景特征，并动态调整输入分辨率和推理间隔等运行策略，从而在接近 85°C 降频阈值之前主动降低负载，提升 Raspberry Pi 4B 上长时间视频推理的稳定性。
```

---

## 15. 最小闭环实现建议

如果时间有限，建议下一阶段先实现最小闭环：

```text
1. 每帧提取 scene features
2. 每秒读取一次 temperature 和 arm_clock_mhz
3. 根据 thermal_state 和 workload 输出 action
4. 至少真实应用 inference_interval
5. 尽可能真实应用 input_resolution
6. 记录 action_mode / temperature / fps / arm_clock_mhz
7. 画出策略切换和温度变化曲线
```

最小闭环完成后，即使 CPU governor、query budget、decoder layer 动态控制暂时没有实现，也已经可以支撑一个完整的项目故事：

```text
RT-DETR deployment
→ profiling
→ R18-lite model optimization
→ thermal throttling observation
→ scene-thermal adaptive runtime
→ stable long-duration video inference
```

---

## 16. 最终结论

当前仓库的基础结构已经比较完整，不需要重新搭建系统。下一步最重要的是把已经存在的 runtime framework 从 placeholder 变成真正可实验验证的 adaptive system。

具体来说，下一步应优先完成：

```text
SceneWorkloadEstimator.classify_workload()
RuntimeDecisionController._rule_based_action()
action_mode logging
ONNX input resolution verification
six-strategy experiment evaluation
```

完成这些之后，项目就可以比较完整地证明：

```text
1. RT-DETR 可以部署到 Raspberry Pi 4B；
2. R18-lite 可以降低 backbone 计算开销；
3. 高温会导致 Raspberry Pi 4B CPU 降频和 FPS 下降；
4. scene-thermal adaptive runtime 可以提前降载；
5. 自适应策略可以在避免 throttling 的同时尽量保留检测能力。
```
