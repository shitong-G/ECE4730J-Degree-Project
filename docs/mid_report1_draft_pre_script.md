# Mid Report 1 Draft Presentation Script

Source deck: `mid_report1.pptx`  
Target length: about 10 minutes  
Language: Chinese draft script with English technical terms kept where useful

## Slide 1: Title, about 40 seconds

各位老师、同学大家好，我们今天汇报的题目是 **Dynamic RT-DETR Inference with Scene-Thermal co-Adaption on Raspberry Pi**。

这个项目的背景是，我们希望把 RT-DETR 这样的目标检测模型部署到 Raspberry Pi 这种边缘设备上。但是在实际运行时，Pi 的算力和散热都比较有限，所以如果只是直接运行原始模型，会遇到推理延迟高、持续运行后温度上升、甚至 thermal throttling 的问题。

所以我们的目标不是重新设计一个 detector backbone，而是在 RT-DETR 外面做一个 runtime control system。它可以根据当前场景变化和设备温度，动态调整推理频率、输入分辨率，以及是否使用 tracking 来复用检测结果，从而提升持续运行时的稳定性和输出效果。

## Slide 2: Long-Time Sustained Inference Test, about 1 minute

首先是我们这段时间的第一个进展：重新跑了 long-time sustained inference test，并且明确观察到了 thermal throttling 的影响。

这页图左上角是 inference latency，右上角是 CPU temperature，右下角是 FPS。可以看到，在长时间运行过程中，CPU 温度很快上升并稳定在 85 摄氏度左右。与此同时，推理延迟从一开始大约 3000 ms 逐渐上升到 3700 到 3900 ms 左右。

对应到 FPS，它一开始可以达到 0.32 左右，但后面会慢慢下降到大约 0.26。也就是说，即使模型和输入都不变，只要设备持续高负载运行，热状态本身就会导致性能下降。

这也是后面我们设计 thermal-aware controller 的直接原因：在边缘设备上，不能只看单次推理速度，还要看模型能不能在几分钟甚至更长时间内稳定运行。

## Slide 3: Resolution Configurations, about 50 seconds

第二个进展是，我们准备了三种不同输入分辨率的 inference configuration，分别是 640、480 和 320 pixels。

这一步的作用是为后面的动态控制提供 action space。640 分辨率通常能保留更好的检测质量，但计算量和热压力也更大；480 和 320 分辨率则可以在设备温度较高时作为降级选择，用一定的精度损失换取更低的计算压力。

右边的检测示例展示的是 640 分辨率下的输出效果，可以看到它能够检测到 person、umbrella、boat 等目标。后续我们就以高分辨率结果作为相对可靠的参考，再比较低分辨率或者 tracking 复用时的质量变化。

所以这一页主要说明：我们不是只做一个固定模型的测试，而是把 RT-DETR 包装成一个可以被 runtime controller 调度的系统。

## Slide 4: Thermal-Aware Controller, about 1 minute 10 seconds

接下来是 thermal-aware controller 的设计。

左边对比了 Native RT-DETR 和 Thermal-aware RT-DETR。Native RT-DETR 使用固定设置，也就是固定 640 分辨率、inference interval 等于 1、CPU threads 等于 4。它的问题是简单直接，但是不管温度如何都持续满负载运行。

Thermal-aware RT-DETR 则把这些参数改成动态设置。这里我们主要控制 resolution、inference interval 和 CPU threads。表格里给出了当前使用的规则：当温度低于 83 摄氏度时，保持 640 分辨率、interval 等于 1、4 threads；当温度在 83 到 85 摄氏度之间时，interval 提高到 2；当温度超过 85 摄氏度时，interval 进一步提高到 3。

如果系统已经检测到 thermal throttled 为 true，就说明设备压力比较明显，这时会把 resolution 降到 480，并把 interval 提高到 4。

这个 controller 的核心思想是：先通过降低 detector duty cycle 来控制热量，也就是减少完整推理的频率；只有当设备真的进入 throttling 时，再进一步降低分辨率。这样可以尽量避免过早牺牲检测质量。

## Slide 5: Live Dashboard, about 50 seconds

第五页是我们搭建的 remote live dashboard，用来实时显示系统运行状态。

左边是 live detection stream，可以看到当前画面上的检测框和类别。右边是 runtime state，包括 temperature、latency、actual FPS、resolution、当前 strategy、tracking 状态、thermal state、decision、threads、interval、frequency、ARM clock，以及 soft temperature limit 和 throttled flag。

下面还有 thermal/FPS 曲线以及 latency/ONNX runtime 曲线。这个 dashboard 对实验很重要，因为我们可以在另一台电脑上远程观察 Pi 的状态，不需要每次登录设备查日志。

它也帮助我们调试 controller。例如，当温度变化、tracking 状态切换，或者 interval 变化时，我们可以直接看到系统是否按照预期做出了决策。

## Slide 6: Ablation Study Overview, about 40 seconds

接下来进入 ablation study。我们主要比较三类策略。

第一类是 scene-aware strategy，这里使用的是 Lucas-Kanade optical flow，也就是 LK tracking。它的思路是不用每一帧都运行 RT-DETR，而是在关键帧检测后，用光流跟踪目标框。

第二类是 thermal-aware strategy，也就是刚才介绍的根据温度调整 interval 和 resolution 的控制器。

第三类是 scene-thermal co-adaption strategy，它把前两者结合起来：既根据场景变化决定是否需要 detector refresh，也根据温度状态限制推理负载。

后面几页我会分别展示这三类策略的运行结果。

## Slide 7: Scene-Aware Strategy With LK, about 1 minute

这一页展示的是 scene-aware strategy，也就是 event-based LK tracking 的结果。

左上角是 full inference latency。可以看到完整 RT-DETR 推理仍然需要大约 2800 到 3000 ms，所以 detector 本身并没有突然变快。关键变化在右下角：effective inference FPS 明显提高，可以达到大约 1.5 FPS，而 actual inference FPS 非常低。

这说明大多数输出帧并不是重新跑 detector 得到的，而是通过 LK tracking 传播上一帧的检测框得到的。换句话说，detector 负责周期性刷新和纠错，tracking 负责中间帧的连续输出。

右上角的温度曲线显示，温度整体在 70 多度到 80 多度之间，中间有一次峰值，但之后会回落并保持稳定。左下角的 scene workload 大部分处于 heavy，说明这个策略是在相对复杂的场景下测试的。

这一页的结论是：如果只靠完整 detector，Pi 上的输出帧率会很低；但加入 LK tracking 后，可以显著提高用户看到的输出连续性。

## Slide 8: Thermal-Aware Strategy, about 1 minute

这一页是 thermal-aware strategy 的结果。

右上角可以看到 CPU temperature 从 50 度附近快速上升，然后在 83 到 85 摄氏度附近稳定。相比完全不控制的情况，它能够把系统维持在一个相对可控的热状态。

左上角的 latency 大多数时候在 2500 到 3000 ms 左右，中间会有一些 spike。右下角的 effective inference FPS 和 actual inference FPS 基本接近，大约在 0.35 到 0.4 之间。

这里需要注意的是，thermal-aware strategy 的目标主要是控制温度和避免 throttling，而不是直接提升输出 FPS。因为它没有引入 tracking，每次输出仍然依赖完整 detector inference。所以它可以让设备更稳定，但不能从根本上解决 RT-DETR 单次推理慢的问题。

因此，这一页的结论是：thermal controller 是必要的，但它更像是稳定性模块；如果要提升可见输出频率，还需要 scene-aware tracking。

## Slide 9: Scene-Thermal Co-Adaption Strategy, about 1 minute

第九页是把两者结合起来的 scene-thermal co-adaption strategy。

从右下角可以看到，effective inference FPS 仍然保持在大约 1.5 左右，和单独的 scene-aware LK 策略接近。这说明加入 thermal control 之后，并没有明显破坏 LK tracking 带来的输出连续性。

同时看右上角温度曲线，温度峰值大约在 83 摄氏度附近，之后会回落到 75 度左右。和单独 thermal-aware 策略相比，这里的 detector duty cycle 更低，所以整体热压力也更小。

左上角 latency 仍然表示完整 detector inference 的耗时，大约在 2800 到 3000 ms。它提醒我们：co-adaptation 并不是让 RT-DETR 单次推理变快，而是通过 runtime scheduling 和 tracking reuse，让系统输出更高频、更稳定的检测结果。

所以这一页的核心结论是：scene-aware LK 解决输出连续性，thermal-aware control 解决温度稳定性，两者结合后更适合真实边缘部署。

## Slide 10: Ablation Summary, about 1 minute 10 seconds

这一页是 ablation study 的汇总。

Native RT-DETR 的 detector FPS 和 output FPS 都大约是 0.35 FPS，平均延迟大约 2785 ms，平均温度约 82.7 摄氏度。这说明直接运行原始策略时，性能和热压力都比较紧张。

Thermal-aware interval-first 策略的 output FPS 仍然在 0.34 左右，但平均温度下降到大约 74.6 摄氏度，最高温度约 83.3 摄氏度。它证明 thermal control 对降温有效，但 FPS 提升有限。

Scene-aware event LK 的 detector FPS 只有大约 0.029，但 output FPS 提升到约 1.48，tracking ratio 接近 98%。这说明大量中间帧是通过 LK tracking 复用检测结果输出的。

Co-adaptive 策略的 output FPS 约 1.46，平均温度约 73.5 摄氏度，throttled ratio 也很低。也就是说，它基本保留了 LK 的输出提升，同时维持了更好的热状态。

因此，目前实验支持我们的设计方向：单独 detector 太慢，单独 thermal control 只解决稳定性，而 scene-thermal co-adaption 可以同时兼顾输出连续性和热稳定性。

## Slide 11: Real-Time Camera Input And Fan Control, about 45 seconds

第十一个进展是，我们把 CSI camera 接到了 Raspberry Pi 上，用它来采集实时视频作为系统输入。

前面的部分实验可以基于视频文件运行，这样方便重复测试；但最终系统应该能够处理真实摄像头输入。所以连接 CSI camera 后，我们可以进一步测试 live scene 下的检测、tracking 和 thermal behavior。

同时，我们也加入了 PWM control 来调节风扇速度。这样后续不仅可以被动观察温度，也可以把主动散热纳入 runtime management。比如在温度升高时提高风扇转速，在温度稳定时降低转速，从而在散热、噪声和功耗之间做平衡。

这一页说明系统已经从离线实验逐步转向更完整的硬件闭环。

## Slide 12: Live Camera Demo, about 40 seconds

这一页展示的是使用 CSI camera 后的 live dashboard demo。

可以看到，输入已经不是预先录制的视频，而是实时摄像头画面。系统仍然能够显示检测框，同时右侧继续显示当前 runtime 状态，例如温度、FPS、resolution、tracking 状态、thermal state、governor、interval 和 throttling flag。

这个 demo 的意义是验证整体 pipeline 是打通的：camera input、RT-DETR inference、LK tracking、runtime decision、dashboard visualization 都可以在线运行。

后续我们会基于这个实时输入继续做更多测试，比如不同场景运动速度、不同风扇策略，以及长时间运行稳定性。

## Slide 13: Closing, about 30 seconds

最后总结一下目前的工作。

我们已经完成了长时间推理测试，确认了 thermal throttling 对延迟和 FPS 的影响；准备了 640、480、320 三种分辨率配置；设计了 thermal-aware controller；搭建了 live dashboard；并完成了 scene-aware、thermal-aware 和 scene-thermal co-adaption 的 ablation study。

当前最重要的发现是：thermal-aware control 可以改善温度稳定性，而 LK tracking 可以显著提高输出帧率。把两者结合后，系统可以在 Raspberry Pi 上以更稳定的方式运行 RT-DETR，并提供更连续的检测输出。

接下来我们会继续完善 scene workload 判断、加入更多真实摄像头实验，并进一步优化风扇和 runtime action 的联合控制。

我的汇报到这里，谢谢大家。


