包含本项目中标准化最小范数完整二次曲面方法的最终代码及相关验证程序。
由于留一法验证表明模型在传感器凸包（sensor convex hull）之外的外推误差较大，
因此最终模型的主要工程分析范围限制在 sensor convex hull 内。凸包外的扩展分析
仅作为前期探索，不用于最终的损伤和疼痛阈值判断。


1. `final_pressure_model.py`

最终压力模型的主程序。

主要内容：

- 对五个传感器的 x、y 坐标进行标准化；
- 使用 minimum-norm 方法拟合六参数完整二次曲面；
- 将标准化坐标下得到的模型转换回原始坐标；
- 验证模型在五个原始传感器位置的拟合结果；
- 计算压力曲面的内部驻点和 Hessian；
- 对 sensor convex hull 的边界进行极值分析。

该文件定义了报告中最终采用的标准化最小范数完整二次压力模型。



 2. `geometry_check.py`

检查五个传感器的空间几何关系。

主要内容：

- 构造并绘制 sensor convex hull；
- 判断各传感器与凸包的位置关系；
- 验证 S2 位于 S1、S3、S4、S5 所形成的凸包内部；
- 为后续区分 interpolation 和 extrapolation 提供几何依据。

因此，在留一验证中：

- 留出 S2 属于 interpolation；
- 留出 S1、S3、S4 或 S5 属于 extrapolation。



 3. `final_model_loocv.py`

对最终二次模型进行 Leave-One-Out Cross-Validation（LOOCV）。

主要内容：

- 分别留出五个传感器进行预测；
- 区分 S2 的内部插值测试与另外四个传感器的外推测试；
- 计算每个留出点的预测误差；
- 计算 interpolation error；
- 计算 extrapolation RMSE 和 MAE；
- 计算 overall LOOCV error。

该程序主要用于评价最终模型的预测能力，并确定模型可靠的空间适用范围。


 4. `scaling_robustness.py`

分析模型对坐标尺度处理方式的敏感性。

主要内容：

- 比较直接使用原始 x、y 坐标得到的 minimum-norm quadratic；
- 比较坐标标准化后得到的 minimum-norm quadratic；
- 比较两种方法对 S2 的留出预测误差；
- 比较两张压力曲面之间的整体差异。

该分析用于说明为什么最终模型采用标准化坐标进行 minimum-norm fitting。



5. `noise_robustness.py`

进行 Monte Carlo 测量噪声稳定性分析。

主要内容：

- 分别加入原始压力范围 2%、5% 和 10% 的 Gaussian noise；
- 在每个噪声水平下重复生成大量随机数据并重新拟合模型；
- 分析 S2 留出预测的均值、标准差和 MAE；
- 分析 sensor convex hull 内的最小和最大压力（由凸包顶点、边上驻点和内部驻点精确求得，不用网格近似）；
- 统计出现 zero/negative pressure 的比例；
- 统计超过 pain threshold 50 的比例；
- 计算相应的 Monte Carlo 95% 区间。

该程序用于评价最终模型及工程结论对 measurement uncertainty 的稳定性。


 6. `part_c_boundary_parabolic.py`

利用 Parabolic Interpolation 对 sensor convex hull 的边界进行优化。

主要内容：

- 将 convex hull 的四条边分别参数化为一维问题；
- 在每条边上搜索内部 stationary point；
- 同时检查每条边的两个 endpoints；
- 比较所有候选点；
- 确定 sensor convex hull 内的 constrained global maximum。

这是报告中确定最终最大压力的主要优化方法。



 7. `part_c_newton.py`

利用 Multidimensional Newton Method 分析压力曲面的内部驻点。

主要内容：

- 计算压力函数的 gradient；
- 计算 Hessian；
- 使用 Newton Method 求内部 stationary point；
- 根据 Hessian 判断驻点性质。

该程序主要用于优化方法比较，并说明仅求解内部 stationary point
不能保证得到 constrained global maximum。


 8. `convex_hull_steepest_ascent.py`

在 sensor convex hull 约束下实现 Projected Steepest Ascent。

主要内容：

- 沿 pressure gradient 方向进行迭代；
- 将超出 convex hull 的 trial point 投影回可行区域；
- 使用 backtracking 调整 step size；
- 比较不同 initial points 对最终结果的影响；
- 比较不同 initial step sizes 对收敛速度的影响；
- 与 Boundary Parabolic Interpolation 得到的参考最大值进行比较。

该程序用于 optimisation method comparison、initial-point sensitivity
和 step-size sensitivity 分析。



 9. `ridge_tradeoff.py`

研究 L2 regularisation 对完整二次模型的影响。

主要内容：

- 测试不同 regularisation parameter；
- 比较 training error；
- 比较 S2 interpolation error；
- 比较 extrapolation RMSE；
- 分析模型拟合能力与预测稳定性之间的 trade-off。

该程序属于模型参数敏感性实验，不是最终采用的压力模型。



10. `simple_test_case.py`

使用具有已知解析解的简单二维二次压力场验证数值程序的准确性。

测试曲面为：

p(x,y) = 40 - 2(x-1)^2 - 3(y-2)^2

该压力场的零点和全局最大值均具有已知解析解，因此可以直接比较
numerical solution 和 analytical solution。

该程序分别验证：

- Bisection Method；
- Newton-Raphson Method；
- Newton optimisation；
- Steepest Ascent；
- Parabolic Interpolation。

需要注意：该 test case 用于验证 numerical programme 的实现是否正确，
并不是用于证明真实 robotic-hand pressure model 的物理准确性。

---

## 最终模型主要结果

最终模型的主要工程解释范围限制在 sensor convex hull 内。

主要结果如下：

- S2 留出预测值约为：17.5737
- S2 插值绝对误差约为：0.5737
- 凸包外 extrapolation RMSE 约为：45.42
- sensor convex hull 内最小压力：2
- sensor convex hull 内最大压力：43
- convex hull 内未预测到 zero/negative pressure
- convex hull 内最大压力未超过 pain threshold 50

因此，现有数据支持的主要结论为：

1. 最终二次模型在唯一可进行真正内部留出验证的 S2 上具有较小的预测误差；
2. 模型在 convex hull 外的外推能力较弱，因此不使用外推结果进行最终工程判断；
3. 在 sensor convex hull 内没有预测到零压力或负压力区域；
4. 在 sensor convex hull 内最大压力为 43，没有超过 pain threshold 50；
5. Monte Carlo 分析表明 pain-threshold 结论对合理范围的测量噪声较稳定，
   而 damage conclusion 对测量误差相对更加敏感。

