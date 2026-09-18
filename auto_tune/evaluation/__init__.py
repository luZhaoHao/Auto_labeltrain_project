"""F1.1-B 受控评估支撑包（离线案例集 + 回放运行器 + 质量度量）。

本包只服务 LLM 调优决策质量评估，不参与生产运行路径：它复用生产的
perception / fact package / decision agent 代码，但不被它们导入。
"""
