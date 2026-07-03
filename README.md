# UUV端侧智能轨迹规划系统 MVP

这是一个本地可运行的 UUV 单艇轨迹规划 MVP，实现两类任务：

- 通用点到点轨迹规划：直线路径检查、A* 栅格避障、路径平滑、约束验证。
- 区域覆盖规划：500m x 500m 等简单多边形区域的往返式扫描覆盖。
- 被动方位滚动规划：根据方位历史、UUV状态和约束，每次输出下一步动作。
- 抵近仿真测试：用真实目标位置生成方位反馈，闭环验证滚动规划行为。

系统按 ReAct 风格组织流程：

```text
Observe -> Reason -> Act -> Feedback
```

当前默认离线运行，不实际调用大模型。代码中保留了云端 LLM 接口；如果设置 `OPENAI_API_KEY` 并安装 `openai` SDK，会尝试调用配置模型。没有 Key、SDK 或调用失败时，会走本地规则解释，不使用任何模型。

## 快速运行

在项目目录运行：

```bash
python3 main.py --input examples/general_scenario.json --pretty
python3 main.py --input examples/coverage_scenario.json --pretty
```

也可以使用内置示例：

```bash
python3 main.py --scenario general --pretty
python3 main.py --scenario area_coverage --pretty
```

## 网页入口

在项目目录启动本地网页：

```bash
python3 web.py
```

然后打开：

```text
http://127.0.0.1:8000
```

网页支持：

- 粘贴 UUV 探测语义，解析为结构化态势输入并直接规划
- 通过对话描述任务
- 切换点到点和区域覆盖示例
- 手动添加、删除、修改障碍物的位置和半径
- 手动添加、删除、修改饵物的位置和逼近半径
- 饵物坐标若落入障碍物安全距离内，会在界面提示并阻止规划
- 设置全局避障安全距离
- 运行规划
- 直接查看轨迹动图、关键指标和推理链

## 运行测试

```bash
python3 -m unittest discover -s tests
```

测试覆盖核心验收场景：

- 无障碍物简单路径
- 单障碍物避障
- 多障碍物复杂避障
- 动态障碍物预测避障
- 矩形区域全覆盖扫描
- 方位滚动规划的航向修正、抵近环绕、低电量放弃和观测不足等待

## 滚动规划模块

`uuv_trajectory_planner.core.rolling_planner` 面向被动方位历程输入，不一次性生成完整轨迹，而是输出下一步决策：

```python
import json

from uuv_trajectory_planner.core.rolling_planner import plan_rolling

with open("examples/rolling_scenario.json", "r", encoding="utf-8") as handle:
    payload = json.load(handle)

decision = plan_rolling(payload)
print(json.dumps(decision, ensure_ascii=False, indent=2))
```

## 仿真测试模块

`uuv_trajectory_planner.simulation` 提供真实目标位置下的闭环仿真。主流程是输入一个或多个真实饵物坐标，再输入当前方位对话；仿真器根据真实坐标生成后续测角反馈，滚动规划器逐轮调整航向，发现目标后绕航 5 圈。预置单场景和批量测试也保留用于回归验证：

```python
from uuv_trajectory_planner.simulation import SimulationRunner, scenario_by_name

runner = SimulationRunner()
closed_loop = runner.run_interactive(
    {
        "target_positions_text": "(800,300,-50)\n(600,-400,-50)",
        "bearing_text": "当前方位北偏东70度，抵近侦察",
        "orbit_turns": 5,
    }
)
result = runner.run(scenario_by_name("右前方"))
report = runner.run_batch()
```

测试输出位于 `test_outputs/`：

- `*.png`：轨迹可视化图片
- `*_report.json`：约束满足报告和耗时统计

## 目录结构

```text
uuv_project/
├── main.py
├── web.py
├── config/
│   └── default.yaml
├── examples/
│   ├── general_scenario.json
│   └── coverage_scenario.json
├── tests/
├── test_outputs/
└── uuv_trajectory_planner/
    ├── core/
    │   ├── llm_client.py
    │   ├── memory_manager.py
    │   ├── rolling_planner.py
    │   └── react_engine.py
    ├── models/
    ├── planners/
    │   ├── general_planner.py
    │   ├── coverage_planner.py
    │   └── utils.py
    ├── simulation/
    │   ├── simulator.py
    │   ├── runner.py
    │   ├── scenarios.py
    │   └── visualization.py
    ├── web/
    ├── web_server.py
    ├── reporting.py
    └── visualization.py
```

## 关键说明

- 轨迹由本地确定性算法生成，不由大模型直接生成。
- `llm_client.py` 中的降级模式只是规则文本生成，不使用 GPT 或本地大模型。
- 当前版本主要面向 MVP 验证，后续可以接入真实 OpenAI API、强化约束模型、增加 Web 可视化界面。
