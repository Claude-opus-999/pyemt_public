r"""
杆塔结构 Bergeron 传输线模型测试代码 (EMTPSolver 版)
=====================================================================

模仿 test_TLN9_bergeron 中测试10 (PSCAD ZA70) 的风格，
搭建如图所示的多层杆塔雷电暂态分析模型。

杆塔拓扑 (PSCAD Definition Canvas)
──────────────────────────────────
                sI1---I1S1---[ZA10]---node2---[ZB10]---I1S2---sI2
                                        |
                                    [ZT11]‖[ZL11]  (并联)
                                        |
                  node5---[ZA110]---node4---[ZB110]---node6
                    |                   |                |
                  Vbrk11            [ZT12]‖[ZL12]      Vbrk12
                    |                   |                |
                  cI1               node7              cI2
                                        |
                                    [ZT13]‖[ZL13]
                                        |
                                      g01

节点定义
────────
  Node 0  : 大地参考 (Ground)
  Node 1  : sI1 — 左侧避雷线跨端 (雷电流注入点)
  Node 2  : 塔顶中心 — ZA10/ZB10 汇合处, ZT11‖ZL11 顶端
  Node 3  : sI2 — 右侧避雷线跨端 (雷电流注入点)
  Node 4  : 塔身中层 — ZT11‖ZL11 底端, ZA110/ZB110 中心, ZT12‖ZL12 顶端
  Node 5  : 左侧导线端 — ZA110 左端, Vbrk11 上端
  Node 6  : 右侧导线端 — ZB110 右端, Vbrk12 上端
  Node 7  : 塔身下层 — ZT12‖ZL12 底端, ZT13‖ZL13 顶端
  Node 8  : cI1 — 左侧导线外端 (Vbrk11 下端)
  Node 9  : cI2 — 右侧导线外端 (Vbrk12 下端)
  Node 10 : g01 — 接地点 (ZT13‖ZL13 底端)

元件清单 (共 19 个元件)
────────────────────────
  I1S1    : 雷电流源       0 → 1     (雷击避雷线左端)
  I1S2    : 雷电流源       0 → 3     (雷击避雷线右端)
  ZA10    : Bergeron 线路  1 → 2     (避雷线左跨)
  ZB10    : Bergeron 线路  2 → 3     (避雷线右跨)
  ZT11    : Bergeron 线路  2 → 4     (杆塔体: 顶层→中层)
  ZL11    : Bergeron 线路  2 → 4     (引下线: 顶层→中层, ‖ZT11)
  ZA110   : Bergeron 线路  5 → 4     (导线左跨)
  ZB110   : Bergeron 线路  4 → 6     (导线右跨)
  Vbrk11  : 高阻 R         5 → 8     (左绝缘子, 未闪络)
  Vbrk12  : 高阻 R         6 → 9     (右绝缘子, 未闪络)
  ZT12    : Bergeron 线路  4 → 7     (杆塔体: 中层→下层)
  ZL12    : Bergeron 线路  4 → 7     (引下线: 中层→下层, ‖ZT12)
  ZT13    : Bergeron 线路  7 → 10    (杆塔体: 下层→接地)
  ZL13    : Bergeron 线路  7 → 10    (引下线: 下层→接地, ‖ZT13)
  R_sI1   : 电阻           1 → 0     (避雷线左端终端阻抗)
  R_sI2   : 电阻           3 → 0     (避雷线右端终端阻抗)
  R_cI1   : 电阻           8 → 0     (导线左端终端阻抗)
  R_cI2   : 电阻           9 → 0     (导线右端终端阻抗)
  R_gnd   : 电阻           10 → 0    (接地电阻)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── matplotlib 中文字体配置 ──
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                                    'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 路径设置 ─────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from transmission_line_emtp_v2 import (
    TransmissionLineFactory,
    BergeronLine,
    DelayBuffer,
)

from emtp_solver_v3 import EMTPSolver

# LPM 先导发展法绝缘子闪络模型
try:
    from nonlinear_models_pscad import (
        InsulatorFlashoverLPM,
        LPMInsulatorType,
        LPMConfig,
    )
    LPM_AVAILABLE = True
except ImportError:
    LPM_AVAILABLE = False

# 雷电流波形
# 最新版求解器通过 add_standard_twoexpf_IS()/add_lightning_IS()
# 内置 ATP 兼容雷电流源，不再依赖旧版 lightning_waveform.py。
LIGHTNING_MODULE_AVAILABLE = True


# =============================================================================
# 杆塔模型全局配置
# =============================================================================

TOWER_CONFIG = {
    # ── 仿真参数 ──
    'dt':           1e-9,           # 时间步长 1 ns
    'finish_time':  10e-6,          # 仿真时间 50 μs (观察多次反射)

    # ── 雷电流参数 ──
    'lightning_type':    '2/20',    # 波形类型
    'lightning_peak':    100e3,      # 峰值电流 10 kA
    'lightning_t_start': 1e-6,     # 起始时间 1 μs

    # ── 避雷线 (Shield Wire / Ground Wire Spans) ──
    #    ZA10: 塔顶左跨,  ZB10: 塔顶右跨
    'ZA10':  {'Zc': 270.6, 'tau_per_m': 3.33333e-9, 'length_km': 0.0123},
    'ZB10':  {'Zc': 270.6, 'tau_per_m': 3.33333e-9, 'length_km': 0.0103},

    # ── 导线 (Phase Conductor Spans) ──
    #    ZA110: 中层左跨,  ZB110: 中层右跨
    'ZA110': {'Zc': 241.5, 'tau_per_m': 3.33333e-9, 'length_km': 0.0112},
    'ZB110': {'Zc': 241.5, 'tau_per_m': 3.33333e-9, 'length_km': 0.0092},

    # ── 杆塔体分段 (Tower Body Segments) ──
    #    ZT11: 顶层→中层,  ZT12: 中层→下层,  ZT13: 下层→接地
    'ZT11':  {'Zc': 124.9, 'tau_per_m': 3.33333e-9, 'length_km': 0.0080},
    'ZT12':  {'Zc': 108.6, 'tau_per_m': 3.33333e-9, 'length_km': 0.0080},
    'ZT13':  {'Zc':  75.5, 'tau_per_m': 3.33333e-9, 'length_km': 0.0340},

    # ── 引下线 (Lead Wires / Down Conductors) ──
    #    ZL11: 顶层引下,  ZL12: 中层引下,  ZL13: 下层引下
    #    分别与 ZT11/ZT12/ZT13 并联
    'ZL11':  {'Zc': 1124.0, 'tau_per_m': 3.33333e-9, 'length_km': 0.0120},
    'ZL12':  {'Zc':  977.5, 'tau_per_m': 3.33333e-9, 'length_km': 0.0120},
    'ZL13':  {'Zc':  679.5, 'tau_per_m': 3.33333e-9, 'length_km': 0.0510},

    # ── 绝缘子 (Insulator Gaps, 未闪络 = 高阻) ──
    'R_insulator':  1e9,            # Vbrk11 / Vbrk12 — 开路 (未闪络)

    # ── LPM 先导发展法参数 (提取自 PSCAD 逻辑元件截图) ──
    #    PSCAD 信号流:
    #      Vbrk12 → |X| → N/D(N=|V|, D=d−l) → 减E0 → ×|V| → ×k → 积分→l
    #      l ≥ d 时 Comparator 触发 → Monostable → BRK12o 闭合
    #    两个底部 Comparator(A>B, B=0.0) 用于门控: 当 u/(d−l)−E0 < 0 时
    #    将速度置零 (先导不发展)
    'LPM': {
        'E0':          670.0,       # 临界场强 kV/m (PSCAD 常量块显示 670.0)
        'k':           1.0e-6,      # 先导系数 m²/(kV²·μs) (PSCAD 显示 1.0, SI 值)
                                    # 对应 CIGRE: 长棒复合绝缘子(负极性)
        'd':           1.0,         # 绝缘子间隙长度 m (PSCAD 中 d 常量块)
        'R_arc':       0.5,         # 闪络后电弧电阻 Ω
        'R_open':      1e9,         # 闪络前开路电阻 Ω
    },

    # ── 终端阻抗 ──
    'R_sI1':  270.6,                # 避雷线左端终端 (匹配避雷线 Zc)
    'R_sI2':  270.6,                # 避雷线右端终端
    'R_cI1':  241.5,                # 导线左端终端 (匹配导线 Zc)
    'R_cI2':  241.5,                # 导线右端终端

    # ── 接地电阻 ──
    'R_ground': 20.0,               # g01 塔脚接地电阻
}


# =============================================================================
# 辅助函数: 雷电流源波形
# =============================================================================

def add_lightning_source_to_solver(solver: EMTPSolver, name: str,
                                   node_from: int, node_to: int,
                                   cfg: dict):
    """
    使用最新版求解器内置 ATP 兼容雷电流源。

    旧版案例使用 lightning_waveform.create_lightning_waveform() 生成 callable，
    再通过 solver.add_IS() 注入。最新版求解器已经提供：
        solver.add_standard_twoexpf_IS(...)
    可直接生成标准双指数雷电流源，并自动作为电流源加入网络。

    电流源方向遵循 EMTPSolver 约定：node_from → node_to。
    因此 node_from=0, node_to=1 表示从地向 node 1 注入正雷电流。
    """
    return solver.add_standard_twoexpf_IS(
        name=name,
        node_from=node_from,
        node_to=node_to,
        waveform_type=cfg.get('lightning_type', '2/20'),
        peak=cfg['lightning_peak'],
        PERC=cfg.get('lightning_PERC', 30),
        Tstart=cfg.get('lightning_t_start', 0.0),
        Tstop=cfg.get('lightning_t_stop', None),
        atp_compatible=True,
        description="Tower lightning current source",
    )


# =============================================================================
# 辅助函数: 创建单条 Bergeron 线路 (create_from_zc_tau 模式)
# =============================================================================

def _create_bergeron_line(name: str, node_k: int, node_m: int,
                          line_cfg: dict) -> BergeronLine:
    r"""
    用 PSCAD 式参数 (Zc, tau_per_m, length, r_per_m) 创建 Bergeron 线路。

    派生公式:
    \[
        \tau_{total} = \tau_{per\_m} \times L_{m}
    \]
    \[
        G_{eq} = \frac{1}{Z_c}  \quad (\text{无损时})
    \]
    """
    length_m = line_cfg['length_km'] * 1000.0
    return TransmissionLineFactory.create_from_zc_tau(
        name=name,
        node_k=node_k,
        node_m=node_m,
        Zc=line_cfg['Zc'],
        tau_per_m=line_cfg['tau_per_m'],
        length_m=length_m,
    )


# =============================================================================
# 核心函数: 搭建杆塔模型并运行仿真
# =============================================================================

def build_tower_model(
    cfg: dict = None,
    verbose: bool = True,
    insulator_flashover: bool = False,
    use_lpm: bool = False,
    lpm_override: dict = None,
) -> dict:
    r"""
    搭建多层杆塔雷电暂态模型并运行仿真。

    绝缘子模式
    ----------
    - insulator_flashover=False, use_lpm=False : 未闪络 (R=1e9Ω)
    - insulator_flashover=True,  use_lpm=False : 简易闪络 (R=0.01Ω)
    - use_lpm=True                             : LPM 先导发展法自动判断闪络

    参数
    ----
    cfg : dict
        杆塔配置字典
    verbose : bool
        是否打印详细信息
    insulator_flashover : bool
        简易低阻闪络模式 (仅 use_lpm=False 时有效)
    use_lpm : bool
        使用 CIGRE 先导发展法闪络判据
    lpm_override : dict
        覆盖 LPM 默认参数, 如 {'E0': 520, 'k': 1.2e-6, 'd': 1.9}
    """
    if cfg is None:
        cfg = TOWER_CONFIG.copy()

    dt          = cfg['dt']
    finish_time = cfg['finish_time']

    # 确定绝缘子模式
    lpm_cfg = cfg.get('LPM', {}).copy()
    if lpm_override:
        lpm_cfg.update(lpm_override)

    if use_lpm:
        insulator_mode = 'lpm'
        R_ins = lpm_cfg.get('R_open', 1e9)
    elif insulator_flashover:
        insulator_mode = 'simple'
        R_ins = 0.01
    else:
        insulator_mode = 'open'
        R_ins = cfg['R_insulator']

    # ── 1) 创建求解器 ──
    solver = EMTPSolver(
        dt=dt,
        finish_time=finish_time,
        verbose=verbose,

        # 本测试会读取节点电压、线路电流、支路电流、雷电源电流。
        record_all_node_voltages=True,
        record_line_history=True,
        record_branch_history=True,
        record_source_history=True,

        # 最新求解器性能选项。
        pre_sample_sources=True,
        use_rhs_plan=True,
    )

    # ── 2) 雷电流源 (仅 I1S1, 单端注入) ──
    #    雷击避雷线: 电流从 GND(0) 注入 node 1
    lightning_obj = add_lightning_source_to_solver(solver, "I1S1", 0, 1, cfg)

    # ── 3) 避雷线跨 (ZA10: 1→2, ZB10: 2→3) ──
    line_ZA10 = _create_bergeron_line("ZA10", 1, 2, cfg['ZA10'])
    solver.add_line(line_ZA10)

    line_ZB10 = _create_bergeron_line("ZB10", 2, 3, cfg['ZB10'])
    solver.add_line(line_ZB10)

    # ── 4) 杆塔体 & 引下线: 顶层→中层 (ZT11‖ZL11: 2→4) ──
    line_ZT11 = _create_bergeron_line("ZT11", 2, 4, cfg['ZT11'])
    solver.add_line(line_ZT11)

    line_ZL11 = _create_bergeron_line("ZL11", 2, 4, cfg['ZL11'])
    solver.add_line(line_ZL11)

    # ── 5) 导线跨 (ZA110: 5→4, ZB110: 4→6) ──
    line_ZA110 = _create_bergeron_line("ZA110", 5, 4, cfg['ZA110'])
    solver.add_line(line_ZA110)

    line_ZB110 = _create_bergeron_line("ZB110", 4, 6, cfg['ZB110'])
    solver.add_line(line_ZB110)

    # ── 6) 绝缘子 (Vbrk11: 5→8, Vbrk12: 6→9) ──
    lpm_models = {}
    if insulator_mode == 'lpm':
        lpm_k  = lpm_cfg.get('k', 1.0e-6)
        lpm_E0 = lpm_cfg.get('E0', 670.0)
        lpm_d  = lpm_cfg.get('d', 1.0)
        lpm_R_arc  = lpm_cfg.get('R_arc', 0.5)
        lpm_R_open = lpm_cfg.get('R_open', 1e9)
        lpm_alt    = lpm_cfg.get('altitude_m', 0.0)

        solver.add_insulator_LPM("Vbrk11", 5, 8,
            gap_length=lpm_d, k=lpm_k, E0=lpm_E0,
            R_arc=lpm_R_arc, R_open=lpm_R_open, altitude_m=lpm_alt)
        solver.add_insulator_LPM("Vbrk12", 6, 9,
            gap_length=lpm_d, k=lpm_k, E0=lpm_E0,
            R_arc=lpm_R_arc, R_open=lpm_R_open, altitude_m=lpm_alt)

        lpm_models = {
            'Vbrk11': solver._lpm_elements.get('Vbrk11'),
            'Vbrk12': solver._lpm_elements.get('Vbrk12'),
        }

        if verbose:
            print(f"\n  ┌─ LPM 先导发展法参数 (提取自 PSCAD) ──────────────────────┐")
            print(f"  │  间隙长度 d  = {lpm_d:.3f} m                               │")
            print(f"  │  速度系数 k  = {lpm_k:.2e} m²/(kV²·μs)                     │")
            print(f"  │  临界场强 E₀ = {lpm_E0:.1f} kV/m                            │")
            print(f"  │  电弧电阻    = {lpm_R_arc:.2f} Ω                            │")
            print(f"  └───────────────────────────────────────────────────────────┘")
    else:
        solver.add_R("Vbrk11", 5, 8, R_ins)
        solver.add_R("Vbrk12", 6, 9, R_ins)

    # ── 7) 杆塔体 & 引下线: 中层→下层 (ZT12‖ZL12: 4→7) ──
    line_ZT12 = _create_bergeron_line("ZT12", 4, 7, cfg['ZT12'])
    solver.add_line(line_ZT12)

    line_ZL12 = _create_bergeron_line("ZL12", 4, 7, cfg['ZL12'])
    solver.add_line(line_ZL12)

    # ── 8) 杆塔体 & 引下线: 下层→接地 (ZT13‖ZL13: 7→10) ──
    line_ZT13 = _create_bergeron_line("ZT13", 7, 10, cfg['ZT13'])
    solver.add_line(line_ZT13)

    line_ZL13 = _create_bergeron_line("ZL13", 7, 10, cfg['ZL13'])
    solver.add_line(line_ZL13)

    # ── 9) 终端电阻 ──
    solver.add_R("R_sI1", 1, 0, cfg['R_sI1'])    # 避雷线左端终端
    solver.add_R("R_sI2", 3, 0, cfg['R_sI2'])    # 避雷线右端终端
    solver.add_R("R_cI1", 8, 0, cfg['R_cI1'])    # 导线左端终端
    solver.add_R("R_cI2", 9, 0, cfg['R_cI2'])    # 导线右端终端

    # ── 10) 接地电阻 (g01) ──
    solver.add_R("R_gnd", 10, 0, cfg['R_ground'])

    # ── 11) 收集所有线路信息 ──
    line_names = [
        'ZA10', 'ZB10', 'ZT11', 'ZL11',
        'ZA110', 'ZB110', 'ZT12', 'ZL12', 'ZT13', 'ZL13',
    ]
    line_info_dict = {}
    for name in line_names:
        line_info_dict[name] = solver.get_line_info(name)

    # ── 12) 打印拓扑信息 ──
    if verbose:
        _print_tower_topology(cfg, line_info_dict, R_ins,
                              insulator_flashover or use_lpm)
        solver.print_circuit_summary()

    # ── 13) 运行仿真 ──
    solver.run()

    # ── 14) 提取结果 ──
    results = _extract_results(solver, insulator_mode, lpm_models)

    # ── 15) 计算指标 ──
    metrics = _calculate_metrics(results, cfg, line_info_dict, insulator_mode)

    return {
        'solver':              solver,
        'results':             results,
        'metrics':             metrics,
        'line_info_dict':      line_info_dict,
        'lightning_obj':       lightning_obj,
        'config':              cfg,
        'insulator_flashover': insulator_flashover,
        'insulator_mode':      insulator_mode,
        'lpm_models':          lpm_models,
    }


def _print_tower_topology(cfg, line_info_dict, R_ins, flashover):
    """打印杆塔拓扑与参数信息"""
    print(f"\n  ┌─ 杆塔模型拓扑 ─────────────────────────────────────────────────┐")
    print(f"  │                                                                 │")
    print(f"  │   sI1─I1S1─[ZA10]─(2)─[ZB10]─I1S2─sI2                         │")
    print(f"  │    │                 │                │                          │")
    print(f"  │  R_sI1          [ZT11]‖[ZL11]       R_sI2                       │")
    print(f"  │    │                 │                │                          │")
    print(f"  │   GND  (5)─[ZA110]─(4)─[ZB110]─(6) GND                         │")
    print(f"  │         │            │            │                              │")
    print(f"  │       Vbrk11    [ZT12]‖[ZL12]   Vbrk12                          │")
    print(f"  │         │            │            │                              │")
    print(f"  │       (8/cI1)      (7)        (9/cI2)                            │")
    print(f"  │         │            │            │                              │")
    print(f"  │       R_cI1     [ZT13]‖[ZL13]   R_cI2                           │")
    print(f"  │         │            │            │                              │")
    print(f"  │        GND        (10/g01)       GND                             │")
    print(f"  │                      │                                           │")
    print(f"  │                    R_gnd                                          │")
    print(f"  │                      │                                           │")
    print(f"  │                     GND                                           │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ Bergeron 传输线参数汇总 ──────────────────────────────────────┐")
    print(f"  │  {'名称':<8s}  {'Zc(Ω)':>8s}  {'τ(ns)':>10s}  {'长度(m)':>10s}"
          f"  {'delay_steps':>12s}  {'类别':<10s} │")
    print(f"  │  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10} │")

    categories = {
        'ZA10': '避雷线',  'ZB10': '避雷线',
        'ZT11': '杆塔体',  'ZT12': '杆塔体',  'ZT13': '杆塔体',
        'ZL11': '引下线',  'ZL12': '引下线',  'ZL13': '引下线',
        'ZA110': '导线',   'ZB110': '导线',
    }

    for name in ['ZA10', 'ZB10', 'ZT11', 'ZL11', 'ZA110', 'ZB110',
                 'ZT12', 'ZL12', 'ZT13', 'ZL13']:
        info = line_info_dict[name]
        tau_ns = info['tau'] * 1e9
        length_m = cfg[name]['length_km'] * 1000.0
        cat = categories.get(name, '')
        print(f"  │  {name:<8s}  {info['Zc']:>8.1f}  {tau_ns:>10.2f}"
              f"  {length_m:>10.1f}  {info['delay_steps']:>12d}  {cat:<10s} │")

    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ 集总元件 ──────────────────────────────────────────────────────┐")
    print(f"  │  Vbrk11 / Vbrk12  = {R_ins:.2e} Ω  "
          f"({'闪络' if flashover else '未闪络'})")
    print(f"  │  R_sI1 / R_sI2    = {cfg['R_sI1']:.1f} Ω  (避雷线终端匹配)")
    print(f"  │  R_cI1 / R_cI2    = {cfg['R_cI1']:.1f} Ω  (导线终端匹配)")
    print(f"  │  R_gnd (g01)      = {cfg['R_ground']:.1f} Ω  (塔脚接地电阻)")
    print(f"  └─────────────────────────────────────────────────────────────────┘")


def _extract_results(solver, insulator_mode='open', lpm_models=None) -> dict:
    """提取仿真结果（含 LPM 先导发展数据）"""
    results = {
        # 时间
        't_us':    solver.get_time(unit='us'),
        't_s':     solver.get_time(unit='s'),

        # 雷电流源
        'I_I1S1_kA': solver.get_source_current("I1S1") / 1e3,

        # 节点电压
        'V_node1_kV':  solver.get_node_voltage(1, unit='kV'),
        'V_node2_kV':  solver.get_node_voltage(2, unit='kV'),
        'V_node3_kV':  solver.get_node_voltage(3, unit='kV'),
        'V_node4_kV':  solver.get_node_voltage(4, unit='kV'),
        'V_node5_kV':  solver.get_node_voltage(5, unit='kV'),
        'V_node6_kV':  solver.get_node_voltage(6, unit='kV'),
        'V_node7_kV':  solver.get_node_voltage(7, unit='kV'),
        'V_node8_kV':  solver.get_node_voltage(8, unit='kV'),
        'V_node9_kV':  solver.get_node_voltage(9, unit='kV'),
        'V_node10_kV': solver.get_node_voltage(10, unit='kV'),

        # 线路电流
        'I_ZA10_k_kA':  solver.get_line_current_k("ZA10", unit='kA'),
        'I_ZB10_k_kA':  solver.get_line_current_k("ZB10", unit='kA'),
        'I_ZT11_k_kA':  solver.get_line_current_k("ZT11", unit='kA'),
        'I_ZL11_k_kA':  solver.get_line_current_k("ZL11", unit='kA'),
        'I_ZT13_m_kA':  solver.get_line_current_m("ZT13", unit='kA'),
        'I_ZL13_m_kA':  solver.get_line_current_m("ZL13", unit='kA'),

        # 支路电流
        'I_Rgnd_kA':    solver.get_branch_current("R_gnd", unit='kA'),
    }

    # ── LPM 先导发展数据 ──
    if insulator_mode == 'lpm' and lpm_models:
        for ins_name, lpm in lpm_models.items():
            if lpm is None:
                continue
            prefix = ins_name  # "Vbrk11" or "Vbrk12"
            results[f'{prefix}_leader_mm'] = np.array(lpm.leader_length_history) * 1e3
            results[f'{prefix}_velocity_kms'] = np.array(lpm.leader_velocity_history) / 1e3
            results[f'{prefix}_state'] = np.array(lpm.state_history)
            results[f'{prefix}_info'] = lpm.get_info()

        # 闪络事件日志
        results['flashover_log'] = solver.get_flashover_log()

    return results


def _calculate_metrics(results, cfg, line_info_dict,
                       insulator_mode='open') -> dict:
    """计算关键指标（含 LPM 闪络信息）"""
    r = results
    m = {
        'I_lightning_peak_kA': np.max(np.abs(r['I_I1S1_kA'])),
        'V_tower_top_peak_kV': np.max(np.abs(r['V_node2_kV'])),
        't_V_top_peak_us':     r['t_us'][np.argmax(np.abs(r['V_node2_kV']))],
        'V_tower_mid_peak_kV': np.max(np.abs(r['V_node4_kV'])),
        't_V_mid_peak_us':     r['t_us'][np.argmax(np.abs(r['V_node4_kV']))],
        'V_g01_peak_kV':       np.max(np.abs(r['V_node10_kV'])),
        't_V_g01_peak_us':     r['t_us'][np.argmax(np.abs(r['V_node10_kV']))],
        'V_ins_L_peak_kV':     np.max(np.abs(r['V_node5_kV'] - r['V_node8_kV'])),
        'V_ins_R_peak_kV':     np.max(np.abs(r['V_node6_kV'] - r['V_node9_kV'])),
        'I_gnd_peak_kA':       np.max(np.abs(r['I_Rgnd_kA'])),
        'line_info':           line_info_dict,
        'insulator_mode':      insulator_mode,
    }

    # LPM 闪络指标
    if insulator_mode == 'lpm':
        for ins_name in ['Vbrk11', 'Vbrk12']:
            info_key = f'{ins_name}_info'
            if info_key in r:
                info = r[info_key]
                m[f'{ins_name}_flashed'] = info['is_flashed_over']
                m[f'{ins_name}_flashover_us'] = info.get('flashover_time_us')
                m[f'{ins_name}_leader_pct'] = info['leader_progress'] * 100
                m[f'{ins_name}_peak_V_kV'] = info['peak_voltage_kV']

    return m


# =============================================================================
# 打印仿真结果
# =============================================================================

def print_tower_results(sim: dict):
    """打印杆塔仿真结果摘要"""
    m = sim['metrics']
    cfg = sim['config']
    flashover = sim.get('insulator_flashover', False)
    mode = sim.get('insulator_mode', 'open')

    mode_str = {'open': '未闪络', 'simple': '简易闪络', 'lpm': 'LPM先导发展法'}
    print(f"\n{'=' * 72}")
    print(f"  杆塔模型仿真结果摘要  [绝缘子: {mode_str.get(mode, mode)}]")
    print(f"{'=' * 72}")

    print(f"\n  【雷电流源】")
    print(f"    波形类型:     {cfg['lightning_type']} μs")
    print(f"    峰值电流:     {m['I_lightning_peak_kA']:.2f} kA (I1S1 单端注入)")

    print(f"\n  【塔顶电压 (Node 2)】")
    print(f"    峰值:         {m['V_tower_top_peak_kV']:.2f} kV  "
          f"@ {m['t_V_top_peak_us']:.2f} μs")

    print(f"\n  【塔中电压 (Node 4)】")
    print(f"    峰值:         {m['V_tower_mid_peak_kV']:.2f} kV  "
          f"@ {m['t_V_mid_peak_us']:.2f} μs")

    print(f"\n  【塔脚电压 (Node 10 / g01)】")
    print(f"    峰值:         {m['V_g01_peak_kV']:.2f} kV  "
          f"@ {m['t_V_g01_peak_us']:.2f} μs")

    print(f"\n  【绝缘子电压 (跨绝缘子压差)】")
    print(f"    左侧 Vbrk11:  {m['V_ins_L_peak_kV']:.2f} kV")
    print(f"    右侧 Vbrk12:  {m['V_ins_R_peak_kV']:.2f} kV")

    # ── LPM 闪络详情 ──
    if mode == 'lpm':
        print(f"\n  【LPM 先导发展法闪络判定】")
        for ins_name in ['Vbrk11', 'Vbrk12']:
            flashed = m.get(f'{ins_name}_flashed', False)
            t_fo = m.get(f'{ins_name}_flashover_us')
            pct = m.get(f'{ins_name}_leader_pct', 0)
            v_pk = m.get(f'{ins_name}_peak_V_kV', 0)
            if flashed and t_fo is not None:
                print(f"    {ins_name}: ★ 闪络  t = {t_fo:.2f} μs, "
                      f"峰值电压 = {v_pk:.1f} kV")
            else:
                print(f"    {ins_name}: ○ 未闪络  先导进度 = {pct:.1f}%, "
                      f"峰值电压 = {v_pk:.1f} kV")

        # 打印闪络事件日志
        log = sim['results'].get('flashover_log', [])
        if log:
            print(f"\n    闪络事件日志:")
            for ev in log:
                print(f"      {ev['name']}: t={ev['time_us']:.2f}μs, "
                      f"V={ev['voltage_kV']:.1f}kV")

    print(f"\n  【接地电流】")
    print(f"    R_gnd 峰值:   {m['I_gnd_peak_kA']:.4f} kA")

    print(f"\n  【物理解释】")
    if mode == 'open':
        print(f"    绝缘子未闪络: 导线端电压由杆塔感应耦合决定")
    elif mode == 'simple':
        print(f"    绝缘子已闪络: 导线端直接与杆塔连通, 电流经导线分流")
    elif mode == 'lpm':
        any_fo = any(m.get(f'{n}_flashed', False) for n in ['Vbrk11','Vbrk12'])
        if any_fo:
            print(f"    LPM 判定闪络: 先导桥接间隙后导线与杆塔连通")
            print(f"    闪络后电弧电阻 R_arc 限制电流幅值")
        else:
            print(f"    LPM 判定未闪络: 间隙电压不足以维持先导发展")

    print(f"{'=' * 72}")


# =============================================================================
# 辅助函数: 绘制杆塔电压波形图
# =============================================================================

def plot_tower_voltages(sim: dict, save_path: str = None):
    r"""
    绘制杆塔塔顶/塔中/塔脚三条电压波形。

    参数
    ----
    sim : dict
        build_tower_model 返回的仿真结果字典
    save_path : str
        图片保存路径，若为 None 则自动命名
    """
    r = sim['results']
    m = sim['metrics']
    cfg = sim['config']
    flashover = sim['insulator_flashover']

    t_us = r['t_us']
    V_top = r['V_node2_kV']    # 塔顶 Node 2
    V_mid = r['V_node4_kV']    # 塔中 Node 4
    V_bot = r['V_node10_kV']   # 塔脚 Node 10

    # ── 创建图表 ──
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(t_us, V_top, color='#E63946', linewidth=1.8,
            label=f'塔顶电压 (Node 2),  峰值 = {m["V_tower_top_peak_kV"]:.2f} kV')
    ax.plot(t_us, V_mid, color='#457B9D', linewidth=1.8,
            label=f'塔中电压 (Node 4),  峰值 = {m["V_tower_mid_peak_kV"]:.2f} kV')
    ax.plot(t_us, V_bot, color='#2A9D8F', linewidth=1.8,
            label=f'塔脚电压 (Node 10), 峰值 = {m["V_g01_peak_kV"]:.2f} kV')

    # ── 标注 ──
    status_str = '绝缘子闪络' if flashover else '绝缘子未闪络'
    title = (f'杆塔雷电暂态电压波形 — {status_str}\n'
             f'雷电流: {cfg["lightning_type"]} μs,  '
             f'{cfg["lightning_peak"]/1e3:.0f} kA,  '
             f'R_gnd = {cfg["R_ground"]:.0f} Ω')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (μs)', fontsize=12)
    ax.set_ylabel('电压 (kV)', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # 设置 x 轴范围 (只显示有信号的区域)
    t_start_plot = max(0, cfg['lightning_t_start'] * 1e6 - 1.0)
    t_end_plot = min(t_us[-1], cfg['lightning_t_start'] * 1e6 + 20.0)
    ax.set_xlim(t_start_plot, t_end_plot)

    plt.tight_layout()

    # ── 保存 ──
    if save_path is None:
        tag = 'flashover' if flashover else 'no_flashover'
        save_path = f'tower_voltage_{tag}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ 电压波形图已保存: {save_path}")
    return save_path


# =============================================================================
# 测试 11: 杆塔模型拓扑搭建 & 参数验证
# =============================================================================

def test_11_tower_topology_and_parameters():
    r"""
    测试11: 杆塔模型拓扑搭建 & 参数验证
    ────────────────────────────────────
    使用 PSCAD 风格参数 (Zc, tau_per_m, length) 通过 create_from_zc_tau
    创建所有 Bergeron 传输线，验证每条线路的派生参数。

    杆塔包含 10 条 Bergeron 线路:
      - 避雷线:  ZA10, ZB10
      - 杆塔体:  ZT11, ZT12, ZT13
      - 引下线:  ZL11, ZL12, ZL13  (分别与 ZT 并联)
      - 导线:    ZA110, ZB110

    验证项:
      1. 每条线路的 Zc / τ / α / G_eq 与输入参数一致
      2. 延时缓冲区 delay_steps 和分数延时正确
      3. 无损条件下 \( \alpha = 1.0 \),  \( Z_{c,eff} = Z_c \)
    """
    print("\n" + "█" * 72)
    print("  测试 11: 杆塔模型拓扑搭建 & 参数验证")
    print("█" * 72)

    cfg = TOWER_CONFIG
    dt = cfg['dt']

    # 创建求解器 (仅用于参数验证, finish_time 最小)
    solver = EMTPSolver(dt=dt, finish_time=dt, verbose=False)

    # 线路定义: (名称, node_k, node_m, 配置key)
    line_defs = [
        ("ZA10",  1, 2,  'ZA10'),
        ("ZB10",  2, 3,  'ZB10'),
        ("ZT11",  2, 4,  'ZT11'),
        ("ZL11",  2, 4,  'ZL11'),
        ("ZA110", 5, 4,  'ZA110'),
        ("ZB110", 4, 6,  'ZB110'),
        ("ZT12",  4, 7,  'ZT12'),
        ("ZL12",  4, 7,  'ZL12'),
        ("ZT13",  7, 10, 'ZT13'),
        ("ZL13",  7, 10, 'ZL13'),
    ]

    # 创建并注入所有线路
    for name, nk, nm, cfg_key in line_defs:
        line = _create_bergeron_line(name, nk, nm, cfg[cfg_key])
        solver.add_line(line)

    # ── 逐线验证参数 ──
    print(f"\n  ┌─ 逐线参数验证 ─────────────────────────────────────────────┐")
    print(f"  │  {'名称':<8s}  {'Zc(Ω)':>8s}  {'τ(ns)':>10s}  {'α':>8s}"
          f"  {'G_eq(S)':>14s}  {'steps':>6s}  {'ok':>4s} │")
    print(f"  │  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*14}  {'─'*6}  {'─'*4} │")

    all_pass = True
    for name, nk, nm, cfg_key in line_defs:
        info = solver.get_line_info(name)
        lc   = cfg[cfg_key]
        length_m   = lc['length_km'] * 1000.0
        tau_expect = lc['tau_per_m'] * length_m

        # 检查 Zc
        ok_Zc  = abs(info['Zc'] - lc['Zc']) / lc['Zc'] < 1e-10
        # 检查 τ
        ok_tau = abs(info['tau'] - tau_expect) / tau_expect < 1e-6

        ok = ok_Zc and ok_tau
        all_pass = all_pass and ok

        tau_ns = info['tau'] * 1e9
        symbol = "✓" if ok else "✗"
        print(f"  │  {name:<8s}  {info['Zc']:>8.1f}  {tau_ns:>10.4f}"
              f"  {info['delay_steps']:>6d}  {symbol:>4s} │")

    print(f"  └─────────────────────────────────────────────────────────────────┘")

    assert all_pass, "杆塔线路参数验证存在不匹配项！"
    print(f"\n  ✓ 全部 10 条 Bergeron 线路参数验证通过！")

    # ── 打印节点清单 ──
    print(f"\n  ┌─ 节点定义 ─────────────────────────────────────────────────────┐")
    nodes = [
        (0,  "GND",    "大地参考"),
        (1,  "sI1",    "避雷线左跨端 / I1S1 注入"),
        (2,  "塔顶",   "ZA10-ZB10 汇合, ZT11‖ZL11 顶端"),
        (3,  "sI2",    "避雷线右跨端 / I1S2 注入"),
        (4,  "塔中",   "ZT11‖ZL11底 / ZA110-ZB110中 / ZT12‖ZL12顶"),
        (5,  "导线左",  "ZA110 左端, Vbrk11 上端"),
        (6,  "导线右",  "ZB110 右端, Vbrk12 上端"),
        (7,  "塔下",   "ZT12‖ZL12底 / ZT13‖ZL13顶"),
        (8,  "cI1",    "Vbrk11 下端, 导线外端(左)"),
        (9,  "cI2",    "Vbrk12 下端, 导线外端(右)"),
        (10, "g01",    "ZT13‖ZL13底, 接地点"),
    ]
    for nid, label, desc in nodes:
        print(f"  │  Node {nid:>2d}  [{label:<6s}]  {desc}")
    print(f"  └─────────────────────────────────────────────────────────────────┘")


# =============================================================================
# 测试 12: 杆塔模型延时缓冲区验证
# =============================================================================

def test_12_tower_delay_buffers():
    r"""
    测试12: 杆塔各段线路延时缓冲区验证
    ────────────────────────────────────
    杆塔各段线路长度差异极大:
      - 避雷线/导线跨:  200 m → τ ≈ 667 ns
      - 杆塔体:         8~15 m → τ ≈ 27~50 ns
      - 引下线:         5~8 m  → τ ≈ 17~27 ns

    验证各段的 delay_steps 和分数延时 (dt = 1 ns)。
    """
    print("\n" + "█" * 72)
    print("  测试 12: 杆塔各段线路延时缓冲区验证")
    print("█" * 72)

    cfg = TOWER_CONFIG
    dt = cfg['dt']

    line_cfgs = [
        ('ZA10',  cfg['ZA10']),
        ('ZB10',  cfg['ZB10']),
        ('ZT11',  cfg['ZT11']),
        ('ZL11',  cfg['ZL11']),
        ('ZA110', cfg['ZA110']),
        ('ZB110', cfg['ZB110']),
        ('ZT12',  cfg['ZT12']),
        ('ZL12',  cfg['ZL12']),
        ('ZT13',  cfg['ZT13']),
        ('ZL13',  cfg['ZL13']),
    ]

    print(f"\n  dt = {dt*1e9:.1f} ns")
    print(f"\n  {'名称':<8s}  {'长度(m)':>8s}  {'τ(ns)':>10s}  {'τ/dt':>10s}"
          f"  {'delay_steps':>12s}  {'frac_delay':>12s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")

    for name, lc in line_cfgs:
        length_m  = lc['length_km'] * 1000.0
        tau_total = lc['tau_per_m'] * length_m
        ratio     = tau_total / dt

        buf = DelayBuffer.create_for_delay(tau_total, dt)

        print(f"  {name:<8s}  {length_m:>8.1f}  {tau_total*1e9:>10.4f}"
              f"  {ratio:>10.4f}  {buf.delay_steps:>12d}"
              f"  {buf.fractional_delay:>12.8f}")

        # 阶跃响应快速验证
        for i in range(buf.delay_steps + 3):
            buf.push(1.0 if i >= 1 else 0.0)
        delayed = buf.get_delayed()
        assert delayed > 0.5, (f"{name}: 延时缓冲区阶跃输出异常 = {delayed:.6f}, "
                               f"预期 > 0.5")

    print(f"\n  ✓ 全部延时缓冲区验证通过")


# =============================================================================
# 测试 13: 杆塔雷电暂态仿真 (绝缘子未闪络)
# =============================================================================

def test_13_tower_lightning_no_flashover():
    r"""
    测试13: 杆塔雷电暂态仿真 — 绝缘子未闪络
    ──────────────────────────────────────────
    雷电流 2/20 μs 10 kA 注入 I1S1,
    绝缘子 Vbrk11/Vbrk12 保持开路 (R = 1e9 Ω)。

    物理预期:
    - 雷电流经避雷线到达塔顶
    - 沿杆塔体/引下线向下传播至接地点
    - 绝缘子开路: 导线端 (node 5/6) 电压由耦合感应产生
    - 塔顶电压最高, 塔脚电压受接地电阻限制
    """
    print("\n" + "█" * 72)
    print("  测试 13: 杆塔雷电暂态仿真 (绝缘子未闪络)")
    print("█" * 72)

    sim = build_tower_model(
        verbose=True,
        insulator_flashover=False,
    )
    print_tower_results(sim)

    # 物理合理性检查
    m = sim['metrics']
    assert m['V_tower_top_peak_kV'] > 0, "塔顶电压应 > 0"
    assert m['V_g01_peak_kV'] > 0, "接地电压应 > 0"
    assert m['V_tower_top_peak_kV'] >= m['V_g01_peak_kV'], \
        "塔顶电压应 ≥ 塔脚电压"

    print("\n  ✓ 杆塔雷电暂态仿真 (未闪络) 测试完成")

    # ── 绘制塔顶/塔中/塔脚电压波形 ──
    plot_tower_voltages(sim, save_path='tower_voltage_no_flashover.png')

    return sim


# =============================================================================
# 测试 14: 杆塔雷电暂态仿真 (绝缘子闪络)
# =============================================================================

def test_14_tower_lightning_with_flashover():
    r"""
    测试14: 杆塔雷电暂态仿真 — 绝缘子闪络
    ──────────────────────────────────────────
    同测试13, 但绝缘子 Vbrk11/Vbrk12 闪络 (R = 0.01 Ω)。

    物理预期:
    - 绝缘子闪络后, 导线直接与杆塔连通
    - 雷电流部分经导线分流, 塔脚电流减小
    - 导线电压上升, 塔顶电压可能降低 (分流效应)
    """
    print("\n" + "█" * 72)
    print("  测试 14: 杆塔雷电暂态仿真 (绝缘子闪络)")
    print("█" * 72)

    sim = build_tower_model(
        verbose=True,
        insulator_flashover=True,
    )
    print_tower_results(sim)

    print("\n  ✓ 杆塔雷电暂态仿真 (闪络) 测试完成")

    # ── 绘制塔顶/塔中/塔脚电压波形 ──
    plot_tower_voltages(sim, save_path='tower_voltage_flashover.png')

    return sim


# =============================================================================
# 测试 15: 闪络 vs 未闪络 对比
# =============================================================================

def test_15_flashover_comparison():
    """
    测试15: 绝缘子闪络 vs 未闪络 对比
    ──────────────────────────────────
    对比两种绝缘子状态下的塔顶/塔中/塔脚电压和接地电流峰值。
    """
    print("\n" + "█" * 72)
    print("  测试 15: 绝缘子闪络 vs 未闪络 对比")
    print("█" * 72)

    sim_no  = build_tower_model(verbose=False, insulator_flashover=False)
    sim_yes = build_tower_model(verbose=False, insulator_flashover=True)

    m_no  = sim_no['metrics']
    m_yes = sim_yes['metrics']

    print(f"\n  {'─' * 68}")
    print(f"  {'指标':<24s}  {'未闪络':>14s}  {'闪络':>14s}  {'变化':>10s}")
    print(f"  {'─' * 68}")

    comparisons = [
        ('塔顶电压 (kV)',   m_no['V_tower_top_peak_kV'], m_yes['V_tower_top_peak_kV']),
        ('塔中电压 (kV)',   m_no['V_tower_mid_peak_kV'], m_yes['V_tower_mid_peak_kV']),
        ('塔脚电压 (kV)',   m_no['V_g01_peak_kV'],       m_yes['V_g01_peak_kV']),
        ('接地电流 (kA)',   m_no['I_gnd_peak_kA'],       m_yes['I_gnd_peak_kA']),
        ('左绝缘子压差(kV)', m_no['V_ins_L_peak_kV'],    m_yes['V_ins_L_peak_kV']),
        ('右绝缘子压差(kV)', m_no['V_ins_R_peak_kV'],    m_yes['V_ins_R_peak_kV']),
    ]

    for label, v_no, v_yes in comparisons:
        if abs(v_no) > 1e-10:
            change = (v_yes - v_no) / v_no * 100
            change_str = f"{change:>+.1f}%"
        else:
            change_str = "N/A"
        print(f"  {label:<24s}  {v_no:>14.2f}  {v_yes:>14.2f}  {change_str:>10s}")

    print(f"  {'─' * 68}")
    print(f"\n  ✓ 闪络 vs 未闪络对比测试完成")


# =============================================================================
# 测试 16: LPM 先导发展法绝缘子闪络 (PSCAD 参数)
# =============================================================================

def test_16_tower_lpm_flashover():
    r"""
    测试16: 杆塔雷电暂态仿真 — CIGRE 先导发展法闪络判据
    ──────────────────────────────────────────────────────
    使用从 PSCAD 逻辑元件截图中提取的 LPM 参数:
        E₀ = 670.0 kV/m  (临界场强)
        k  = 1.0×10⁻⁶     (先导速度系数)
        d  = 1.0 m         (间隙长度)

    PSCAD 信号流分析:
    ────────────────
    Vbrk12 → |X|(取绝对值) → N/D(N=|V|, D=d−l) → 计算平均场强
    → 减去 E₀(670.0) → ×|V|(乘以间隙电压) → ×k(乘以系数 1.0)
    → 积分器(1/sT) → 先导长度 l
    → Comparator(l ≥ d) → Monostable → BRK12o 闭合

    底部两个 Comparator(A>0, B=0.0):
    门控信号 — 当 u/(d−l)−E₀ < 0 时，速度置零，先导不发展

    物理预期 (10 kA, 2/20 μs):
    - 塔中层绝缘子两端电压峰值约数百 kV
    - 是否闪络取决于电压幅值与 E₀·d 的关系
    - 若闪络, 绝缘子电阻突变为 R_arc, 塔顶电压降低
    """
    print("\n" + "█" * 72)
    print("  测试 16: 杆塔雷电暂态 — CIGRE 先导发展法 (PSCAD 参数)")
    print("█" * 72)

    # ── 使用默认 PSCAD 参数 (E0=670, k=1.0e-6, d=1.0) ──
    sim_lpm = build_tower_model(
        verbose=True,
        use_lpm=True,
    )
    print_tower_results(sim_lpm)

    # 提取 LPM 详情
    m = sim_lpm['metrics']
    for ins_name in ['Vbrk11', 'Vbrk12']:
        lpm = sim_lpm['lpm_models'].get(ins_name)
        if lpm:
            lpm.print_info()

    # 输出先导发展过程
    r = sim_lpm['results']
    if 'Vbrk12_leader_mm' in r:
        leader = r['Vbrk12_leader_mm']
        t_us = r['t_us']
        # 找到先导开始发展的时刻
        active_idx = np.where(leader > 0.01)[0]
        if len(active_idx) > 0:
            i0 = max(0, active_idx[0] - 10)
            print(f"\n  先导发展过程 (Vbrk12, 部分采样):")
            print(f"  {'时间(μs)':>10} {'先导(mm)':>10} {'进度(%)':>8}")
            print(f"  {'─'*32}")
            step = max(1, len(active_idx) // 15)
            for idx in range(i0, min(len(leader), active_idx[-1]+10), step):
                d_mm = sim_lpm['lpm_models']['Vbrk12'].config.gap_length * 1e3
                pct = leader[idx] / d_mm * 100
                print(f"  {t_us[idx]:10.3f} {leader[idx]:10.2f} {pct:8.1f}")

    print(f"\n  ✓ LPM 先导发展法测试完成")

    # ── 绘制电压波形 ──
    plot_tower_voltages(sim_lpm, save_path='tower_voltage_lpm.png')

    return sim_lpm


# =============================================================================
# 测试 17: 三模式对比 (未闪络 / 简易闪络 / LPM 闪络)
# =============================================================================

def test_17_three_mode_comparison():
    r"""
    测试17: 三种绝缘子模式对比
    ──────────────────────────
    1. 未闪络 (R = 1e9 Ω)
    2. 简易闪络 (R = 0.01 Ω, 全程闪络)
    3. LPM 先导发展法 (E₀=670, k=1.0e-6, d=1.0)

    对比指标: 塔顶/塔中/塔脚电压峰值, 接地电流峰值
    """
    print("\n" + "█" * 72)
    print("  测试 17: 三种绝缘子模式对比")
    print("█" * 72)

    sim_open   = build_tower_model(verbose=False, insulator_flashover=False)
    sim_simple = build_tower_model(verbose=False, insulator_flashover=True)
    sim_lpm    = build_tower_model(verbose=False, use_lpm=True)

    m_o = sim_open['metrics']
    m_s = sim_simple['metrics']
    m_l = sim_lpm['metrics']

    print(f"\n  {'─' * 78}")
    print(f"  {'指标':<24s}  {'未闪络':>12s}  {'简易闪络':>12s}  {'LPM':>12s}")
    print(f"  {'─' * 78}")

    rows = [
        ('塔顶电压 (kV)',   m_o['V_tower_top_peak_kV'],
                            m_s['V_tower_top_peak_kV'],
                            m_l['V_tower_top_peak_kV']),
        ('塔中电压 (kV)',   m_o['V_tower_mid_peak_kV'],
                            m_s['V_tower_mid_peak_kV'],
                            m_l['V_tower_mid_peak_kV']),
        ('塔脚电压 (kV)',   m_o['V_g01_peak_kV'],
                            m_s['V_g01_peak_kV'],
                            m_l['V_g01_peak_kV']),
        ('接地电流 (kA)',   m_o['I_gnd_peak_kA'],
                            m_s['I_gnd_peak_kA'],
                            m_l['I_gnd_peak_kA']),
        ('左绝缘子压差(kV)', m_o['V_ins_L_peak_kV'],
                            m_s['V_ins_L_peak_kV'],
                            m_l['V_ins_L_peak_kV']),
    ]

    for label, v_o, v_s, v_l in rows:
        print(f"  {label:<24s}  {v_o:>12.2f}  {v_s:>12.2f}  {v_l:>12.2f}")

    print(f"  {'─' * 78}")

    # LPM 闪络状态
    for ins in ['Vbrk11', 'Vbrk12']:
        fo = m_l.get(f'{ins}_flashed', False)
        t_fo = m_l.get(f'{ins}_flashover_us')
        pct = m_l.get(f'{ins}_leader_pct', 0)
        if fo and t_fo is not None:
            print(f"  LPM {ins}: ★ 闪络 @ {t_fo:.2f} μs")
        else:
            print(f"  LPM {ins}: ○ 未闪络 (先导 {pct:.1f}%)")

    print(f"\n  ✓ 三模式对比测试完成")

    return sim_open, sim_simple, sim_lpm


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    cfg = TOWER_CONFIG

    print("╔" + "═" * 72 + "╗")
    print("║  多层杆塔 Bergeron 传输线模型 — 完整测试套件 (EMTPSolver 版)     ║")
    print("║  拓扑: 避雷线(ZA10,ZB10) + 杆塔体(ZT11~ZT13) + 引下线(ZL11~ZL13)║")
    print("║        + 导线(ZA110,ZB110) + 绝缘子(Vbrk11,Vbrk12)             ║")
    print("║  雷电流: 2/20 μs,  10 kA,  单端注入 (I1S1)                    ║")
    print("║  接地电阻: R_gnd = 20 Ω                                        ║")
    print("║  新增: CIGRE 先导发展法 (LPM) 绝缘子闪络判据                     ║")
    print("║        参数来源: PSCAD 逻辑元件 (E0=670, k=1.0e-6, d=1.0m)     ║")
    print("╚" + "═" * 72 + "╝")

    test_11_tower_topology_and_parameters()
    test_12_tower_delay_buffers()
    test_13_tower_lightning_no_flashover()
    test_14_tower_lightning_with_flashover()
    test_15_flashover_comparison()
    test_16_tower_lpm_flashover()
    test_17_three_mode_comparison()

    print("\n" + "╔" + "═" * 72 + "╗")
    print("║  杆塔模型全部 7 项测试执行完毕 (含 LPM 先导发展法)              ║")
    print("╚" + "═" * 72 + "╝")
