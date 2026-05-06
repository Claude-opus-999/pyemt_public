#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三芯管型电缆 (Pipe-Type Cable) ULM 完整计算 + PSCAD 对比脚本
================================================================================

本脚本基于 cable_model_0110.py 的 PipeTypeCableGeometry 模型，
计算 Cable_Summary.txt 中三芯管型电缆 (Cable_14) 的 ULM 参数，
并与 PSCAD 输出进行完整对比。

模块化结构：

  第1部分: 导入模块 (含 pscad_reader)
  第2部分: 电缆参数配置 (来自 Cable_Summary.txt)
  第3部分: 计算 Z/Y 矩阵 (cable_model_0110.py → PT Cable 7×7)
  第4部分: 通过 Vector Fitting 模块计算 Yc 和 H
  第5部分: 从 PSCAD 读取参考数据
  第6部分: 误差计算
  第7部分: Ti 矩阵对比工具
  第8部分: 绘图工具 (全矩阵对比、模态 H、Lambda、Ti、误差、VF 拟合等)
  第9部分: 典型频率点数值打印
  第10部分: fitULM 文件导出
  第11部分: 主函数

对比内容（计算值 vs PSCAD 参考值的完整对比）：
  - Z 阻抗矩阵全矩阵幅值与相角 (自研 vs PSCAD)
  - Y 导纳矩阵全矩阵幅值与相角 (自研 vs PSCAD)
  - Z/Y 全矩阵误差 vs 频率曲线
  - Z/Y 典型频率点全矩阵数值对比
  - Yc 完整 n×n 矩阵幅值与相角 (VF 计算值 vs PSCAD)
  - H (相域) 完整 n×n 矩阵幅值与相角
  - H (模态域) 幅值与相角
  - Yc/H 全矩阵误差 vs 频率曲线
  - 传播常数 λ = γ² (实部 + 虚部, 与 PSCAD Lambda 直接对比)
  - Ti 变换矩阵 (列向量归一化对比 + 元素级幅值/相角对比)
  - PSCAD 内部拟合图 (Yc calc vs fitted, H calc vs fitted)
  - VF 拟合质量 (tr(Yc) RMSE, H 重构 RMSE, 被动性, 极点分布)
  - 误差统计 (全频率范围, 全矩阵元素)

电缆结构 (Cable_14):
  三芯管型电缆 (Pipe-Type Cable)
  ├── 3 根同轴内导体 (各含芯线 + 绝缘 + 铅护套 + 外绝缘)
  │   ├── 芯线: r = 1.175 mm, Cu (ρ = 1.72e-8 Ω·m)
  │   ├── 绝缘1: R = 25.05 mm, XLPE (εr = 2.3)
  │   ├── 铅护套: R = 27.15 mm (ρ = 2.2e-7 Ω·m)
  │   └── 绝缘2: R = 29.55 mm, XLPE (εr = 2.3)
  ├── 管内填充绝缘 (εr = 2.3)
  ├── 钢铠装管道: Rin = 66.5 mm, Rout = 71.5 mm
  │   (ρ = 9.78e-8 Ω·m, μr = 200)
  └── 外护套: R = 74.5 mm (εr = 2.3)

矩阵维度: 7×7 (3×芯线 + 3×护套 + 1×管道)
导体排列: [Core1, Sheath1, Core2, Sheath2, Core3, Sheath3, Pipe]

依赖模块:
    - cable_model_0110.py                  : 电缆 Z/Y 计算 (PT Cable)
    - vector_fitting_v48_nr_consistent.py  : ULM 完整拟合 (Vector Fitting)
    - pscad_reader.py                      : PSCAD 输出读取器

使用方法:
    python ulm_three_core_cable_v2.py [PSCAD文件前缀路径] [线路长度m] [fitULM文件名]

示例:
    python ulm_three_core_cable_v2.py ./Cable_14/Cable_14 5000
    python ulm_three_core_cable_v2.py ./Cable_14/Cable_14 5000 cable14.fitULM
    python ulm_three_core_cable_v2.py ./Cable_14 5000

作者: Claude
日期: 2026-03-18
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Union

# =============================================================================
# 路径配置
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# =============================================================================
# 物理常量
# =============================================================================
MU_0 = 4 * np.pi * 1e-7
EPSILON_0 = 8.854187817e-12
C_LIGHT = 299792458.0


# =============================================================================
# 第一部分：导入模块 (含 pscad_reader)
# =============================================================================
def import_all_modules() -> Dict:
    """导入所有需要的模块"""
    modules = {}

    # 1. PSCAD 读取器
    try:
        from pscad_reader import PSCADFileReader, PSCADPlotter
        modules['PSCADFileReader'] = PSCADFileReader
        modules['PSCADPlotter'] = PSCADPlotter
        print("✓ pscad_reader 导入成功")
    except ImportError as e:
        print(f"✗ pscad_reader 导入失败: {e}")
        modules['PSCADFileReader'] = None
        modules['PSCADPlotter'] = None

    # 2. 电缆模型 (cable_model_0110)
    try:
        import cable_model as cable_model
        modules['cable_model'] = cable_model
        print("✓ cable_model_0110 导入成功")
        for fn in ['PipeTypeCableGeometry', 'InnerConductor',
                    'compute_pipe_type_cable_impedance',
                    'compute_pipe_type_cable_potential',
                    'generate_frequency_vector', 'SoilParameters']:
            if hasattr(cable_model, fn):
                print(f"  ✓ {fn}")
            else:
                print(f"  ✗ {fn} 不存在!")
    except ImportError as e:
        print(f"✗ cable_model_0110 导入失败: {e}")
        modules['cable_model'] = None

    # 3. Vector Fitting / ULM 模块
    try:
        import vector_fitting_v411_independent as vf
        modules['vf'] = vf
        ver = getattr(vf, '__version__', 'unknown')
        print(f"✓ vector_fitting V{ver} 导入成功")
        if hasattr(vf, 'ulm_complete_fitting'):
            print("  ✓ ulm_complete_fitting 可用")
        if hasattr(vf, 'write_fitULM'):
            print("  ✓ fitULM 导出功能可用")
        if hasattr(vf, 'IterativePoleFindingConfig'):
            print("  ✓ IterativePoleFindingConfig 可用")
    except ImportError as e:
        print(f"✗ vector_fitting 导入失败: {e}")
        modules['vf'] = None

    return modules


# =============================================================================
# 第二部分：电缆参数配置 (来自 Cable_Summary.txt)
# =============================================================================
@dataclass
class CableLineConfig:
    """
    电缆线路参数配置
    ——从 Cable_Summary.txt 提取全部参数
    """
    line_name: str = "Cable_14"
    n_inner_cables: int = 3
    n_total_conductors: int = 7    # 3×(Core+Sheath) + 1×Pipe

    # 线路参数
    line_length: float = 5000.0    # 5.0 km = 5000 m
    steady_state_freq: float = 50.0

    # 频率范围
    freq_min: float = 0.5
    freq_max: float = 1e6
    n_freq: int = 200             # 频率增量数 (对数分布)

    # 土壤参数
    ground_resistivity: float = 100.0     # Ω·m
    ground_permeability: float = 1.0
    ground_permittivity: float = 1.0

    # 管道/铠装参数
    pipe_inner_radius: float = 0.0665     # Inner Insulator Outer Radius
    pipe_outer_radius: float = 0.0715     # Conductor Outer Radius
    pipe_rho: float = 9.78e-8             # 钢铠装电阻率
    pipe_mu_r: float = 200.0              # 钢相对磁导率
    jacket_radius: float = 0.0745         # Outer Insulator Outer Radius
    pipe_inner_insulation_epsilon_r: float = 2.3   # 管内填充 εr
    pipe_outer_insulation_epsilon_r: float = 2.3   # 外护套 εr

    # 敷设参数
    burial_depth: float = 1.0             # P1 第二个参数
    horizontal_pos: float = 0.0           # P1 第一个参数

    # 内导体参数 (各相相同, Identical Cables = 1)
    core_radius: float = 0.001175         # 芯线外半径
    core_rho: float = 1.72e-8             # Cu
    core_mu_r: float = 1.0
    insulation_radius: float = 0.02505    # 绝缘1外半径
    insulation_epsilon_r: float = 2.3     # XLPE
    sheath_outer_radius: float = 0.02715  # 铅护套外半径
    sheath_rho: float = 2.2e-7            # 铅
    sheath_mu_r: float = 1.0
    outer_insulation_radius: float = 0.02955  # 绝缘2外半径
    outer_insulation_epsilon_r: float = 2.3

    # 三根内导体角度位置 (度)
    cable_angles_deg: List[float] = field(default_factory=lambda: [270.0, 30.0, 150.0])
    distance_from_center: float = 0.03415     # 各内导体到管道中心距离

    # Vector Fitting 参数
    Yc_poles_min: int = 12
    Yc_poles_max: int = 20
    Yc_target_error: float = 0.002      # 0.2%
    H_poles_min: int = 12
    H_poles_max: int = 20
    H_target_error: float = 0.002       # 0.2%


def create_pipe_type_cable(cable_model, config: CableLineConfig):
    """根据 CableLineConfig 创建 PipeTypeCableGeometry 对象"""
    angles_rad = [np.deg2rad(a) for a in config.cable_angles_deg]

    inner_conductors = []
    for i, angle in enumerate(angles_rad):
        ic = cable_model.InnerConductor(
            core_radius=config.core_radius,
            core_rho=config.core_rho,
            core_mu_r=config.core_mu_r,
            insulation_radius=config.insulation_radius,
            insulation_epsilon_r=config.insulation_epsilon_r,
            insulation_mu_r=1.0,
            has_sheath=True,
            sheath_inner_radius=config.insulation_radius,
            sheath_outer_radius=config.sheath_outer_radius,
            sheath_rho=config.sheath_rho,
            sheath_mu_r=config.sheath_mu_r,
            outer_insulation_radius=config.outer_insulation_radius,
            outer_insulation_epsilon_r=config.outer_insulation_epsilon_r,
            distance_from_center=config.distance_from_center,
            angular_position=angle,
        )
        inner_conductors.append(ic)

    pt_cable = cable_model.PipeTypeCableGeometry(
        inner_conductors=inner_conductors,
        pipe_inner_radius=config.pipe_inner_radius,
        pipe_outer_radius=config.pipe_outer_radius,
        pipe_rho=config.pipe_rho,
        pipe_mu_r=config.pipe_mu_r,
        jacket_radius=config.jacket_radius,
        jacket_epsilon_r=config.pipe_outer_insulation_epsilon_r,
        pipe_inner_insulation_epsilon_r=config.pipe_inner_insulation_epsilon_r,
        pipe_outer_insulation_epsilon_r=config.pipe_outer_insulation_epsilon_r,
        burial_depth=config.burial_depth,
        horizontal_pos=config.horizontal_pos,
        finite_pipe_thickness=True,
    )

    return pt_cable


# =============================================================================
# 第三部分：计算 Z/Y 矩阵
# =============================================================================
def compute_ZY_self(modules: Dict,
                    config: CableLineConfig,
                    freq: np.ndarray,
                    verbose: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    使用 cable_model_0110 计算三芯管型电缆的阻抗/导纳矩阵

    Returns
    -------
    Z_matrix : ndarray [n_freq, 7, 7]  阻抗矩阵 [Ω/m]
    Y_matrix : ndarray [n_freq, 7, 7]  导纳矩阵 [S/m]
    P_matrix : ndarray [7, 7]          电位系数矩阵 [m/F]
    pt_cable : PipeTypeCableGeometry   电缆几何对象
    """
    cable_model = modules['cable_model']

    if verbose:
        print("\n" + "=" * 65)
        print(" 使用 cable_model_0110 计算 Z/Y (PT Cable 模型)")
        print("=" * 65)

    pt_cable = create_pipe_type_cable(cable_model, config)
    n_total = pt_cable.total_conductors

    if verbose:
        print(f"\n  电缆名称: {config.line_name}")
        print(f"  内导体数: {pt_cable.n_inner_conductors} (每根含芯线+护套)")
        print(f"  总导体数: {n_total} (3×Core + 3×Sheath + 1×Pipe)")
        print(f"  矩阵维度: {n_total}×{n_total}")
        print(f"  管道: Rin = {config.pipe_inner_radius*1000:.1f} mm, "
              f"Rout = {config.pipe_outer_radius*1000:.1f} mm")
        print(f"  铠装: ρ = {config.pipe_rho:.2e} Ω·m, μr = {config.pipe_mu_r}")
        print(f"  外护套: R = {config.jacket_radius*1000:.1f} mm")
        print(f"  埋深: {config.burial_depth} m")
        for i, ic in enumerate(pt_cable.inner_conductors):
            angle_deg = config.cable_angles_deg[i]
            print(f"  内导体 {i+1}: d = {ic.distance_from_center*1000:.2f} mm, "
                  f"θ = {angle_deg}°, "
                  f"r_core = {ic.core_radius*1000:.3f} mm, "
                  f"R_sheath = {ic.sheath_outer_radius*1000:.2f} mm, "
                  f"R_outer = {ic.outer_insulation_radius*1000:.2f} mm")

    soil = cable_model.SoilParameters(
        rho=config.ground_resistivity,
        epsilon_r=config.ground_permittivity,
        mu_r=config.ground_permeability,
    )
    gamma_soil = soil.get_gamma(freq)

    if verbose:
        print(f"\n  土壤电阻率: {config.ground_resistivity} Ω·m")
        print(f"  频率范围: {freq[0]:.4f} ~ {freq[-1]:.2e} Hz ({len(freq)} 点)")

    if verbose:
        print("\n  计算阻抗矩阵 Z (Pollaczek 数值积分)...")
    Z_matrix = cable_model.compute_pipe_type_cable_impedance(freq, pt_cable, gamma_soil)

    if verbose:
        print("  计算电位系数矩阵 P ...")
    P_matrix = cable_model.compute_pipe_type_cable_potential(freq, pt_cable)

    if verbose:
        print("  计算导纳矩阵 Y = jω · P⁻¹ ...")
    omega = cable_model.omega_from_freq(freq)
    n_freq = len(freq)
    Y_matrix = np.zeros((n_freq, n_total, n_total), dtype=complex)

    try:
        P_inv = np.linalg.inv(P_matrix)
        P_cond = np.linalg.cond(P_matrix)
        for i, om in enumerate(omega):
            Y_matrix[i] = 1j * om * P_inv
        if verbose:
            print(f"  P 矩阵条件数: {P_cond:.2e}")
    except np.linalg.LinAlgError:
        print("  ✗ 电位系数矩阵 P 奇异，无法求逆!")
        return Z_matrix, Y_matrix, P_matrix, pt_cable

    if verbose:
        print(f"\n  Z_matrix: {Z_matrix.shape},  Y_matrix: {Y_matrix.shape}")
        print(f"  P_matrix: {P_matrix.shape}")

        Z_sym_err = np.max(np.abs(Z_matrix - np.transpose(Z_matrix, (0, 2, 1))))
        P_sym_err = np.max(np.abs(P_matrix - P_matrix.T))
        print(f"\n  对称性检查:")
        print(f"    Z 矩阵: max|Z - Z^T| = {Z_sym_err:.2e}")
        print(f"    P 矩阵: max|P - P^T| = {P_sym_err:.2e}")

        P_eigvals = np.linalg.eigvalsh(P_matrix)
        print(f"    P 矩阵特征值: min = {np.min(P_eigvals):.4e}, "
              f"max = {np.max(P_eigvals):.4e}")
        if np.all(P_eigvals > 0):
            print(f"    P 矩阵: ✓ 正定")
        else:
            print(f"    P 矩阵: ✗ 非正定!")

    return Z_matrix, Y_matrix, P_matrix, pt_cable


# =============================================================================
# 第四部分：通过 Vector Fitting 模块计算 Yc 和 H
# =============================================================================
def compute_Yc_H_via_VF(modules: Dict,
                         freq: np.ndarray,
                         Z_matrix: np.ndarray,
                         Y_matrix: np.ndarray,
                         config: CableLineConfig,
                         use_freq_dependent: str = 'auto',
                         verbose: bool = True
                         ) -> Dict:
    """调用 vector_fitting 模块的 ulm_complete_fitting() 计算 Yc 和 H"""
    vf = modules['vf']

    if verbose:
        print("\n" + "=" * 65)
        print(f" 通过 Vector Fitting 模块计算 Yc 和 H")
        print(f" (线路长度 L = {config.line_length/1000:.1f} km)")
        print("=" * 65)

    vf_config = vf.IterativePoleFindingConfig(
        Ymin=config.Yc_poles_min,
        Ymax=config.Yc_poles_max,
        epsY=config.Yc_target_error,
        Hmin=config.H_poles_min,
        Hmax=config.H_poles_max,
        epsH=config.H_target_error,
        pole_step=2,
        eps_deg=10.0,
        use_pscad_style=True,
        compute_H_reconstruction_metrics=True,
        verbose_H_metrics=verbose,
    )

    if verbose:
        print(f"  VF 配置:")
        print(f"    tr(Yc): {vf_config.Ymin}-{vf_config.Ymax} 极点, "
              f"目标误差 {vf_config.epsY * 100}%")
        print(f"    H 模态: {vf_config.Hmin}-{vf_config.Hmax} 极点, "
              f"目标误差 {vf_config.epsH * 100}%")
        print(f"    use_freq_dependent: {use_freq_dependent}")
        print(f"    Ti 来源: 自研 Newton-Raphson 特征求解")

    ulm_params, fitting_result = vf.ulm_complete_fitting(
        freq=freq,
        Z_matrix=Z_matrix,
        Y_matrix=Y_matrix,
        length=config.line_length,
        velocity_freq=1e5,
        config=vf_config,
        use_freq_dependent=use_freq_dependent,
        enforce_passivity_flag=True,
        verbose=verbose,
    )

    n_cond = Z_matrix.shape[1]

    if verbose:
        print(f"\n  [VF 结果提取]")
        print(f"  Yc_matrix: {ulm_params.Yc_matrix.shape}")
        print(f"  H_matrix:  {ulm_params.H_matrix.shape}")
        print(f"  H_modes:   {ulm_params.H_modes.shape}")
        print(f"  gamma:     {ulm_params.gamma_modes.shape}")
        print(f"  T_ref:     {ulm_params.T_ref.shape}")
        print(f"  Ti source: {ulm_params.source}")
        print(f"  freq_dep:  {ulm_params.is_freq_dependent}")

        print(f"\n  tr(Yc) 拟合 RMSE: {fitting_result.Yc_trace_rmse * 100:.4f}%")
        print(f"  H 矩阵重构 RMSE: {fitting_result.H_matrix_rmse * 100:.4f}%")
        print(f"  被动性: {'✓' if fitting_result.is_passive else '✗'}")

        for i in range(n_cond):
            tau_us = ulm_params.tau[i] * 1e6
            v_ratio = config.line_length / (ulm_params.tau[i]) / C_LIGHT \
                if ulm_params.tau[i] > 0 else float('inf')
            print(f"    Mode {i+1}: τ = {tau_us:.4f} μs  (v/c = {v_ratio:.4f})")

        idx_50 = np.argmin(np.abs(freq - 50))
        print(f"\n  @ {freq[idx_50]:.2f} Hz:")
        for i in range(n_cond):
            g = ulm_params.gamma_modes[idx_50, i]
            v_phase = 2 * np.pi * freq[idx_50] / np.abs(np.imag(g)) \
                if np.abs(np.imag(g)) > 1e-30 else np.inf
            v_ratio = v_phase / C_LIGHT
            h_mag = np.abs(ulm_params.H_modes[idx_50, i])
            print(f"    Mode {i+1}: γ = {np.real(g):.6e} + j{np.imag(g):.6e}  "
                  f"|H| = {h_mag:.6f}  v/c = {v_ratio:.4f}")

        if fitting_result.H_modes_fits:
            print(f"\n  H 模态拟合详情:")
            for fit in fitting_result.H_modes_fits:
                if fit is not None:
                    print(f"    Mode {fit.mode_index + 1}: "
                          f"{len(fit.poles)} 极点, "
                          f"RMSE = {fit.rmse * 100:.4f}%, "
                          f"τ = {fit.tau * 1e6:.4f} μs")

        if fitting_result.H_reconstruction_metrics is not None:
            m = fitting_result.H_reconstruction_metrics
            print(f"\n  [论文公式验证]")
            print(f"    Eq.(12)(13) RMSE: {m.method1_rmse * 100:.6f}%")
            print(f"    D 矩阵完备性误差: {m.D_identity_error:.2e}")

    return {
        'Yc_phase':       ulm_params.Yc_matrix,
        'H_phase':        ulm_params.H_matrix,
        'H_mode':         ulm_params.H_modes,
        'gamma_mode':     ulm_params.gamma_modes,
        'T_I':            ulm_params.TI_matrix,
        'T_ref':          ulm_params.T_ref,
        'T_ref_inv':      ulm_params.T_ref_inv,
        'D_matrices':     ulm_params.D_matrices,
        'Yc_trace':       ulm_params.Yc_trace,
        'tau':            ulm_params.tau,
        'ulm_params':     ulm_params,
        'fitting_result': fitting_result,
    }


# =============================================================================
# 第五部分：从 PSCAD 读取参考数据
# =============================================================================
def load_Yc_H_pscad(modules: Dict,
                     pscad_path: str,
                     verbose: bool = True
                     ) -> Dict:
    """
    从 PSCAD 输出文件读取全部参考数据

    Returns
    -------
    dict 包含 freq_pscad, Z_pscad, Y_pscad, Yc_pscad, H_phase_pscad,
             H_mode_pscad, Lambda_pscad, Ti_pscad, reader, ...
    """
    PSCADFileReader = modules['PSCADFileReader']

    if verbose:
        print("\n" + "=" * 65)
        print(" 从 PSCAD 输出文件读取参考数据")
        print("=" * 65)

    reader = PSCADFileReader(pscad_path)

    if verbose:
        print(f"  文件前缀: {reader.file_prefix}")
        print(f"  检测导体数: {reader.n_conductors}")

    file_status = reader.check_files()
    for name, ok in file_status.items():
        mark = "✓" if ok else "✗"
        if verbose:
            print(f"    {mark} {name}")

    result = {
        'reader': reader,
        'file_status': file_status,
    }

    # ---- 读取频率 (从 Z 文件获得基准) ----
    if file_status.get('impedance', False):
        z_data = reader.get_impedance()
        result['freq_pscad'] = z_data.frequency
    elif file_status.get('char_admittance', False):
        yc_data = reader.get_char_admittance()
        result['freq_pscad'] = yc_data.frequency
    else:
        raise FileNotFoundError("PSCAD 文件中既无 Z 也无 Yc 数据")

    freq_pscad = result['freq_pscad']
    if verbose:
        print(f"\n  频率范围: {freq_pscad[0]:.4f} ~ {freq_pscad[-1]:.2e} Hz "
              f"({len(freq_pscad)} 点)")

    # ---- Z (阻抗矩阵) ----
    if file_status.get('impedance', False):
        z_data = reader.get_impedance()
        result['Z_pscad'] = z_data.complex_data
        result['Z_pscad_mag'] = z_data.magnitude
        result['Z_pscad_phase'] = z_data.phase_deg
        if verbose:
            print(f"  Z (阻抗矩阵): {z_data.complex_data.shape}  "
                  f"单位: {z_data.unit}")
    else:
        result['Z_pscad'] = None

    # ---- Y (导纳矩阵) ----
    if file_status.get('admittance', False):
        y_data = reader.get_admittance()
        result['Y_pscad'] = y_data.complex_data
        result['Y_pscad_mag'] = y_data.magnitude
        result['Y_pscad_phase'] = y_data.phase_deg
        if verbose:
            print(f"  Y (导纳矩阵): {y_data.complex_data.shape}  "
                  f"单位: {y_data.unit}")
    else:
        result['Y_pscad'] = None

    # ---- Yc (相域矩阵) ----
    if file_status.get('char_admittance', False):
        yc_data = reader.get_char_admittance()
        result['Yc_pscad'] = yc_data.complex_data
        result['Yc_pscad_mag'] = yc_data.magnitude
        result['Yc_pscad_phase'] = yc_data.phase_deg
        result['Yc_has_fitted'] = yc_data.has_fitted
        result['Yc_fitted'] = yc_data.complex_fitted
        result['Yc_mag_fitted'] = yc_data.magnitude_fitted
        result['Yc_phase_fitted'] = yc_data.phase_fitted
        if verbose:
            print(f"  Yc (相域): {yc_data.complex_data.shape}  "
                  f"(含拟合: {yc_data.has_fitted})")
    else:
        result['Yc_pscad'] = None

    # ---- H 模态域 ----
    if file_status.get('h_mode', False):
        h_mode_data = reader.get_h_mode()
        result['H_mode_pscad'] = h_mode_data.complex_data
        result['H_mode_pscad_mag'] = h_mode_data.magnitude
        result['H_mode_pscad_phase'] = h_mode_data.phase_deg
        result['H_mode_has_fitted'] = h_mode_data.has_fitted
        result['H_mode_fitted'] = h_mode_data.complex_fitted
        result['H_mode_mag_fitted'] = h_mode_data.magnitude_fitted
        result['H_mode_phase_fitted'] = h_mode_data.phase_fitted
        if verbose:
            n_modes = h_mode_data.magnitude.shape[1] if h_mode_data.magnitude.ndim == 2 else 1
            print(f"  H (模态域): {n_modes} 模态  (含拟合: {h_mode_data.has_fitted})")
    else:
        result['H_mode_pscad'] = None

    # ---- H 相域 ----
    if file_status.get('h_phase', False):
        h_phase_data = reader.get_h_phase()
        result['H_phase_pscad'] = h_phase_data.complex_data
        result['H_phase_pscad_mag'] = h_phase_data.magnitude
        result['H_phase_pscad_phase'] = h_phase_data.phase_deg
        result['H_phase_has_fitted'] = h_phase_data.has_fitted
        result['H_phase_fitted'] = h_phase_data.complex_fitted
        if verbose:
            print(f"  H (相域): {h_phase_data.complex_data.shape}  "
                  f"(含拟合: {h_phase_data.has_fitted})")
    else:
        result['H_phase_pscad'] = None

    # ---- Lambda (传播常数) ----
    if file_status.get('propagation', False):
        lam_data = reader.get_propagation_constant()
        result['Lambda_pscad'] = lam_data.complex_data
        result['Lambda_pscad_mag'] = lam_data.magnitude
        result['Lambda_pscad_phase'] = lam_data.phase_deg
        if verbose:
            shape = lam_data.complex_data.shape
            print(f"  Lambda (传播常数): {shape}")
    else:
        result['Lambda_pscad'] = None

    # ---- Ti (变换矩阵) ----
    if file_status.get('transform', False):
        ti_data = reader.get_transform_matrix()
        result['Ti_pscad'] = ti_data.complex_data
        if verbose:
            print(f"  Ti (变换矩阵): {ti_data.complex_data.shape}")
    else:
        result['Ti_pscad'] = None

    return result


# =============================================================================
# 第六部分：误差计算
# =============================================================================
def compute_element_error(calc: np.ndarray,
                          ref: np.ndarray,
                          min_ratio: float = 0.01) -> Dict:
    """计算单个元素的幅值与相角误差"""
    mag_calc = np.abs(calc)
    mag_ref = np.abs(ref)
    phase_calc = np.angle(calc, deg=True)
    phase_ref = np.angle(ref, deg=True)

    threshold = np.max(mag_ref) * min_ratio
    valid = mag_ref > threshold

    mag_err = np.zeros_like(mag_ref)
    mag_err[valid] = np.abs(mag_calc[valid] - mag_ref[valid]) / mag_ref[valid] * 100

    phase_err = np.abs(phase_calc - phase_ref)
    phase_err = np.minimum(phase_err, 360 - phase_err)
    phase_err[~valid] = 0

    return {
        'mag_err_pct':   mag_err,
        'phase_err_deg': phase_err,
        'valid_mask':    valid,
        'max_mag_err':   np.max(mag_err[valid]) if np.any(valid) else 0.0,
        'mean_mag_err':  np.mean(mag_err[valid]) if np.any(valid) else 0.0,
        'max_phase_err': np.max(phase_err[valid]) if np.any(valid) else 0.0,
        'mean_phase_err':np.mean(phase_err[valid]) if np.any(valid) else 0.0,
    }


def compute_matrix_error(M_calc: np.ndarray,
                          M_ref: np.ndarray,
                          names: List[str],
                          label: str = "Yc",
                          verbose: bool = True) -> Dict:
    """计算整个矩阵的误差统计"""
    n = M_calc.shape[1]
    results = {}

    if verbose:
        print(f"\n  {label} 矩阵误差统计:")
        print(f"  {'元素':>12s} | {'最大幅值误差%':>14s} | {'平均幅值误差%':>14s} | "
              f"{'最大相角误差°':>14s} | {'平均相角误差°':>14s}")
        print("  " + "-" * 80)

    for i in range(n):
        for j in range(n):
            key = f'{label}{i+1}{j+1}'
            err = compute_element_error(M_calc[:, i, j], M_ref[:, i, j])
            results[key] = err
            if verbose:
                print(f"  {key:>12s} | {err['max_mag_err']:>14.4f} | "
                      f"{err['mean_mag_err']:>14.4f} | "
                      f"{err['max_phase_err']:>14.4f} | "
                      f"{err['mean_phase_err']:>14.4f}")

    all_max_mag = max(v['max_mag_err'] for v in results.values())
    all_mean_mag = np.mean([v['mean_mag_err'] for v in results.values()])
    all_max_phase = max(v['max_phase_err'] for v in results.values())

    if verbose:
        print("  " + "-" * 80)
        print(f"  {'全局':>12s} | {all_max_mag:>14.4f} | {all_mean_mag:>14.4f} | "
              f"{all_max_phase:>14.4f} |")

    results['_global'] = {
        'max_mag_err':  all_max_mag,
        'mean_mag_err': all_mean_mag,
        'max_phase_err': all_max_phase,
    }
    return results


def compute_vector_error(calc: np.ndarray,
                          ref: np.ndarray,
                          mode_names: List[str],
                          label: str = "H_mode",
                          verbose: bool = True) -> Dict:
    """计算模态向量的误差 (一维: 每列为一个模态)"""
    n_modes = calc.shape[1]
    results = {}

    if verbose:
        print(f"\n  {label} 向量误差统计:")
        print(f"  {'模态':>12s} | {'最大幅值误差%':>14s} | {'平均幅值误差%':>14s} | "
              f"{'最大相角误差°':>14s} | {'平均相角误差°':>14s}")
        print("  " + "-" * 80)

    for i in range(n_modes):
        key = f'{label}_{i+1}'
        err = compute_element_error(calc[:, i], ref[:, i])
        results[key] = err
        if verbose:
            name = mode_names[i] if i < len(mode_names) else f'Mode{i+1}'
            print(f"  {name:>12s} | {err['max_mag_err']:>14.4f} | "
                  f"{err['mean_mag_err']:>14.4f} | "
                  f"{err['max_phase_err']:>14.4f} | "
                  f"{err['mean_phase_err']:>14.4f}")

    all_max_mag = max(v['max_mag_err'] for v in results.values())
    all_mean_mag = np.mean([v['mean_mag_err'] for v in results.values()])
    all_max_phase = max(v['max_phase_err'] for v in results.values())

    if verbose:
        print("  " + "-" * 80)
        print(f"  {'全局':>12s} | {all_max_mag:>14.4f} | {all_mean_mag:>14.4f} | "
              f"{all_max_phase:>14.4f} |")

    results['_global'] = {
        'max_mag_err':  all_max_mag,
        'mean_mag_err': all_mean_mag,
        'max_phase_err': all_max_phase,
    }
    return results


# =============================================================================
# 第七部分：Ti 矩阵对比工具
# =============================================================================
def normalize_Ti_columns(Ti: np.ndarray) -> np.ndarray:
    """对变换矩阵的每列进行归一化"""
    n = Ti.shape[0]
    Ti_norm = Ti.copy()
    for j in range(n):
        col = Ti_norm[:, j]
        nrm = np.linalg.norm(col)
        if nrm > 1e-30:
            col = col / nrm
        idx_max = np.argmax(np.abs(col))
        phase_max = np.angle(col[idx_max])
        col = col * np.exp(-1j * phase_max)
        Ti_norm[:, j] = col
    return Ti_norm


def align_Ti_columns(Ti_calc: np.ndarray, Ti_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对齐两个变换矩阵的列顺序和符号"""
    n = Ti_calc.shape[0]
    Ti_c = normalize_Ti_columns(Ti_calc)
    Ti_r = normalize_Ti_columns(Ti_ref)

    corr = np.abs(Ti_c.conj().T @ Ti_r)

    perm = np.full(n, -1, dtype=int)
    used_ref = set()
    for _ in range(n):
        best_val = -1
        best_i, best_j = -1, -1
        for i in range(n):
            if perm[i] >= 0:
                continue
            for j in range(n):
                if j in used_ref:
                    continue
                if corr[i, j] > best_val:
                    best_val = corr[i, j]
                    best_i, best_j = i, j
        perm[best_i] = best_j
        used_ref.add(best_j)

    Ti_ref_aligned = Ti_r[:, perm]
    return Ti_c, Ti_ref_aligned, perm


def compute_Ti_comparison(Ti_calc_all: np.ndarray,
                          Ti_ref_all: np.ndarray,
                          freq: np.ndarray,
                          names: List[str],
                          verbose: bool = True) -> Dict:
    """计算 Ti 矩阵的对比误差"""
    K = len(freq)
    n_calc = Ti_calc_all.shape[1]
    n_ref = Ti_ref_all.shape[1]
    n = min(n_calc, n_ref)

    if verbose:
        if n_calc != n_ref:
            print(f"\n  ⚠ Ti 维度不匹配: 自研 {n_calc}×{n_calc}, PSCAD {n_ref}×{n_ref}")
            print(f"  → 取前 {n}×{n} 进行对比 (需注意物理含义)")

    mode_corr = np.zeros((K, n))
    col_errors = np.zeros((K, n))

    for k in range(K):
        Ti_c = Ti_calc_all[k, :n, :n]
        Ti_r = Ti_ref_all[k, :n, :n]
        Ti_c_norm, Ti_r_aligned, _ = align_Ti_columns(Ti_c, Ti_r)

        for j in range(n):
            inner = np.abs(np.vdot(Ti_c_norm[:, j], Ti_r_aligned[:, j]))
            mode_corr[k, j] = inner
            col_errors[k, j] = np.linalg.norm(Ti_c_norm[:, j] - Ti_r_aligned[:, j])

    summary = {
        'n_calc': n_calc,
        'n_ref': n_ref,
        'n_compared': n,
        'mean_correlation': np.mean(mode_corr),
        'min_correlation': np.min(mode_corr),
        'max_column_error': np.max(col_errors),
        'mean_column_error': np.mean(col_errors),
    }

    if verbose:
        print(f"\n  Ti 矩阵对比统计 (归一化列向量比较):")
        print(f"  {'模态':>10s} | {'平均相关系数':>14s} | {'最小相关系数':>14s} | "
              f"{'平均列误差':>12s} | {'最大列误差':>12s}")
        print("  " + "-" * 75)
        for j in range(n):
            print(f"  Mode {j+1:>4d} | {np.mean(mode_corr[:, j]):>14.6f} | "
                  f"{np.min(mode_corr[:, j]):>14.6f} | "
                  f"{np.mean(col_errors[:, j]):>12.6f} | "
                  f"{np.max(col_errors[:, j]):>12.6f}")
        print("  " + "-" * 75)
        print(f"  {'全局':>10s} | {summary['mean_correlation']:>14.6f} | "
              f"{summary['min_correlation']:>14.6f} | "
              f"{summary['mean_column_error']:>12.6f} | "
              f"{summary['max_column_error']:>12.6f}")

    return {
        'mode_correlations': mode_corr,
        'column_errors': col_errors,
        'summary': summary,
    }


# =============================================================================
# 第八部分：绘图工具
# =============================================================================

# 导体标签和颜色 (7 导体)
COND_LABELS = ['Core1', 'Sheath1', 'Core2', 'Sheath2',
               'Core3', 'Sheath3', 'Pipe']
COND_COLORS = ['#E74C3C', '#E74C3C',
               '#27AE60', '#27AE60',
               '#2980B9', '#2980B9',
               '#7F8C8D']
COND_STYLES = ['-', '--', '-', '--', '-', '--', '-']


def plot_full_matrix_comparison(freq: np.ndarray,
                                M_calc: np.ndarray,
                                M_ref: Optional[np.ndarray],
                                names: List[str],
                                matrix_label: str = "Yc",
                                unit_mag: str = "S",
                                mag_scale: float = 1.0,
                                use_loglog_mag: bool = True,
                                save_path: Optional[str] = None):
    """
    绘制完整矩阵（n×n 所有元素）的幅值与相角对比图
    当 M_ref 不为 None 时绘制双曲线对比；否则仅绘制单曲线。
    """
    n = M_calc.shape[1]
    has_ref = M_ref is not None

    def get_name(idx):
        return names[idx] if idx < len(names) else f'C{idx+1}'

    # ---- 幅值对比图 ----
    fig_mag, axes_mag = plt.subplots(n, n, figsize=(4.2 * n, 3.2 * n), sharex=True)
    if n == 1:
        axes_mag = np.array([[axes_mag]])

    if has_ref:
        fig_mag.suptitle(f'Complete {matrix_label} Matrix — Magnitude Comparison\n'
                         f'(Blue solid = Calculated, Red dashed = PSCAD)',
                         fontsize=14, y=1.02)
    else:
        fig_mag.suptitle(f'Complete {matrix_label} Matrix — Magnitude\n'
                         f'Cable_14: Three-Core Pipe-Type Cable (7x7)',
                         fontsize=14, y=1.02)

    for i in range(n):
        for j in range(n):
            ax = axes_mag[i, j]
            mag_calc = np.abs(M_calc[:, i, j]) * mag_scale

            if use_loglog_mag:
                ax.loglog(freq, mag_calc, 'b-', lw=1.3, label='Calculated')
            else:
                ax.semilogx(freq, mag_calc, 'b-', lw=1.3, label='Calculated')

            if has_ref:
                mag_ref = np.abs(M_ref[:, i, j]) * mag_scale
                if use_loglog_mag:
                    ax.loglog(freq, mag_ref, 'r--', lw=1.0, label='PSCAD')
                else:
                    ax.semilogx(freq, mag_ref, 'r--', lw=1.0, label='PSCAD')

            ax.set_title(f'{matrix_label}({get_name(i)}, {get_name(j)})',
                         fontsize=9, pad=3)
            ax.grid(True, which='both', ls=':', alpha=0.4)
            ax.tick_params(labelsize=7)
            if i == j:
                ax.set_facecolor('#f0f8ff')
            if j == 0:
                ax.set_ylabel(f'|{matrix_label}| [{unit_mag}]', fontsize=8)
            if i == n - 1:
                ax.set_xlabel('Frequency [Hz]', fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best')

    fig_mag.tight_layout()
    if save_path:
        p = save_path.replace('.png', f'_{matrix_label}_full_magnitude.png')
        fig_mag.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    # ---- 相角对比图 ----
    fig_phase, axes_phase = plt.subplots(n, n, figsize=(4.2 * n, 3.2 * n), sharex=True)
    if n == 1:
        axes_phase = np.array([[axes_phase]])

    if has_ref:
        fig_phase.suptitle(f'Complete {matrix_label} Matrix — Phase Comparison\n'
                           f'(Blue solid = Calculated, Red dashed = PSCAD)',
                           fontsize=14, y=1.02)
    else:
        fig_phase.suptitle(f'Complete {matrix_label} Matrix — Phase\n'
                           f'Cable_14: Three-Core Pipe-Type Cable (7x7)',
                           fontsize=14, y=1.02)

    for i in range(n):
        for j in range(n):
            ax = axes_phase[i, j]
            phase_calc = np.angle(M_calc[:, i, j], deg=True)
            ax.semilogx(freq, phase_calc, 'b-', lw=1.3, label='Calculated')

            if has_ref:
                phase_ref = np.angle(M_ref[:, i, j], deg=True)
                ax.semilogx(freq, phase_ref, 'r--', lw=1.0, label='PSCAD')

            ax.set_title(f'∠{matrix_label}({get_name(i)}, {get_name(j)})',
                         fontsize=9, pad=3)
            ax.grid(True, which='both', ls=':', alpha=0.4)
            ax.tick_params(labelsize=7)
            if i == j:
                ax.set_facecolor('#f0f8ff')
            if j == 0:
                ax.set_ylabel(f'∠{matrix_label} [°]', fontsize=8)
            if i == n - 1:
                ax.set_xlabel('Frequency [Hz]', fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best')

    fig_phase.tight_layout()
    if save_path:
        p = save_path.replace('.png', f'_{matrix_label}_full_phase.png')
        fig_phase.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    return fig_mag, fig_phase


def plot_modal_H_comparison(freq, H_mode_calc, H_mode_ref, mode_names, save_path=None):
    """绘制模态传播函数 H 的对比图 (幅值 + 相角)"""
    n_modes_calc = H_mode_calc.shape[1]
    n_modes_ref = H_mode_ref.shape[1]
    n_modes = max(n_modes_calc, n_modes_ref)

    fig, axes = plt.subplots(n_modes, 2, figsize=(14, 3.5 * max(n_modes, 1)), sharex=True)
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    subtitle = 'Modal Propagation Function H — Calculated vs PSCAD'
    if n_modes_calc != n_modes_ref:
        subtitle += f'\n(Calc: {n_modes_calc} modes, PSCAD: {n_modes_ref} modes)'
    fig.suptitle(subtitle, fontsize=13, y=1.0)

    for i in range(n_modes):
        name = mode_names[i] if i < len(mode_names) else f'Mode {i+1}'
        has_calc = i < n_modes_calc
        has_ref = i < n_modes_ref

        ax = axes[i, 0]
        if has_calc:
            ax.semilogx(freq, np.abs(H_mode_calc[:, i]), 'b-', lw=1.5, label='Calculated')
        if has_ref:
            ax.semilogx(freq, np.abs(H_mode_ref[:, i]), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel(f'|H| Mode {i+1}')
        suffix = ''
        if has_calc and not has_ref: suffix = ' (Calc only)'
        elif has_ref and not has_calc: suffix = ' (PSCAD only)'
        ax.set_title(f'{name} — Magnitude{suffix}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.set_ylim([0, 1.1])

        ax = axes[i, 1]
        if has_calc:
            ax.semilogx(freq, np.angle(H_mode_calc[:, i], deg=True), 'b-', lw=1.5, label='Calculated')
        if has_ref:
            ax.semilogx(freq, np.angle(H_mode_ref[:, i], deg=True), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel(f'∠H Mode {i+1} [°]')
        ax.set_title(f'{name} — Phase{suffix}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls=':', alpha=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel('Frequency [Hz]')
    fig.tight_layout()

    if save_path:
        p = save_path.replace('.png', '_H_mode_comparison.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")
    return fig


def plot_lambda_comparison(freq, gamma_calc, lambda_ref, mode_names, save_path=None):
    """绘制传播常数 λ = γ² 对比图 (实部 + 虚部)"""
    lambda_calc = gamma_calc ** 2
    n_modes = lambda_calc.shape[1]

    fig, axes = plt.subplots(2, n_modes, figsize=(5 * n_modes, 8), sharex=True)
    if n_modes == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle('Eigenvalue λ = γ²  — Calculated vs PSCAD\n'
                 '(λ_calc = γ_calc²,  λ_PSCAD from Lambda file)',
                 fontsize=13, y=1.02)

    for i in range(n_modes):
        name = mode_names[i] if i < len(mode_names) else f'Mode {i+1}'
        ax = axes[0, i]
        ax.loglog(freq, np.abs(np.real(lambda_calc[:, i])), 'b-', lw=1.5, label='Calc (γ²)')
        ax.loglog(freq, np.abs(np.real(lambda_ref[:, i])), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('|Re(λ)| [1/m²]')
        ax.set_title(f'{name} — Re(λ)')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        ax = axes[1, i]
        ax.loglog(freq, np.abs(np.imag(lambda_calc[:, i])), 'b-', lw=1.5, label='Calc (γ²)')
        ax.loglog(freq, np.abs(np.imag(lambda_ref[:, i])), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('|Im(λ)| [1/m²]')
        ax.set_title(f'{name} — Im(λ)')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_lambda_comparison.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")
    return fig


def plot_Ti_matrix_elements(freq, Ti_calc, Ti_ref, names, save_path=None):
    """绘制 Ti 矩阵每个元素的幅值和相角对比图"""
    n_calc = Ti_calc.shape[1]
    n_ref = Ti_ref.shape[1]
    n = min(n_calc, n_ref)

    def get_name(idx):
        return names[idx] if idx < len(names) else f'C{idx+1}'

    fig_mag, axes_mag = plt.subplots(n, n, figsize=(4.2 * n, 3.2 * n), sharex=True)
    if n == 1: axes_mag = np.array([[axes_mag]])
    fig_mag.suptitle(f'Ti Matrix — Magnitude Comparison (n={n})\n'
                     f'(Blue solid = Calculated, Red dashed = PSCAD)', fontsize=14, y=1.02)
    for i in range(n):
        for j in range(n):
            ax = axes_mag[i, j]
            ax.semilogx(freq, np.abs(Ti_calc[:, i, j]), 'b-', lw=1.3, label='Calculated')
            ax.semilogx(freq, np.abs(Ti_ref[:, i, j]), 'r--', lw=1.0, label='PSCAD')
            ax.set_title(f'Ti({get_name(i)}, {get_name(j)})', fontsize=9, pad=3)
            ax.grid(True, which='both', ls=':', alpha=0.4)
            ax.tick_params(labelsize=7)
            if i == j: ax.set_facecolor('#f0f8ff')
            if j == 0: ax.set_ylabel('|Ti|', fontsize=8)
            if i == n - 1: ax.set_xlabel('Frequency [Hz]', fontsize=8)
            if i == 0 and j == 0: ax.legend(fontsize=7, loc='best')
    fig_mag.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_Ti_full_magnitude.png')
        fig_mag.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    fig_phase, axes_phase = plt.subplots(n, n, figsize=(4.2 * n, 3.2 * n), sharex=True)
    if n == 1: axes_phase = np.array([[axes_phase]])
    fig_phase.suptitle(f'Ti Matrix — Phase Comparison (n={n})\n'
                       f'(Blue solid = Calculated, Red dashed = PSCAD)', fontsize=14, y=1.02)
    for i in range(n):
        for j in range(n):
            ax = axes_phase[i, j]
            ax.semilogx(freq, np.angle(Ti_calc[:, i, j], deg=True), 'b-', lw=1.3, label='Calculated')
            ax.semilogx(freq, np.angle(Ti_ref[:, i, j], deg=True), 'r--', lw=1.0, label='PSCAD')
            ax.set_title(f'∠Ti({get_name(i)}, {get_name(j)})', fontsize=9, pad=3)
            ax.grid(True, which='both', ls=':', alpha=0.4)
            ax.tick_params(labelsize=7)
            if i == j: ax.set_facecolor('#f0f8ff')
            if j == 0: ax.set_ylabel('∠Ti [°]', fontsize=8)
            if i == n - 1: ax.set_xlabel('Frequency [Hz]', fontsize=8)
            if i == 0 and j == 0: ax.legend(fontsize=7, loc='best')
    fig_phase.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_Ti_full_phase.png')
        fig_phase.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    return fig_mag, fig_phase


def plot_Ti_column_correlation(freq, mode_corr, col_errors, save_path=None):
    """绘制 Ti 各模态列向量的相关系数和列误差随频率变化"""
    n_modes = mode_corr.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Ti Column Vector Alignment Quality\n'
                 '(after normalization and sign alignment)', fontsize=13, y=1.0)

    ax = axes[0]
    for j in range(n_modes):
        ax.semilogx(freq, mode_corr[:, j], lw=1.5, label=f'Mode {j+1}')
    ax.axhline(1.0, color='k', ls='--', alpha=0.3)
    ax.set_ylabel('Column Correlation |⟨c, r⟩|')
    ax.set_title('Normalized Column Inner Product (ideal = 1.0)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.set_ylim([max(0, np.min(mode_corr) - 0.05), 1.05])

    ax = axes[1]
    for j in range(n_modes):
        ax.semilogx(freq, col_errors[:, j], lw=1.5, label=f'Mode {j+1}')
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.set_ylabel('Column Error ‖c − r‖₂')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_title('Normalized Column Difference (ideal = 0)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', ls=':', alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_Ti_column_quality.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")
    return fig


def plot_Yc_self_comparison(freq, Yc_calc, Yc_ref, names, save_path=None):
    """绘制 Yc 自导纳 (对角元素) 详细对比 + 等效 Zc"""
    n_cond = Yc_calc.shape[1]
    n_show = min(n_cond, 7)

    fig, axes = plt.subplots(3, n_show, figsize=(3.5 * n_show, 10), sharex=True)
    if n_show == 1: axes = axes.reshape(-1, 1)
    fig.suptitle('Characteristic Admittance Yc — Self Elements Comparison', fontsize=13, y=1.0)

    for i in range(n_show):
        name = names[i] if i < len(names) else f'C{i+1}'
        ax = axes[0, i]
        ax.semilogx(freq, np.abs(Yc_calc[:, i, i]) * 1e3, 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, np.abs(Yc_ref[:, i, i]) * 1e3, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('|Yc| [mS]')
        ax.set_title(f'{name}')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        ax = axes[1, i]
        ax.semilogx(freq, np.angle(Yc_calc[:, i, i], deg=True), 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, np.angle(Yc_ref[:, i, i], deg=True), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('∠Yc [°]')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        with np.errstate(divide='ignore', invalid='ignore'):
            Zc_calc = np.where(np.abs(Yc_calc[:, i, i]) > 1e-30, 1.0 / np.abs(Yc_calc[:, i, i]), np.nan)
            Zc_ref = np.where(np.abs(Yc_ref[:, i, i]) > 1e-30, 1.0 / np.abs(Yc_ref[:, i, i]), np.nan)
        ax = axes[2, i]
        ax.semilogx(freq, Zc_calc, 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, Zc_ref, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('|Zc| [Ω]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_Yc_self_detail.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")
    return fig


def plot_error_vs_freq(freq, Yc_calc, Yc_ref, H_calc, H_ref, names, save_path=None):
    """绘制 Yc 和 H 的全矩阵误差随频率变化曲线"""
    n_cond = Yc_calc.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Yc / H Full Matrix Error vs Frequency', fontsize=13)

    for panel_idx, (M_c, M_r, lbl) in enumerate([(Yc_calc, Yc_ref, 'Yc'), (H_calc, H_ref, 'H')]):
        for col, mode in enumerate(['mag', 'phase']):
            ax = axes[panel_idx, col]
            for i in range(n_cond):
                for j in range(n_cond):
                    if mode == 'mag':
                        mag_r = np.abs(M_r[:, i, j])
                        mag_c = np.abs(M_c[:, i, j])
                        threshold = np.max(mag_r) * 0.01
                        valid = mag_r > threshold
                        err = np.zeros_like(mag_r)
                        err[valid] = np.abs(mag_c[valid] - mag_r[valid]) / mag_r[valid] * 100
                    else:
                        ph_c = np.angle(M_c[:, i, j], deg=True)
                        ph_r = np.angle(M_r[:, i, j], deg=True)
                        err = np.abs(ph_c - ph_r)
                        err = np.minimum(err, 360 - err)
                    name_i = names[i] if i < len(names) else f'C{i+1}'
                    name_j = names[j] if j < len(names) else f'C{j+1}'
                    ls = '-' if i == j else '--'
                    ax.semilogx(freq, err, ls=ls, lw=1.2, label=f'{lbl}({name_i},{name_j})')
            ylabel = 'Magnitude Error [%]' if mode == 'mag' else 'Phase Error [°]'
            ax.set_ylabel(ylabel)
            ax.set_title(f'{lbl} Full Matrix {"Magnitude" if mode == "mag" else "Phase"} Error')
            if panel_idx == 1: ax.set_xlabel('Frequency [Hz]')
            ax.legend(fontsize=6, ncol=2, loc='best')
            ax.grid(True, which='both', ls=':', alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_Yc_H_error.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")
    return fig


def plot_ZY_comparison(freq, Z_calc, Z_ref, Y_calc, Y_ref, names, save_path=None):
    """生成 Z/Y 全矩阵对比图 (自阻抗、互阻抗、RLC、误差、全矩阵)"""
    n_cond = Z_calc.shape[1]
    omega = 2 * np.pi * freq
    all_figs = []

    # 图1: 自阻抗对比
    fig1, axes1 = plt.subplots(n_cond, 2, figsize=(14, 3.5 * n_cond), sharex=True)
    if n_cond == 1: axes1 = axes1.reshape(1, -1)
    fig1.suptitle('Self Impedance Comparison: Calculated vs PSCAD', fontsize=13, y=1.0)
    for i in range(n_cond):
        name = names[i] if i < len(names) else f'Cond{i+1}'
        ax = axes1[i, 0]
        ax.loglog(freq, np.abs(Z_calc[:, i, i]) * 1000, 'b-', lw=1.5, label='Calculated')
        ax.loglog(freq, np.abs(Z_ref[:, i, i]) * 1000, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel(f'|Z{i+1}{i+1}| [mΩ/m]')
        ax.set_title(f'{name} — Magnitude')
        ax.legend(fontsize=8); ax.grid(True, which='both', ls=':', alpha=0.5)
        ax = axes1[i, 1]
        ax.semilogx(freq, np.angle(Z_calc[:, i, i], deg=True), 'b-', lw=1.5, label='Calculated')
        ax.semilogx(freq, np.angle(Z_ref[:, i, i], deg=True), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel(f'∠Z{i+1}{i+1} [°]')
        ax.set_title(f'{name} — Phase')
        ax.legend(fontsize=8); ax.grid(True, which='both', ls=':', alpha=0.5)
    for ax in axes1[-1, :]: ax.set_xlabel('Frequency [Hz]')
    fig1.tight_layout(); all_figs.append(fig1)
    if save_path:
        p = save_path.replace('.png', '_ZY_1_Zself.png')
        fig1.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")

    # 图2: 互阻抗对比
    pairs = [(i, j) for i in range(min(n_cond, 4)) for j in range(i+1, min(n_cond, 4))]
    if pairs:
        n_pairs = len(pairs)
        fig2, axes2 = plt.subplots(n_pairs, 2, figsize=(14, 3.5 * max(n_pairs, 1)), sharex=True)
        if n_pairs == 1: axes2 = axes2.reshape(1, -1)
        fig2.suptitle('Mutual Impedance Comparison: Calculated vs PSCAD', fontsize=13, y=1.0)
        for idx, (i, j) in enumerate(pairs):
            ax = axes2[idx, 0]
            ax.loglog(freq, np.abs(Z_calc[:, i, j]) * 1000, 'b-', lw=1.5, label='Calculated')
            ax.loglog(freq, np.abs(Z_ref[:, i, j]) * 1000, 'r--', lw=1.2, label='PSCAD')
            ax.set_ylabel(f'|Z{i+1}{j+1}| [mΩ/m]'); ax.set_title(f'Z{i+1}{j+1} — Magnitude')
            ax.legend(fontsize=8); ax.grid(True, which='both', ls=':', alpha=0.5)
            ax = axes2[idx, 1]
            ax.semilogx(freq, np.angle(Z_calc[:, i, j], deg=True), 'b-', lw=1.5, label='Calculated')
            ax.semilogx(freq, np.angle(Z_ref[:, i, j], deg=True), 'r--', lw=1.2, label='PSCAD')
            ax.set_ylabel(f'∠Z{i+1}{j+1} [°]'); ax.set_title(f'Z{i+1}{j+1} — Phase')
            ax.legend(fontsize=8); ax.grid(True, which='both', ls=':', alpha=0.5)
        for ax in axes2[-1, :]: ax.set_xlabel('Frequency [Hz]')
        fig2.tight_layout(); all_figs.append(fig2)
        if save_path:
            p = save_path.replace('.png', '_ZY_2_Zmutual.png')
            fig2.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")

    # 图3: 等效 R/L/C
    n_show = min(n_cond, 7)
    fig3, axes3 = plt.subplots(3, n_show, figsize=(3.5 * n_show, 10), sharex=True)
    if n_show == 1: axes3 = axes3.reshape(-1, 1)
    fig3.suptitle('Equivalent R / L / C Comparison', fontsize=13, y=1.0)
    for i in range(n_show):
        name = names[i] if i < len(names) else f'C{i+1}'
        ax = axes3[0, i]
        ax.loglog(freq, np.abs(np.real(Z_calc[:, i, i]) * 1000), 'b-', lw=1.5, label='Calc')
        ax.loglog(freq, np.abs(np.real(Z_ref[:, i, i]) * 1000), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('R [mΩ/m]'); ax.set_title(f'{name}')
        ax.legend(fontsize=7); ax.grid(True, which='both', ls=':', alpha=0.5)
        with np.errstate(divide='ignore', invalid='ignore'):
            L_calc = np.imag(Z_calc[:, i, i]) / omega * 1e6
            L_ref = np.imag(Z_ref[:, i, i]) / omega * 1e6
        ax = axes3[1, i]
        ax.semilogx(freq, L_calc, 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, L_ref, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('L [μH/m]'); ax.legend(fontsize=7); ax.grid(True, which='both', ls=':', alpha=0.5)
        with np.errstate(divide='ignore', invalid='ignore'):
            C_calc = np.imag(Y_calc[:, i, i]) / omega * 1e12
            C_ref = np.imag(Y_ref[:, i, i]) / omega * 1e12
        ax = axes3[2, i]
        ax.semilogx(freq, C_calc, 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, C_ref, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('C [pF/m]'); ax.set_xlabel('Frequency [Hz]')
        ax.legend(fontsize=7); ax.grid(True, which='both', ls=':', alpha=0.5)
    fig3.tight_layout(); all_figs.append(fig3)
    if save_path:
        p = save_path.replace('.png', '_ZY_3_RLC.png')
        fig3.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")

    # 图4: Z/Y 误差
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('Z / Y Full Matrix Error vs Frequency', fontsize=13)
    for panel_idx, (Mc, Mr, lbl) in enumerate([(Z_calc, Z_ref, 'Z'), (Y_calc, Y_ref, 'Y')]):
        for col, mode in enumerate(['mag', 'phase']):
            ax = axes4[panel_idx, col]
            for i in range(n_cond):
                for j in range(n_cond):
                    if mode == 'mag':
                        mag_r = np.abs(Mr[:, i, j]); mag_c = np.abs(Mc[:, i, j])
                        thr = np.max(mag_r) * 0.01; valid = mag_r > thr
                        err = np.zeros_like(mag_r); err[valid] = np.abs(mag_c[valid] - mag_r[valid]) / mag_r[valid] * 100
                    else:
                        ph_c = np.angle(Mc[:, i, j], deg=True); ph_r = np.angle(Mr[:, i, j], deg=True)
                        err = np.abs(ph_c - ph_r); err = np.minimum(err, 360 - err)
                    nm_i = names[i] if i < len(names) else f'C{i+1}'
                    nm_j = names[j] if j < len(names) else f'C{j+1}'
                    ls = '-' if i == j else '--'
                    ax.semilogx(freq, err, ls=ls, lw=1.2, label=f'{lbl}({nm_i},{nm_j})')
            ax.set_ylabel('Magnitude Error [%]' if mode == 'mag' else 'Phase Error [°]')
            ax.set_title(f'{lbl} {"Magnitude" if mode == "mag" else "Phase"} Error')
            if panel_idx == 1: ax.set_xlabel('Frequency [Hz]')
            ax.legend(fontsize=6, ncol=2, loc='best'); ax.grid(True, which='both', ls=':', alpha=0.5)
    fig4.tight_layout(); all_figs.append(fig4)
    if save_path:
        p = save_path.replace('.png', '_ZY_4_Error.png')
        fig4.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")

    # 图5-6: Z 全矩阵
    fig5m, fig5p = plot_full_matrix_comparison(freq, Z_calc, Z_ref, names,
        matrix_label="Z", unit_mag="mΩ/m", mag_scale=1000.0, use_loglog_mag=True, save_path=save_path)
    all_figs.extend([fig5m, fig5p])

    # 图7-8: Y 全矩阵
    fig6m, fig6p = plot_full_matrix_comparison(freq, Y_calc, Y_ref, names,
        matrix_label="Y", unit_mag="μS/m", mag_scale=1e6, use_loglog_mag=True, save_path=save_path)
    all_figs.extend([fig6m, fig6p])

    return all_figs


def plot_VF_Yc_calc_vs_fitted(freq, calc_results, names, save_path=None):
    """绘制 VF 自研 Yc 计算值 vs VF 拟合值对比图"""
    fitting_result = calc_results.get('fitting_result')
    Yc_calc = calc_results.get('Yc_phase')
    if fitting_result is None or Yc_calc is None: return None
    poles_Yc = getattr(fitting_result, 'poles_Yc', None)
    k0 = getattr(fitting_result, 'k0', None)
    k_residues = getattr(fitting_result, 'k_residues', None)
    if poles_Yc is None or k0 is None or k_residues is None:
        print("  ⚠ VF Yc 拟合数据不完整，跳过 Yc calc vs fitted 图"); return None

    n_cond = Yc_calc.shape[1]; K = len(freq)
    s = 1j * 2 * np.pi * freq
    Yc_fitted = np.zeros((K, n_cond, n_cond), dtype=complex)
    for k_idx in range(K):
        Yk = k0.astype(complex).copy()
        for n_p, p in enumerate(poles_Yc):
            if np.isreal(p): Yk += k_residues[n_p].real / (s[k_idx] - p)
            else:
                Yk += k_residues[n_p] / (s[k_idx] - p)
                Yk += k_residues[n_p].conj() / (s[k_idx] - p.conj())
        Yc_fitted[k_idx] = Yk

    n_show = min(n_cond, 6)
    fig, axes = plt.subplots(2, n_show, figsize=(4.5 * n_show, 7), sharex=True)
    if n_show == 1: axes = axes.reshape(-1, 1)
    fig.suptitle(f'VF Yc: Calculated vs Vector Fitting Result\n'
                 f'(poles = {len(poles_Yc)}, RMSE = {fitting_result.Yc_trace_rmse * 100:.3f}%)', fontsize=13)
    for i in range(n_show):
        name = names[i] if i < len(names) else f'C{i+1}'
        ax = axes[0, i]
        ax.semilogx(freq, np.abs(Yc_fitted[:, i, i]) * 1e3, 'r--', lw=2.5, label='VF Fit')
        ax.semilogx(freq, np.abs(Yc_calc[:, i, i]) * 1e3, 'b-', lw=1.2, label='VF Calc')
        ax.set_ylabel(f'|Yc{i+1}{i+1}| [mS]'); ax.set_title(f'Yc{i+1}{i+1}')
        ax.legend(fontsize=7); ax.grid(True, which='both', ls=':', alpha=0.5)
        ax = axes[1, i]
        ax.semilogx(freq, np.angle(Yc_fitted[:, i, i], deg=True), 'r--', lw=2.5, label='VF Fit')
        ax.semilogx(freq, np.angle(Yc_calc[:, i, i], deg=True), 'b-', lw=1.2, label='VF Calc')
        ax.set_ylabel(f'∠Yc{i+1}{i+1} [°]'); ax.set_xlabel('Frequency [Hz]')
        ax.legend(fontsize=7); ax.grid(True, which='both', ls=':', alpha=0.5)
    fig.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_Yc_VF_fitting.png')
        fig.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")
    return fig


def plot_VF_H_mode_calc_vs_fitted(freq, calc_results, save_path=None):
    """绘制 VF 自研 H 模态计算值 vs VF 拟合值对比图"""
    fitting_result = calc_results.get('fitting_result')
    H_mode_calc = calc_results.get('H_mode')
    tau = calc_results.get('tau')
    if fitting_result is None or H_mode_calc is None: return None
    H_fits = getattr(fitting_result, 'H_modes_fits', None)
    if H_fits is None or len(H_fits) == 0: return None

    n_cond = H_mode_calc.shape[1]; K = len(freq)
    s = 1j * 2 * np.pi * freq
    H_mode_fitted = np.zeros((K, n_cond), dtype=complex)
    fitted_modes = set(); fit_info = []

    for fit in H_fits:
        if fit is None: continue
        m = fit.mode_index
        if m >= n_cond: continue
        poles = fit.poles; residues = fit.residues
        tau_m = fit.tau if hasattr(fit, 'tau') else (tau[m] if tau is not None and m < len(tau) else 0.0)
        d = getattr(fit, 'd', 0.0); e = getattr(fit, 'e', 0.0)
        for k_idx in range(K):
            val = complex(d) + complex(e) * s[k_idx] if d is not None else 0.0 + 0.0j
            for n_p, p in enumerate(poles):
                if np.isreal(p): val += residues[n_p].real / (s[k_idx] - p)
                else:
                    val += residues[n_p] / (s[k_idx] - p)
                    val += residues[n_p].conj() / (s[k_idx] - p.conj())
            val *= np.exp(-s[k_idx] * tau_m)
            H_mode_fitted[k_idx, m] = val
        fitted_modes.add(m)
        fit_info.append((m, len(poles), fit.rmse * 100, tau_m * 1e6))

    mode_colors = plt.cm.tab10(np.linspace(0, 1, max(n_cond, 1)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('VF H (Modal): Calculated vs Vector Fitting Result', fontsize=13)
    ax = axes[0]
    for m in range(n_cond):
        if m in fitted_modes:
            ax.semilogx(freq, np.abs(H_mode_fitted[:, m]), '--', color=mode_colors[m], lw=2.5, label=f'Mode {m+1} Fit')
    for m in range(n_cond):
        ax.semilogx(freq, np.abs(H_mode_calc[:, m]), color=mode_colors[m], lw=1.2, label=f'Mode {m+1} Calc')
    ax.set_ylabel('|H|'); ax.set_title('Magnitude'); ax.legend(fontsize=7, ncol=3)
    ax.grid(True, which='both', ls=':', alpha=0.5); ax.set_ylim([0, 1.1])
    ax = axes[1]
    for m in range(n_cond):
        if m in fitted_modes:
            ax.semilogx(freq, np.angle(H_mode_fitted[:, m], deg=True), '--', color=mode_colors[m], lw=2.5, label=f'Mode {m+1} Fit')
    for m in range(n_cond):
        ax.semilogx(freq, np.angle(H_mode_calc[:, m], deg=True), color=mode_colors[m], lw=1.2, label=f'Mode {m+1} Calc')
    ax.set_ylabel('∠H [°]'); ax.set_xlabel('Frequency [Hz]')
    ax.legend(fontsize=7, ncol=3); ax.grid(True, which='both', ls=':', alpha=0.5)
    if fit_info:
        info_text = '  '.join([f'Mode {m+1}: {np_}p, RMSE={rmse:.3f}%, τ={tau_us:.2f}μs' for m, np_, rmse, tau_us in fit_info])
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=8, style='italic', color='gray')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    if save_path:
        p = save_path.replace('.png', '_H_VF_fitting.png')
        fig.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")
    return fig


def plot_cable_cross_section(cable_model, config, save_path=None):
    """绘制三芯管型电缆横截面示意图"""
    import matplotlib.patches as patches
    pt_cable = create_pipe_type_cable(cable_model, config)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8)); ax.set_aspect('equal')
    r_total = pt_cable.jacket_radius; margin = r_total * 0.25
    ax.set_xlim(-r_total - margin, r_total + margin)
    ax.set_ylim(-r_total - margin, r_total + margin)
    ax.add_patch(patches.Circle((0, 0), pt_cable.jacket_radius, facecolor='#2E4A3E', edgecolor='black', lw=1))
    ax.add_patch(patches.Circle((0, 0), pt_cable.pipe_outer_radius, facecolor='#6B7B8C', edgecolor='black', lw=0.8))
    ax.add_patch(patches.Circle((0, 0), pt_cable.pipe_inner_radius, facecolor='#D4C4A8', edgecolor='black', lw=0.5))
    phase_labels = ['A', 'B', 'C']; phase_colors = ['#E74C3C', '#27AE60', '#2980B9']
    for i, ic in enumerate(pt_cable.inner_conductors):
        cx = ic.distance_from_center * np.cos(ic.angular_position)
        cy = ic.distance_from_center * np.sin(ic.angular_position)
        ax.add_patch(patches.Circle((cx, cy), ic.outer_insulation_radius, facecolor='#E8E8E8', edgecolor='black', lw=0.3))
        ax.add_patch(patches.Circle((cx, cy), ic.sheath_outer_radius, facecolor='#4A4A4A', edgecolor='black', lw=0.3))
        ax.add_patch(patches.Circle((cx, cy), ic.insulation_radius, facecolor='#F5DEB3', edgecolor='black', lw=0.3))
        ax.add_patch(patches.Circle((cx, cy), ic.core_radius, facecolor=phase_colors[i], edgecolor='black', lw=0.3))
        ax.text(cx, cy, phase_labels[i], ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    legend_items = [('#E74C3C', 'Core (Cu)'), ('#F5DEB3', 'Insulation (XLPE)'), ('#4A4A4A', 'Sheath (Pb)'),
                    ('#D4C4A8', f'Pipe filler (er={config.pipe_inner_insulation_epsilon_r})'),
                    ('#6B7B8C', f'Steel pipe (ur={config.pipe_mu_r})'), ('#2E4A3E', 'Jacket')]
    for idx_l, (color, label) in enumerate(legend_items):
        ax.add_patch(patches.Rectangle((r_total * 0.55, r_total * 0.85 - idx_l * r_total * 0.15),
                                        r_total * 0.1, r_total * 0.1, facecolor=color, edgecolor='black', lw=0.3))
        ax.text(r_total * 0.7, r_total * 0.9 - idx_l * r_total * 0.15, label, fontsize=8, va='center')
    ax.set_title(f'Cable_14: Three-Core Pipe-Type Cable Cross Section\n'
                 f'(Rin={config.pipe_inner_radius*1000:.1f}mm, Rout={config.pipe_outer_radius*1000:.1f}mm, '
                 f'Jacket={config.jacket_radius*1000:.1f}mm)', fontsize=11, fontweight='bold')
    ax.axis('off')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"  横截面图已保存: {save_path}")
    return fig


# =============================================================================
# 第八部分 (续)：总控绘图函数
# =============================================================================
def plot_all_comparisons(freq, calc_results, pscad_results, names, save_path=None):
    """生成全部对比图"""
    figs = []
    Yc_calc = calc_results['Yc_phase']
    H_phase_calc = calc_results['H_phase']
    H_mode_calc = calc_results['H_mode']
    gamma_calc = calc_results['gamma_mode']
    Z_calc = calc_results.get('Z_calc')
    Y_calc = calc_results.get('Y_calc')

    # 0. Z/Y 矩阵对比
    Z_pscad = pscad_results.get('Z_pscad')
    Y_pscad = pscad_results.get('Y_pscad')
    if Z_pscad is not None and Y_pscad is not None and Z_calc is not None and Y_calc is not None:
        zy_figs = plot_ZY_comparison(freq, Z_calc, Z_pscad, Y_calc, Y_pscad, names, save_path=save_path)
        figs.extend(zy_figs)
    else:
        if Z_pscad is not None and Z_calc is not None:
            fm, fp = plot_full_matrix_comparison(freq, Z_calc, Z_pscad, names, matrix_label="Z", unit_mag="mΩ/m", mag_scale=1000.0, use_loglog_mag=True, save_path=save_path)
            figs.extend([fm, fp])
        if Y_pscad is not None and Y_calc is not None:
            fm, fp = plot_full_matrix_comparison(freq, Y_calc, Y_pscad, names, matrix_label="Y", unit_mag="μS/m", mag_scale=1e6, use_loglog_mag=True, save_path=save_path)
            figs.extend([fm, fp])

    # 1. Yc 完整矩阵对比 + 误差
    if pscad_results.get('Yc_pscad') is not None:
        Yc_ref = pscad_results['Yc_pscad']
        fm, fp = plot_full_matrix_comparison(freq, Yc_calc, Yc_ref, names, matrix_label="Yc", unit_mag="mS", mag_scale=1e3, use_loglog_mag=True, save_path=save_path)
        figs.extend([fm, fp])
        if pscad_results.get('H_phase_pscad') is not None:
            fig_err = plot_error_vs_freq(freq, Yc_calc, Yc_ref, H_phase_calc, pscad_results['H_phase_pscad'], names, save_path=save_path)
            figs.append(fig_err)

    # 2. Yc 自导纳详细对比
    if pscad_results.get('Yc_pscad') is not None:
        fig_yc_self = plot_Yc_self_comparison(freq, Yc_calc, pscad_results['Yc_pscad'], names, save_path=save_path)
        figs.append(fig_yc_self)

    # 3. H 相域完整矩阵对比
    if pscad_results.get('H_phase_pscad') is not None:
        fm, fp = plot_full_matrix_comparison(freq, H_phase_calc, pscad_results['H_phase_pscad'], names, matrix_label="H_phase", unit_mag="—", mag_scale=1.0, use_loglog_mag=False, save_path=save_path)
        figs.extend([fm, fp])

    # 4. H 模态域对比
    if pscad_results.get('H_mode_pscad') is not None:
        H_mode_ref = pscad_results['H_mode_pscad']
        if H_mode_ref.ndim == 3:
            n_mr = H_mode_ref.shape[1]
            H_mode_ref_vec = np.zeros((len(freq), n_mr), dtype=complex)
            for m in range(n_mr): H_mode_ref_vec[:, m] = H_mode_ref[:, m, m]
            H_mode_ref = H_mode_ref_vec
        n_modes = max(H_mode_ref.shape[1], H_mode_calc.shape[1])
        mode_names = [f'Mode {i+1}' for i in range(n_modes)]
        fig_hm = plot_modal_H_comparison(freq, H_mode_calc, H_mode_ref, mode_names, save_path=save_path)
        figs.append(fig_hm)

    # 5. Lambda 对比
    if pscad_results.get('Lambda_pscad') is not None:
        Lambda_ref = pscad_results['Lambda_pscad']
        if Lambda_ref.ndim == 3:
            n_mr = Lambda_ref.shape[1]
            lambda_ref = np.zeros((len(freq), n_mr), dtype=complex)
            for m in range(n_mr): lambda_ref[:, m] = Lambda_ref[:, m, m]
        else:
            lambda_ref = Lambda_ref.copy()
        n_modes = min(lambda_ref.shape[1], gamma_calc.shape[1])
        mode_names = [f'Mode {i+1}' for i in range(n_modes)]
        fig_lam = plot_lambda_comparison(freq, gamma_calc[:, :n_modes], lambda_ref[:, :n_modes], mode_names, save_path=save_path)
        figs.append(fig_lam)

    # 6. Ti 变换矩阵对比
    Ti_pscad = pscad_results.get('Ti_pscad')
    Ti_calc_full = calc_results.get('T_I')
    T_ref = calc_results.get('T_ref')
    if Ti_pscad is not None:
        if Ti_calc_full is not None:
            Ti_for_plot = Ti_calc_full
        elif T_ref is not None:
            Ti_for_plot = np.tile(T_ref[np.newaxis, :, :], (len(freq), 1, 1))
            print("  ⚠ Ti 计算值仅有参考频率 T_ref，扩展为频率无关矩阵进行对比")
        else:
            Ti_for_plot = None

        if Ti_for_plot is not None:
            n_ti = min(Ti_for_plot.shape[1], Ti_pscad.shape[1])
            ti_names = names[:n_ti] if len(names) >= n_ti else [f'C{i+1}' for i in range(n_ti)]
            fm, fp = plot_Ti_matrix_elements(freq, Ti_for_plot[:, :n_ti, :n_ti], Ti_pscad[:, :n_ti, :n_ti], ti_names, save_path=save_path)
            figs.extend([fm, fp])
            ti_cmp = compute_Ti_comparison(Ti_for_plot, Ti_pscad, freq, ti_names, verbose=False)
            fig_tc = plot_Ti_column_correlation(freq, ti_cmp['mode_correlations'], ti_cmp['column_errors'], save_path=save_path)
            figs.append(fig_tc)

    # 7. PSCAD 内部拟合对比 (Yc calc vs fitted)
    if pscad_results.get('Yc_has_fitted') and pscad_results.get('Yc_fitted') is not None:
        Yc_pscad_calc = pscad_results['Yc_pscad']; Yc_pscad_fit = pscad_results['Yc_fitted']
        n = Yc_pscad_calc.shape[1]; n_show = min(n, 7)
        fig_fit, axes_fit = plt.subplots(2, n_show, figsize=(4.5 * n_show, 7), sharex=True)
        if n_show == 1: axes_fit = axes_fit.reshape(-1, 1)
        fig_fit.suptitle('PSCAD Yc: Calculated vs Fitted (PSCAD internal)', fontsize=13)
        for i in range(n_show):
            axes_fit[0, i].semilogx(freq, np.abs(Yc_pscad_fit[:, i, i]) * 1e3, 'r--', lw=2.5, label='PSCAD Fit')
            axes_fit[0, i].semilogx(freq, np.abs(Yc_pscad_calc[:, i, i]) * 1e3, 'b-', lw=1.2, label='PSCAD Calc')
            axes_fit[0, i].set_ylabel(f'|Yc{i+1}{i+1}| [mS]'); axes_fit[0, i].set_title(f'Yc{i+1}{i+1}')
            axes_fit[0, i].legend(fontsize=7); axes_fit[0, i].grid(True, which='both', ls=':', alpha=0.5)
            axes_fit[1, i].semilogx(freq, np.angle(Yc_pscad_fit[:, i, i], deg=True), 'r--', lw=2.5, label='PSCAD Fit')
            axes_fit[1, i].semilogx(freq, np.angle(Yc_pscad_calc[:, i, i], deg=True), 'b-', lw=1.2, label='PSCAD Calc')
            axes_fit[1, i].set_ylabel(f'∠Yc{i+1}{i+1} [°]'); axes_fit[1, i].set_xlabel('Frequency [Hz]')
            axes_fit[1, i].legend(fontsize=7); axes_fit[1, i].grid(True, which='both', ls=':', alpha=0.5)
        fig_fit.tight_layout(); figs.append(fig_fit)
        if save_path:
            p = save_path.replace('.png', '_Yc_PSCAD_fitting.png')
            fig_fit.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")

    # 8. PSCAD H mode calc vs fitted
    if pscad_results.get('H_mode_has_fitted') and pscad_results.get('H_mode_mag_fitted') is not None:
        h_mag = pscad_results['H_mode_pscad_mag']; h_phase = pscad_results['H_mode_pscad_phase']
        h_mag_fit = pscad_results['H_mode_mag_fitted']; h_phase_fit = pscad_results['H_mode_phase_fitted']
        n_modes = h_mag.shape[1] if h_mag.ndim == 2 else 1
        mode_colors = plt.cm.tab10(np.linspace(0, 1, max(n_modes, 1)))
        fig_hfit, axes_hfit = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig_hfit.suptitle('PSCAD H (Modal): Calculated vs Fitted', fontsize=13)
        if h_mag_fit is not None:
            for m in range(n_modes):
                if h_mag_fit.ndim == 2:
                    axes_hfit[0].semilogx(freq, h_mag_fit[:, m], '--', color=mode_colors[m], lw=2.5, label=f'Mode {m+1} Fit')
                else:
                    axes_hfit[0].semilogx(freq, h_mag_fit, '--', color=mode_colors[0], lw=2.5, label='Fit')
        for m in range(n_modes):
            if h_mag.ndim == 2:
                axes_hfit[0].semilogx(freq, h_mag[:, m], color=mode_colors[m], lw=1.2, label=f'Mode {m+1} Calc')
            else:
                axes_hfit[0].semilogx(freq, h_mag, color=mode_colors[0], lw=1.2, label='Calc')
        axes_hfit[0].set_ylabel('|H|'); axes_hfit[0].set_title('Magnitude')
        axes_hfit[0].legend(fontsize=7, ncol=3); axes_hfit[0].grid(True, which='both', ls=':', alpha=0.5)
        axes_hfit[0].set_ylim([0, 1.1])
        if h_phase_fit is not None:
            for m in range(n_modes):
                if h_phase_fit.ndim == 2:
                    axes_hfit[1].semilogx(freq, h_phase_fit[:, m], '--', color=mode_colors[m], lw=2.5, label=f'Mode {m+1} Fit')
                else:
                    axes_hfit[1].semilogx(freq, h_phase_fit, '--', color=mode_colors[0], lw=2.5, label='Fit')
        for m in range(n_modes):
            if h_phase.ndim == 2:
                axes_hfit[1].semilogx(freq, h_phase[:, m], color=mode_colors[m], lw=1.2, label=f'Mode {m+1} Calc')
            else:
                axes_hfit[1].semilogx(freq, h_phase, color=mode_colors[0], lw=1.2)
        axes_hfit[1].set_ylabel('∠H [°]'); axes_hfit[1].set_xlabel('Frequency [Hz]')
        axes_hfit[1].legend(fontsize=7, ncol=3); axes_hfit[1].grid(True, which='both', ls=':', alpha=0.5)
        fig_hfit.tight_layout(); figs.append(fig_hfit)
        if save_path:
            p = save_path.replace('.png', '_H_PSCAD_fitting.png')
            fig_hfit.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")

    # 9. VF 拟合质量图
    fitting_result = calc_results.get('fitting_result')
    ulm_params_obj = calc_results.get('ulm_params')
    if fitting_result is not None and ulm_params_obj is not None:
        n_cond = Yc_calc.shape[1]
        fig_vf, axes_vf = plt.subplots(2, 2, figsize=(14, 10))
        fig_vf.suptitle('Vector Fitting Quality: tr(Yc) + H Modes + Poles', fontsize=13, y=1.0)
        ax = axes_vf[0, 0]
        ax.loglog(freq, np.abs(ulm_params_obj.Yc_trace), 'b-', lw=2, label='Calculated')
        if hasattr(fitting_result, 'poles_Yc') and fitting_result.poles_Yc is not None:
            s = 1j * 2 * np.pi * freq; Yc_fit_trace = np.zeros(len(freq), dtype=complex)
            k0 = fitting_result.k0; k_res = fitting_result.k_residues; poles = fitting_result.poles_Yc
            for k in range(len(freq)):
                Yk = k0.astype(complex).copy()
                for n_p, p in enumerate(poles): Yk += k_res[n_p] / (s[k] - p)
                Yc_fit_trace[k] = np.trace(Yk)
            ax.loglog(freq, np.abs(Yc_fit_trace), 'r--', lw=1.5, label='VF Fitted')
            ax.set_title(f'tr(Yc) Fitting (RMSE = {fitting_result.Yc_trace_rmse * 100:.3f}%)')
        else:
            ax.set_title('tr(Yc)')
        ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('|tr(Yc)| [S]')
        ax.legend(fontsize=8); ax.grid(True, which='both', ls=':', alpha=0.5)
        ax = axes_vf[0, 1]
        for i in range(n_cond): ax.semilogx(freq, np.abs(calc_results['H_mode'][:, i]), lw=1.5, label=f'Mode {i+1}')
        ax.axhline(1.0, color='k', ls='--', alpha=0.3); ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('|H|')
        ax.set_title('Modal Propagation Functions'); ax.legend(fontsize=7); ax.grid(True, which='both', ls=':', alpha=0.5); ax.set_ylim([0, 1.1])
        ax = axes_vf[1, 0]
        if hasattr(fitting_result, 'poles_Yc') and fitting_result.poles_Yc is not None:
            pYc = fitting_result.poles_Yc
            ax.scatter(np.real(pYc), np.imag(pYc) / (2 * np.pi), c='steelblue', s=60, marker='x', label='tr(Yc)', zorder=3)
        H_fits = getattr(fitting_result, 'H_modes_fits', []) or []
        colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(H_fits))))
        for i, fit in enumerate(H_fits):
            if fit is None or fit.poles is None or len(fit.poles) == 0: continue
            ax.scatter(np.real(fit.poles), np.imag(fit.poles) / (2 * np.pi), c=[colors[i]], s=40, marker='o', alpha=0.7, label=f'H Mode {i+1}')
        ax.set_xlabel('Real Part [Np/s]'); ax.set_ylabel('Imaginary Part [Hz]'); ax.set_title('Pole Distribution')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, ls=':', alpha=0.5)
        ax = axes_vf[1, 1]
        if fitting_result.H_modes_fits:
            mi = []; rs = []; npl = []
            for fit in fitting_result.H_modes_fits:
                if fit is not None: mi.append(fit.mode_index + 1); rs.append(fit.rmse * 100); npl.append(len(fit.poles))
            if mi:
                bars = ax.bar(mi, rs, color='steelblue', alpha=0.7, edgecolor='navy')
                for bar, np_ in zip(bars, npl): ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{np_}p', ha='center', va='bottom', fontsize=8)
                ax.axhline(fitting_result.Yc_trace_rmse * 100, color='r', ls='--', alpha=0.7, label=f'tr(Yc) RMSE')
                ax.set_xlabel('Mode Index'); ax.set_ylabel('RMSE [%]'); ax.set_title('H Mode Fitting RMSE')
                ax.legend(fontsize=7); ax.grid(True, axis='y', ls=':', alpha=0.5)
        fig_vf.tight_layout(); figs.append(fig_vf)
        if save_path:
            p = save_path.replace('.png', '_VF_fitting_quality.png')
            fig_vf.savefig(p, dpi=150, bbox_inches='tight'); print(f"  已保存: {p}")

    # 10. VF Yc calc vs fitted
    fig_vf_yc = plot_VF_Yc_calc_vs_fitted(freq, calc_results, names, save_path=save_path)
    if fig_vf_yc is not None: figs.append(fig_vf_yc)

    # 11. VF H calc vs fitted
    fig_vf_h = plot_VF_H_mode_calc_vs_fitted(freq, calc_results, save_path=save_path)
    if fig_vf_h is not None: figs.append(fig_vf_h)

    return figs


# =============================================================================
# 第九部分：典型频率点数值打印
# =============================================================================
def print_ZY_spot_comparison(freq, Z_calc, Z_ref, Y_calc, Y_ref, names, spot_freqs=None):
    """在几个典型频率点打印 Z 和 Y 的详细对比值"""
    if spot_freqs is None: spot_freqs = [0.5, 50, 500, 5000, 50000, 100000]
    n = Z_calc.shape[1]
    print("\n" + "=" * 90)
    print(" 典型频率点 Z / Y 详细对比")
    print("=" * 90)
    for f_target in spot_freqs:
        if f_target < freq[0] or f_target > freq[-1]: continue
        idx = np.argmin(np.abs(freq - f_target)); f_actual = freq[idx]
        print(f"\n  ──── @ {f_actual:.2f} Hz ────")
        print(f"  {'元素':>10s} | {'|Calc|':>12s} | {'|PSCAD|':>12s} | {'幅值误差%':>10s} | {'∠Calc°':>10s} | {'∠PSCAD°':>10s} | {'相角误差°':>10s}")
        print("  " + "-" * 88)
        for i in range(n):
            for j in range(n):
                zc = Z_calc[idx, i, j]; zr = Z_ref[idx, i, j]
                mc, mr = np.abs(zc) * 1000, np.abs(zr) * 1000; pc, pr = np.angle(zc, deg=True), np.angle(zr, deg=True)
                me = abs(mc - mr) / max(mr, 1e-30) * 100; pe = min(abs(pc - pr), 360 - abs(pc - pr))
                print(f"  Z{i+1}{j+1:8d} | {mc:12.6f} | {mr:12.6f} | {me:10.4f} | {pc:10.4f} | {pr:10.4f} | {pe:10.4f}")
        print()
        for i in range(n):
            for j in range(n):
                yc = Y_calc[idx, i, j]; yr = Y_ref[idx, i, j]
                mc, mr = np.abs(yc) * 1e6, np.abs(yr) * 1e6; pc, pr = np.angle(yc, deg=True), np.angle(yr, deg=True)
                me = abs(mc - mr) / max(mr, 1e-30) * 100; pe = min(abs(pc - pr), 360 - abs(pc - pr))
                print(f"  Y{i+1}{j+1:8d} | {mc:12.6f} | {mr:12.6f} | {me:10.4f} | {pc:10.4f} | {pr:10.4f} | {pe:10.4f}")


def print_spot_comparison(freq, Yc_calc, Yc_ref, H_calc, H_ref, names, spot_freqs=None):
    """在典型频率点打印 Yc 和 H 的详细对比值"""
    if spot_freqs is None: spot_freqs = [0.5, 50, 500, 5000, 50000, 100000]
    n = Yc_calc.shape[1]
    print("\n" + "=" * 90)
    print(" 典型频率点 Yc / H 详细对比")
    print("=" * 90)
    for f_target in spot_freqs:
        if f_target < freq[0] or f_target > freq[-1]: continue
        idx = np.argmin(np.abs(freq - f_target)); f_actual = freq[idx]
        print(f"\n  ──── @ {f_actual:.2f} Hz ────")
        print(f"  {'元素':>14s} | {'|Calc|':>14s} | {'|PSCAD|':>14s} | {'幅值误差%':>10s} | {'∠Calc°':>10s} | {'∠PSCAD°':>10s} | {'相角误差°':>10s}")
        print("  " + "-" * 96)
        for i in range(n):
            for j in range(n):
                yc_c = Yc_calc[idx, i, j]; yc_r = Yc_ref[idx, i, j]
                mc, mr = np.abs(yc_c) * 1e3, np.abs(yc_r) * 1e3; pc, pr = np.angle(yc_c, deg=True), np.angle(yc_r, deg=True)
                me = abs(mc - mr) / max(mr, 1e-30) * 100; pe = min(abs(pc - pr), 360 - abs(pc - pr))
                print(f"  Yc{i+1}{j+1:>9d} | {mc:>14.6f} | {mr:>14.6f} | {me:>10.4f} | {pc:>10.4f} | {pr:>10.4f} | {pe:>10.4f}")
        print()
        for i in range(n):
            for j in range(n):
                h_c = H_calc[idx, i, j]; h_r = H_ref[idx, i, j]
                mc, mr = np.abs(h_c), np.abs(h_r); pc, pr = np.angle(h_c, deg=True), np.angle(h_r, deg=True)
                me = abs(mc - mr) / max(mr, 1e-30) * 100; pe = min(abs(pc - pr), 360 - abs(pc - pr))
                print(f"  Hp{i+1}{j+1:>9d} | {mc:>14.6f} | {mr:>14.6f} | {me:>10.4f} | {pc:>10.4f} | {pr:>10.4f} | {pe:>10.4f}")


# =============================================================================
# 第十部分：fitULM 文件导出
# =============================================================================
def export_fitulm_file(modules, fitting_result, fitulm_filename="cable14.fitULM", fitulm_precision=16, verbose=True):
    """导出 fitULM 格式文件"""
    vf = modules.get('vf')
    if vf is None or not hasattr(vf, 'write_fitULM'):
        print("\n✗ 无法导出 fitULM: 模块不支持 write_fitULM 函数"); return None
    if fitting_result is None:
        print("\n✗ 无法导出 fitULM: fitting_result 为 None"); return None
    if os.path.isabs(fitulm_filename):
        fitulm_path = fitulm_filename
    else:
        test_dir = os.path.join(str(PROJECT_DIR), 'TEST'); os.makedirs(test_dir, exist_ok=True)
        fitulm_path = os.path.join(test_dir, fitulm_filename)
    if verbose:
        print("\n" + "=" * 60); print(" 导出 fitULM 文件"); print("=" * 60)
        print(f"  输出路径: {fitulm_path}"); print(f"  精度: {fitulm_precision} 位")
    try:
        vf.write_fitULM(result=fitting_result, filepath=fitulm_path, precision=fitulm_precision, verbose=verbose)
        if hasattr(vf, 'verify_fitULM_file'): vf.verify_fitULM_file(fitulm_path, verbose=verbose)
        if verbose: print(f"\n  ✓ fitULM 文件已成功导出: {fitulm_path}")
        return fitulm_path
    except Exception as e:
        print(f"\n✗ fitULM 导出失败: {e}"); import traceback; traceback.print_exc(); return None


# =============================================================================
# 第十一部分：主函数
# =============================================================================
def main(pscad_path: str = None,
         line_length: float = 5000.0,
         verbose: bool = True,
         save_plots: bool = True,
         export_fitulm: bool = True,
         fitulm_filename: str = "cable14.fitULM",
         fitulm_precision: int = 16):
    """
    主流程：计算电缆 Z/Y → VF 计算 Yc/H → 读取 PSCAD → 对齐 → 误差 → 全套对比图

    Parameters
    ----------
    pscad_path : str   PSCAD 输出文件路径前缀
    line_length : float  线路长度 [m]
    """
    print("\n" + "=" * 70)
    print(" 三芯管型电缆 (Cable_14) ULM 完整计算 + PSCAD 对比")
    print(" Pipe-Type Cable, 7x7 矩阵")
    print("=" * 70)
    print(f"  线路长度: {line_length/1000:.1f} km")

    # ---- 导入模块 ----
    modules = import_all_modules()
    if modules['cable_model'] is None:
        print("✗ 缺少 cable_model_0110 模块，中止"); return None
    if modules.get('vf') is None:
        print("✗ 缺少 vector_fitting 模块，中止"); return None
    if modules['PSCADFileReader'] is None:
        print("✗ 缺少 pscad_reader，无法进行 PSCAD 对比"); return None

    cable_model = modules['cable_model']
    config = CableLineConfig()
    config.line_length = line_length

    print(f"\n  电缆配置: {config.line_name}")
    print(f"  导体总数: {config.n_total_conductors} ({config.n_inner_cables}x(Core+Sheath) + Pipe)")

    # ---- 确定 PSCAD 路径 ----
    if pscad_path is None:
        candidates = ['./Cable_14/Cable_14', './Cable_14', './Cable_1/Cable_1', './Cable_1']
        for c in candidates:
            test = c.rstrip('/') + '_zm.out'
            if os.path.exists(test):
                pscad_path = c; break
            if os.path.isdir(c):
                for f in os.listdir(c):
                    if f.endswith('_zm.out'):
                        pscad_path = c; break
            if pscad_path: break
        if pscad_path is None:
            print("✗ 未找到 PSCAD 输出文件，请指定路径")
            print("  用法: python ulm_three_core_cable_v2.py <PSCAD文件前缀> [线路长度m] [fitULM文件名]")
            return None

    print(f"  PSCAD 数据路径: {pscad_path}")

    # ---- 创建输出目录 ----
    test_dir = os.path.join(str(PROJECT_DIR), 'TEST')
    os.makedirs(test_dir, exist_ok=True)

    # ---- 步骤 0: 绘制电缆横截面 ----
    if save_plots:
        xsection_path = os.path.join(test_dir, 'cable14_cross_section.png')
        plot_cable_cross_section(cable_model, config, save_path=xsection_path)

    # ---- 步骤 1: 读取 PSCAD 所有数据 ----
    pscad_results = load_Yc_H_pscad(modules, pscad_path, verbose)
    freq_pscad = pscad_results['freq_pscad']

    # ---- 步骤 2: 用 PSCAD 频率向量计算 Z/Y ----
    freq = freq_pscad   # 直接用 PSCAD 频率，保证频点完全一致
    Z_matrix, Y_matrix, P_matrix, pt_cable = compute_ZY_self(modules, config, freq, verbose)

    # ---- 步骤 3: 通过 VF 计算 Yc 和 H ----
    calc_results = compute_Yc_H_via_VF(
        modules, freq, Z_matrix, Y_matrix, config,
        use_freq_dependent='auto', verbose=verbose)

    calc_results['Z_calc'] = Z_matrix
    calc_results['Y_calc'] = Y_matrix
    calc_results['P_matrix'] = P_matrix

    names = COND_LABELS
    Yc_calc = calc_results['Yc_phase']
    H_phase_calc = calc_results['H_phase']

    # ---- 步骤 4: 维度对齐 ----
    Yc_pscad = pscad_results.get('Yc_pscad')
    H_phase_pscad = pscad_results.get('H_phase_pscad')

    if Yc_pscad is not None and Yc_calc.shape[1] != Yc_pscad.shape[1]:
        n_min = min(Yc_calc.shape[1], Yc_pscad.shape[1])
        print(f"\n  ⚠ Yc 维度不匹配: 自研 {Yc_calc.shape[1]}, PSCAD {Yc_pscad.shape[1]}")
        Yc_calc_cmp = Yc_calc[:, :n_min, :n_min]
        Yc_pscad_cmp = Yc_pscad[:, :n_min, :n_min]
        names_cmp = names[:n_min]
        print(f"  → 截取为 {n_min}×{n_min} 进行对比")
    else:
        Yc_calc_cmp = Yc_calc; Yc_pscad_cmp = Yc_pscad; names_cmp = names

    if H_phase_pscad is not None and H_phase_calc.shape[1] != H_phase_pscad.shape[1]:
        n_min = min(H_phase_calc.shape[1], H_phase_pscad.shape[1])
        H_phase_calc_cmp = H_phase_calc[:, :n_min, :n_min]
        H_phase_pscad_cmp = H_phase_pscad[:, :n_min, :n_min]
    else:
        H_phase_calc_cmp = H_phase_calc; H_phase_pscad_cmp = H_phase_pscad

    Z_pscad = pscad_results.get('Z_pscad'); Y_pscad = pscad_results.get('Y_pscad')
    Z_calc_cmp = Z_matrix; Z_pscad_cmp = Z_pscad
    Y_calc_cmp = Y_matrix; Y_pscad_cmp = Y_pscad

    if Z_pscad is not None and Z_matrix.shape[1] != Z_pscad.shape[1]:
        n_min = min(Z_matrix.shape[1], Z_pscad.shape[1])
        Z_calc_cmp = Z_matrix[:, :n_min, :n_min]; Z_pscad_cmp = Z_pscad[:, :n_min, :n_min]
    if Y_pscad is not None and Y_matrix.shape[1] != Y_pscad.shape[1]:
        n_min = min(Y_matrix.shape[1], Y_pscad.shape[1])
        Y_calc_cmp = Y_matrix[:, :n_min, :n_min]; Y_pscad_cmp = Y_pscad[:, :n_min, :n_min]

    # ---- 步骤 5: 误差统计 ----
    print("\n" + "=" * 65)
    print(" 误差统计")
    print("=" * 65)

    Yc_errors = H_phase_errors = H_mode_errors = Z_errors = Y_errors = Ti_errors = None

    if Z_pscad_cmp is not None:
        Z_errors = compute_matrix_error(Z_calc_cmp, Z_pscad_cmp, names_cmp, label='Z', verbose=verbose)
    if Y_pscad_cmp is not None:
        Y_errors = compute_matrix_error(Y_calc_cmp, Y_pscad_cmp, names_cmp, label='Y', verbose=verbose)
    if Yc_pscad_cmp is not None:
        Yc_errors = compute_matrix_error(Yc_calc_cmp, Yc_pscad_cmp, names_cmp, label='Yc', verbose=verbose)
    if H_phase_pscad_cmp is not None:
        H_phase_errors = compute_matrix_error(H_phase_calc_cmp, H_phase_pscad_cmp, names_cmp, label='H_phase', verbose=verbose)

    # H 模态域误差
    H_mode_pscad = pscad_results.get('H_mode_pscad')
    if H_mode_pscad is not None:
        H_mode_calc = calc_results['H_mode']
        if H_mode_pscad.ndim == 3:
            n_mr = H_mode_pscad.shape[1]
            H_mode_pscad_vec = np.zeros((len(freq), n_mr), dtype=complex)
            for m in range(n_mr): H_mode_pscad_vec[:, m] = H_mode_pscad[:, m, m]
        else:
            H_mode_pscad_vec = H_mode_pscad
        n_modes = min(H_mode_calc.shape[1], H_mode_pscad_vec.shape[1])
        mode_names = [f'Mode {i+1}' for i in range(n_modes)]
        H_mode_errors = compute_vector_error(
            H_mode_calc[:, :n_modes], H_mode_pscad_vec[:, :n_modes],
            mode_names, label='H_mode', verbose=verbose)

    # Ti 变换矩阵误差
    Ti_pscad = pscad_results.get('Ti_pscad')
    Ti_calc_full = calc_results.get('T_I')
    T_ref_mat = calc_results.get('T_ref')
    if Ti_pscad is not None:
        if Ti_calc_full is not None: Ti_for_cmp = Ti_calc_full
        elif T_ref_mat is not None: Ti_for_cmp = np.tile(T_ref_mat[np.newaxis, :, :], (len(freq), 1, 1))
        else: Ti_for_cmp = None
        if Ti_for_cmp is not None:
            ti_names = names_cmp if len(names_cmp) >= min(Ti_for_cmp.shape[1], Ti_pscad.shape[1]) \
                else [f'C{i+1}' for i in range(min(Ti_for_cmp.shape[1], Ti_pscad.shape[1]))]
            Ti_errors = compute_Ti_comparison(Ti_for_cmp, Ti_pscad, freq, ti_names, verbose=verbose)

    # ---- 步骤 6: 典型频率点对比 ----
    if Z_pscad_cmp is not None and Y_pscad_cmp is not None:
        print_ZY_spot_comparison(freq, Z_calc_cmp, Z_pscad_cmp, Y_calc_cmp, Y_pscad_cmp, names_cmp)
    if Yc_pscad_cmp is not None and H_phase_pscad_cmp is not None:
        print_spot_comparison(freq, Yc_calc_cmp, Yc_pscad_cmp, H_phase_calc_cmp, H_phase_pscad_cmp, names_cmp)

    # ---- 步骤 7: 绘图 ----
    if save_plots:
        print("\n" + "=" * 65)
        print(" 生成对比图表")
        print("=" * 65)

        pscad_for_plot = dict(pscad_results)
        pscad_for_plot['Yc_pscad'] = Yc_pscad_cmp
        pscad_for_plot['H_phase_pscad'] = H_phase_pscad_cmp
        pscad_for_plot['Z_pscad'] = Z_pscad_cmp
        pscad_for_plot['Y_pscad'] = Y_pscad_cmp

        calc_results_for_plot = dict(calc_results)
        calc_results_for_plot['Z_calc'] = Z_calc_cmp
        calc_results_for_plot['Y_calc'] = Y_calc_cmp

        save_base = os.path.join(test_dir, 'cable14_compare_results.png')
        figs = plot_all_comparisons(freq, calc_results_for_plot, pscad_for_plot, names_cmp, save_path=save_base)

    # ---- 步骤 8: 导出 fitULM ----
    fitulm_path = None
    if export_fitulm:
        fitting_result_obj = calc_results.get('fitting_result')
        fitulm_path = export_fitulm_file(
            modules=modules, fitting_result=fitting_result_obj,
            fitulm_filename=fitulm_filename, fitulm_precision=fitulm_precision, verbose=verbose)

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print(" 对比总结")
    print("=" * 70)

    fitting_result = calc_results.get('fitting_result')
    if fitting_result is not None:
        print(f"\n  [Vector Fitting 拟合质量]")
        print(f"  tr(Yc) 拟合 RMSE:    {fitting_result.Yc_trace_rmse * 100:.4f}%")
        print(f"  H 矩阵重构 RMSE:     {fitting_result.H_matrix_rmse * 100:.4f}%")
        print(f"  被动性:               {'✓ 满足' if fitting_result.is_passive else '✗ 违反'}")
        if fitting_result.H_reconstruction_metrics is not None:
            m = fitting_result.H_reconstruction_metrics
            print(f"  论文 Eq.(12)(13) RMSE: {m.method1_rmse * 100:.6f}%")
            print(f"  D 矩阵完备性误差:     {m.D_identity_error:.2e}")

    print(f"\n  [与 PSCAD 的对比误差]")
    if Z_errors: print(f"  Z 矩阵最大幅值误差:  {Z_errors['_global']['max_mag_err']:.4f} %")
    if Y_errors: print(f"  Y 矩阵最大幅值误差:  {Y_errors['_global']['max_mag_err']:.4f} %")
    if Yc_errors: print(f"  Yc 矩阵最大幅值误差: {Yc_errors['_global']['max_mag_err']:.4f} %")
    if H_phase_errors: print(f"  H(相域) 最大幅值误差: {H_phase_errors['_global']['max_mag_err']:.4f} %")
    if H_mode_errors: print(f"  H(模态) 最大幅值误差: {H_mode_errors['_global']['max_mag_err']:.4f} %")
    if Ti_errors:
        s = Ti_errors['summary']
        print(f"  Ti 变换矩阵: 平均相关系数 {s['mean_correlation']:.6f}, 最小 {s['min_correlation']:.6f}")

    # 综合判断
    max_errs = []
    if Z_errors: max_errs.append(('Z', Z_errors['_global']['max_mag_err']))
    if Y_errors: max_errs.append(('Y', Y_errors['_global']['max_mag_err']))
    if Yc_errors: max_errs.append(('Yc', Yc_errors['_global']['max_mag_err']))
    if H_phase_errors: max_errs.append(('H_phase', H_phase_errors['_global']['max_mag_err']))
    if max_errs:
        overall_max = max(e[1] for e in max_errs)
        worst_item = max(max_errs, key=lambda x: x[1])
        if overall_max < 1.0:
            print(f"\n  ✓ 所有计算结果与 PSCAD 吻合良好 (最大误差 < 1%)")
        elif overall_max < 5.0:
            print(f"\n  ⚠ 存在一定差异 (最大误差 {worst_item[1]:.2f}% 在 {worst_item[0]})")
        else:
            print(f"\n  ✗ 差异较大 (最大误差 {worst_item[1]:.2f}% 在 {worst_item[0]})，建议排查")

    print(f"\n  [输出文件]")
    if save_plots: print(f"  图表目录: {test_dir}")
    if fitulm_path: print(f"  fitULM 文件: {fitulm_path}")

    return {
        'freq': freq,
        'calc_results': calc_results,
        'pscad_results': pscad_results,
        'ulm_params': calc_results.get('ulm_params'),
        'fitting_result': fitting_result,
        'Z_errors': Z_errors, 'Y_errors': Y_errors,
        'Yc_errors': Yc_errors, 'H_phase_errors': H_phase_errors,
        'H_mode_errors': H_mode_errors, 'Ti_errors': Ti_errors,
        'names': names_cmp,
        'fitulm_path': fitulm_path,
    }


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    pscad_path = sys.argv[1] if len(sys.argv) > 1 else None
    line_length = float(sys.argv[2]) if len(sys.argv) > 2 else 5000.0
    fitulm_filename = sys.argv[3] if len(sys.argv) > 3 else "cable14.fitULM"

    results = main(pscad_path=pscad_path,
                   line_length=line_length,
                   verbose=True,
                   save_plots=True,
                   export_fitulm=True,
                   fitulm_filename=fitulm_filename,
                   fitulm_precision=16)
