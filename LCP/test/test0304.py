#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特性导纳 Yc / 传播函数 H / 阻抗导纳 Z·Y 对比脚本：Vector Fitting 计算 vs PSCAD
================================================================================

本脚本将以下两组结果进行对比：
  ① Vector Fitting 模块 (vector_fitting_v47_fixed.py)：
     基于 ulm_atp_zy_deri_semlyen.py 计算 Z/Y 后，
     调用 ulm_complete_fitting() 通过矩阵开方和 Newton-Raphson
     特征求解计算 Yc 和 H，并执行完整的 Vector Fitting 极点拟合
  ② PSCAD：通过 pscad_reader.py 从 PSCAD Line Constants 输出文件读取

VF 模块内部计算流程：
  1. Newton-Raphson 特征求解 → T_I 变换矩阵 (全频率)
  2. 矩阵开方: Yc = Z⁻¹ × √(ZY),  H = T × diag(e^{-γL}) × T⁻¹
  3. 延时提取 τᵢ (Gustavsen 2017)
  4. tr(Yc) → Vector Fitting 极点拟合
  5. 各模态 H → Vector Fitting 极点拟合
  6. 被动性检查

对比内容（全部为完整 n×n 矩阵对比，与 compare_zy_with_pscad_0211.py 风格一致）：
  - Z 阻抗矩阵全矩阵幅值与相角 (自研 vs PSCAD)
  - Y 导纳矩阵全矩阵幅值与相角 (自研 vs PSCAD)
  - Z/Y 全矩阵误差 vs 频率曲线
  - Z/Y 典型频率点全矩阵数值对比
  - Yc 完整 n×n 矩阵幅值与相角 (VF 计算值 vs PSCAD)
  - H (相域) 完整 n×n 矩阵幅值与相角
  - H (模态域) 幅值与相角
  - Yc/H 全矩阵误差 vs 频率曲线
  - Yc/H 典型频率点全矩阵数值对比
  - 传播常数 λ = γ² (实部 + 虚部, 与 PSCAD Lambda 直接对比)
  - Ti 变换矩阵 (列向量归一化对比 + 元素级幅值/相角对比)
  - VF 拟合质量 (tr(Yc) RMSE, H 重构 RMSE, 被动性)
  - 误差统计 (全频率范围, 全矩阵元素)

输出目录：代码当前目录下的 TEST 文件夹

使用方法:
    python compare_yc_h_with_pscad.py [PSCAD文件前缀路径] [线路长度m] [fitULM文件名]

示例:
    python compare_yc_h_with_pscad.py ./TLine_2/TLine_2 20000
    python compare_yc_h_with_pscad.py ./TLine_2/TLine_2 20000 my_model.fitULM
    python compare_yc_h_with_pscad.py ./Cable_1 100000

依赖模块:
    - ulm_atp_zy_deri_semlyen.py : Z/Y 计算 (Deri-Semlyen)
    - vector_fitting_v47_fixed.py : ULM 完整拟合 (Vector Fitting V4.7)
    - pscad_reader.py            : PSCAD 输出读取器

作者: Claude
日期: 2026-02-16
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
# 第一部分：导入模块
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

    # 2. Z/Y 计算模块 (Deri-Semlyen)
    try:
        import ulm_atp_zy_deri_semlyen as zy
        modules['zy'] = zy
        print("✓ ulm_atp_zy_deri_semlyen 导入成功 (Deri-Semlyen 复数深度法)")
    except ImportError as e:
        print(f"✗ ulm_atp_zy_deri_semlyen 导入失败: {e}")
        modules['zy'] = None

    # 3. Vector Fitting / ULM 模块
    try:
        import vector_fitting_v411_independent as vf
        modules['vf'] = vf
        print(f"✓ vector_fitting_v47_fixed V{vf.__version__} 导入成功")
        if hasattr(vf, 'ulm_complete_fitting'):
            print("  ✓ ulm_complete_fitting 可用")
        if hasattr(vf, 'write_fitULM'):
            print("  ✓ fitULM 导出功能可用")
    except ImportError as e:
        print(f"✗ vector_fitting_v47_fixed 导入失败: {e}")
        modules['vf'] = None

    return modules


# =============================================================================
# 第二部分：线路参数配置
# =============================================================================
@dataclass
class PSCADLineConfig:
    """PSCAD 线路参数配置"""
    line_name: str = "TLine_2"
    n_conductors: int = 4
    n_phases: int = 2
    n_ground_wires: int = 2

    # 土壤参数
    ground_resistivity: float = 1000.0   # Ω·m
    ground_permeability: float = 1.0
    ground_permittivity: float = 1.0     # 相对介电常数

    # 相导线参数 (4分裂)
    phase_radius: float = 0.03           # 子导线半径 m
    phase_dc_resistance: float = 0.05741 # Ω/km
    phase_mu_r: float = 1.0
    phase_bundle_n: int = 4              # 分裂数
    phase_bundle_spacing: float = 0.5    # 分裂间距 m

    # 相导线位置 (水平位置, 弧垂修正后平均高度)
    phase_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-11.777, 48.0 - 9.2 * 2 / 3),
        (11.777,  48.0 - 9.2 * 2 / 3),
    ])
    phase_sag: float = 9.2

    # 地线参数
    gw_radius: float = 0.00875
    gw_dc_resistance: float = 0.7098     # Ω/km
    gw_mu_r: float = 1.0

    # 地线位置
    gw_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-19.25, 63.0 - 6.1 * 2 / 3),
        (19.25,  63.0 - 6.1 * 2 / 3),
    ])
    gw_sag: float = 6.1

    # Kron 缩减控制
    kron_reduction: bool = False    # False = 保留全部导体 (含地线), True = 消去地线

    def get_average_height(self, tower_height: float, sag: float) -> float:
        return tower_height - sag * 2 / 3


# =============================================================================
# 第三部分：用自研代码计算 Z/Y
# =============================================================================
def compute_ZY_self(modules: Dict,
                    pscad_config: PSCADLineConfig,
                    freq: np.ndarray,
                    verbose: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    使用 ulm_atp_zy_deri_semlyen 计算阻抗/导纳矩阵

    Returns
    -------
    Z_matrix : ndarray [n_freq, n_cond, n_cond]
    Y_matrix : ndarray [n_freq, n_cond, n_cond]
    line     : MultiConductorLine 对象
    """
    zy = modules['zy']

    if verbose:
        print("\n" + "=" * 65)
        print(" 使用自研代码计算 Z/Y (Deri-Semlyen 复数深度法)")
        if not pscad_config.kron_reduction:
            print(" ★ 不进行地线 Kron 缩减 (保留全部导体)")
        else:
            print(" ★ 进行地线 Kron 缩减")
        print("=" * 65)

    # ---------- 创建导线几何 ----------
    conductors = []
    names = []
    is_ground_wire = []

    # 相导线
    for i, (x, h_avg) in enumerate(pscad_config.phase_positions):
        rdc = pscad_config.phase_dc_resistance / 1000.0
        cond = zy.ConductorGeometry(
            height=h_avg,
            horizontal_pos=x,
            radius=pscad_config.phase_radius,
            rdc=rdc,
            mu_r=pscad_config.phase_mu_r,
            bundle_n=pscad_config.phase_bundle_n,
            bundle_spacing=pscad_config.phase_bundle_spacing
        )
        conductors.append(cond)
        names.append(f'Phase_{i+1}')
        is_ground_wire.append(False)

        if verbose:
            print(f"  相导线 {i+1}: x={x:.3f}m, h={h_avg:.2f}m, "
                  f"r={pscad_config.phase_radius*1000:.1f}mm, "
                  f"r_eq={cond.equivalent_radius*1000:.2f}mm "
                  f"({pscad_config.phase_bundle_n}分裂, d={pscad_config.phase_bundle_spacing}m)")

    # 地线
    for i, (x, h_avg) in enumerate(pscad_config.gw_positions):
        rdc = pscad_config.gw_dc_resistance / 1000.0
        cond = zy.ConductorGeometry(
            height=h_avg,
            horizontal_pos=x,
            radius=pscad_config.gw_radius,
            rdc=rdc,
            mu_r=pscad_config.gw_mu_r,
            bundle_n=1,
            bundle_spacing=0.0
        )
        conductors.append(cond)
        names.append(f'GW_{i+1}')
        # 只有在启用 Kron 缩减时才标记为地线
        # is_ground_wire.append(pscad_config.kron_reduction)
        is_ground_wire.append(True)

        if verbose:
            print(f"  地线   {i+1}: x={x:.3f}m, h={h_avg:.2f}m, "
                  f"r={pscad_config.gw_radius*1000:.2f}mm")

    line = zy.MultiConductorLine(
        conductors=conductors,
        names=names,
        is_ground_wire=is_ground_wire
    )

    # ---------- 土壤参数 ----------
    soil = zy.get_constant_soil_params(
        freq,
        pscad_config.ground_resistivity,
        epsilon_r=pscad_config.ground_permittivity
    )

    if verbose:
        idx_50 = np.argmin(np.abs(freq - 50))
        p50 = soil.p_complex[idx_50]
        print(f"\n  土壤电阻率: {pscad_config.ground_resistivity} Ω·m")
        print(f"  复数穿透深度 |p| @ 50Hz: {np.abs(p50):.1f} m")

    # ---------- 计算 Z/Y ----------
    if verbose:
        print("\n  计算阻抗矩阵 Z ...")
    Z_result = zy.compute_impedance_matrix(freq, line, soil.p_complex, verbose=False)
    Z_matrix = Z_result.Z_matrix

    if verbose:
        print("  计算导纳矩阵 Y ...")
    Y_result = zy.compute_admittance_matrix(freq, line, verbose=False)
    Y_matrix = Y_result.Y_matrix

    if verbose:
        print(f"\n  Z_matrix: {Z_matrix.shape},  Y_matrix: {Y_matrix.shape}")

    return Z_matrix, Y_matrix, line


# =============================================================================
# 第四部分：通过 Vector Fitting 模块计算 Yc 和 H
# =============================================================================
def compute_Yc_H_via_VF(modules: Dict,
                         freq: np.ndarray,
                         Z_matrix: np.ndarray,
                         Y_matrix: np.ndarray,
                         line_length: float,
                         vf_config: Optional[object] = None,
                         use_freq_dependent: Union[str, bool] = 'never',
                         verbose: bool = True
                         ) -> Dict:
    """
    调用 vector_fitting_v47_fixed 模块的 ulm_complete_fitting() 计算 Yc 和 H

    该函数内部执行:
      1. Newton-Raphson 特征求解 → T_I 变换矩阵
      2. 矩阵开方 → Yc = Z⁻¹ × √(ZY),  H = T diag(e^{-γL}) T⁻¹
      3. 延时提取 τᵢ (Gustavsen 2017)
      4. tr(Yc) 的 Vector Fitting 极点拟合
      5. 各模态 H 的 Vector Fitting 极点拟合
      6. 被动性检查

    Parameters
    ----------
    modules       : Dict  已导入的模块
    freq          : (K,)  频率向量 [Hz]
    Z_matrix      : (K, n, n)  阻抗矩阵
    Y_matrix      : (K, n, n)  导纳矩阵
    line_length   : float  线路长度 [m]
    vf_config     : IterativePoleFindingConfig, optional
    use_freq_dependent : Union[str, bool]  'auto'/'always'/'never' or bool
    verbose       : bool

    Returns
    -------
    dict 包含:
      Yc_phase      : (K, n, n) 相域特性导纳 (计算值)
      H_phase       : (K, n, n) 相域传播函数 (计算值)
      H_mode        : (K, n)    模态传播函数 (计算值)
      gamma_mode    : (K, n)    模态传播常数
      T_I           : (K, n, n) 变换矩阵 (频率相关时) 或 None
      T_ref         : (n, n)    参考频率处变换矩阵
      D_matrices    : D 矩阵 (论文公式13)
      ulm_params    : ULMParameters 对象
      fitting_result: ULMFittingResult 对象 (含极点拟合信息)
      Yc_trace      : (K,) tr(Yc) 值
      tau           : (n,) 各模态延时
    """
    vf = modules['vf']

    if verbose:
        print("\n" + "=" * 65)
        print(f" 通过 Vector Fitting 模块计算 Yc 和 H")
        print(f" (线路长度 L = {line_length/1000:.1f} km)")
        print("=" * 65)

    # ---------- 配置 VF 参数 ----------
    if vf_config is None:
        vf_config = vf.IterativePoleFindingConfig(
            Ymin=6,
            Ymax=20,
            epsY=0.002,
            Hmin=8,
            Hmax=20,
            epsH=0.002,
            pole_step=2,
            eps_deg=10.0,
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

    # ---------- 调用 ulm_complete_fitting ----------
    ulm_params, fitting_result = vf.ulm_complete_fitting(
        freq=freq,
        Z_matrix=Z_matrix,
        Y_matrix=Y_matrix,
        length=line_length,
        velocity_freq=1e5,
        config=vf_config,
        use_freq_dependent=use_freq_dependent,
        enforce_passivity_flag=True,
        verbose=verbose,
    )

    # ---------- 提取结果 ----------
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

        # tr(Yc) 拟合质量
        print(f"\n  tr(Yc) 拟合 RMSE: {fitting_result.Yc_trace_rmse * 100:.4f}%")
        print(f"  H 矩阵重构 RMSE: {fitting_result.H_matrix_rmse * 100:.4f}%")
        print(f"  被动性: {'✓' if fitting_result.is_passive else '✗'}")

        # 延时信息
        for i in range(n_cond):
            tau_us = ulm_params.tau[i] * 1e6
            print(f"    Mode {i+1}: τ = {tau_us:.4f} μs")

        # 显示典型频率值
        idx_50 = np.argmin(np.abs(freq - 50))
        print(f"\n  @ {freq[idx_50]:.2f} Hz:")
        for i in range(n_cond):
            g = ulm_params.gamma_modes[idx_50, i]
            v_phase = 2 * np.pi * freq[idx_50] / np.abs(np.imag(g)) if np.abs(np.imag(g)) > 1e-30 else np.inf
            v_ratio = v_phase / C_LIGHT
            h_mag = np.abs(ulm_params.H_modes[idx_50, i])
            print(f"    Mode {i+1}: γ = {np.real(g):.6e} + j{np.imag(g):.6e}  "
                  f"|H| = {h_mag:.6f}  v/c = {v_ratio:.4f}")

        # H 模态拟合信息
        if fitting_result.H_modes_fits:
            print(f"\n  H 模态拟合详情:")
            for fit in fitting_result.H_modes_fits:
                if fit is not None:
                    print(f"    Mode {fit.mode_index + 1}: "
                          f"{len(fit.poles)} 极点, "
                          f"RMSE = {fit.rmse * 100:.4f}%, "
                          f"τ = {fit.tau * 1e6:.4f} μs")

        # H 重构验证
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
        'T_I':            ulm_params.TI_matrix,       # 全频率 Ti (频率相关时非 None)
        'T_ref':          ulm_params.T_ref,            # 参考频率处 Ti
        'T_ref_inv':      ulm_params.T_ref_inv,
        'D_matrices':     ulm_params.D_matrices,
        'Yc_trace':       ulm_params.Yc_trace,
        'tau':            ulm_params.tau,
        'ulm_params':     ulm_params,
        'fitting_result': fitting_result,
    }


# =============================================================================
# 第五部分：从 PSCAD 读取 Yc 和 H
# =============================================================================
def load_Yc_H_pscad(modules: Dict,
                     pscad_path: str,
                     verbose: bool = True
                     ) -> Dict:
    """
    从 PSCAD 输出文件读取 Yc 和 H

    Returns
    -------
    dict 包含 freq_pscad, Yc_pscad, H_phase_pscad, H_mode_pscad, reader, ...
    """
    PSCADFileReader = modules['PSCADFileReader']

    if verbose:
        print("\n" + "=" * 65)
        print(" 从 PSCAD 输出文件读取 Yc / H")
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
# 第六部分：误差计算 (复用 compare_zy_with_pscad 的逻辑)
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
    """
    计算模态向量的误差 (一维: 每列为一个模态)

    Parameters
    ----------
    calc : (K, n_modes)  计算值
    ref  : (K, n_modes)  参考值
    """
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
# 第六-B 部分：Ti 矩阵对比工具
# =============================================================================
def normalize_Ti_columns(Ti: np.ndarray) -> np.ndarray:
    """
    对变换矩阵的每列进行归一化：
    1. 按列 2-范数归一化
    2. 使绝对值最大元素为正实数 (消除符号/相位不确定性)

    Parameters
    ----------
    Ti : ndarray, shape (n, n) — 单频率点的变换矩阵

    Returns
    -------
    Ti_norm : ndarray, shape (n, n)
    """
    n = Ti.shape[0]
    Ti_norm = Ti.copy()
    for j in range(n):
        col = Ti_norm[:, j]
        # 2-范数归一化
        nrm = np.linalg.norm(col)
        if nrm > 1e-30:
            col = col / nrm
        # 使绝对值最大元素的相角为 0 (相当于整列乘以 e^{-jθ})
        idx_max = np.argmax(np.abs(col))
        phase_max = np.angle(col[idx_max])
        col = col * np.exp(-1j * phase_max)
        Ti_norm[:, j] = col
    return Ti_norm


def align_Ti_columns(Ti_calc: np.ndarray, Ti_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对齐两个变换矩阵的列顺序和符号。
    通过最大相关性匹配列，并调整符号使内积为正。

    Parameters
    ----------
    Ti_calc : (n, n) 计算值
    Ti_ref  : (n, n) 参考值 (PSCAD)

    Returns
    -------
    Ti_calc_aligned : (n, n) 对齐后的计算值
    Ti_ref_aligned  : (n, n) 对齐后的参考值
    perm            : (n,) 列排列索引
    """
    n = Ti_calc.shape[0]
    Ti_c = normalize_Ti_columns(Ti_calc)
    Ti_r = normalize_Ti_columns(Ti_ref)

    # 相关矩阵: corr[i,j] = |Ti_c[:,i]^H · Ti_r[:,j]|
    corr = np.abs(Ti_c.conj().T @ Ti_r)

    # 贪心匹配
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

    # 重排列参考列以匹配计算列
    Ti_ref_aligned = Ti_r[:, perm]

    return Ti_c, Ti_ref_aligned, perm


def compute_Ti_comparison(Ti_calc_all: np.ndarray,
                          Ti_ref_all: np.ndarray,
                          freq: np.ndarray,
                          names: List[str],
                          verbose: bool = True) -> Dict:
    """
    计算 Ti 矩阵的对比误差（考虑特征向量的归一化和符号不确定性）

    Parameters
    ----------
    Ti_calc_all : (K, n_calc, n_calc) 计算值 (全频率)
    Ti_ref_all  : (K, n_ref, n_ref)   PSCAD参考值 (全频率)
    freq        : (K,) 频率
    names       : 导体名称列表
    verbose     : 是否输出

    Returns
    -------
    dict 包含:
        mode_correlations : (K, n) 各模态的相关系数 (理想=1)
        max_column_error  : (K, n) 各模态归一化列误差
        summary           : dict 汇总统计
    """
    K = len(freq)
    n_calc = Ti_calc_all.shape[1]
    n_ref = Ti_ref_all.shape[1]
    n = min(n_calc, n_ref)

    if verbose:
        if n_calc != n_ref:
            print(f"\n  ⚠ Ti 维度不匹配: 自研 {n_calc}×{n_calc}, PSCAD {n_ref}×{n_ref}")
            print(f"  → 取前 {n}×{n} 进行对比 (需注意物理含义)")

    # 逐频率对齐和比较
    mode_corr = np.zeros((K, n))
    col_errors = np.zeros((K, n))

    for k in range(K):
        Ti_c = Ti_calc_all[k, :n, :n]
        Ti_r = Ti_ref_all[k, :n, :n]

        Ti_c_norm, Ti_r_aligned, _ = align_Ti_columns(Ti_c, Ti_r)

        # 各列相关系数
        for j in range(n):
            inner = np.abs(np.vdot(Ti_c_norm[:, j], Ti_r_aligned[:, j]))
            mode_corr[k, j] = inner

            # 列误差 (2-范数)
            col_errors[k, j] = np.linalg.norm(Ti_c_norm[:, j] - Ti_r_aligned[:, j])

    # 汇总
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
# 第七部分：绘图工具
# =============================================================================

def plot_full_matrix_comparison(freq: np.ndarray,
                                M_calc: np.ndarray,
                                M_ref: np.ndarray,
                                names: List[str],
                                matrix_label: str = "Yc",
                                unit_mag: str = "S",
                                mag_scale: float = 1.0,
                                use_loglog_mag: bool = True,
                                save_path: Optional[str] = None):
    """
    绘制完整矩阵（n×n 所有元素）的幅值与相角对比图
    """
    n = M_calc.shape[1]

    def get_name(idx):
        return names[idx] if idx < len(names) else f'C{idx+1}'

    # =====================================================================
    # 幅值对比图
    # =====================================================================
    fig_mag, axes_mag = plt.subplots(n, n, figsize=(4.2 * n, 3.2 * n), sharex=True)
    if n == 1:
        axes_mag = np.array([[axes_mag]])
    fig_mag.suptitle(f'Complete {matrix_label} Matrix — Magnitude Comparison\n'
                     f'(Blue solid = Calculated, Red dashed = PSCAD)',
                     fontsize=14, y=1.02)

    for i in range(n):
        for j in range(n):
            ax = axes_mag[i, j]
            mag_calc = np.abs(M_calc[:, i, j]) * mag_scale
            mag_ref = np.abs(M_ref[:, i, j]) * mag_scale

            if use_loglog_mag:
                ax.loglog(freq, mag_calc, 'b-', lw=1.3, label='Calculated')
                ax.loglog(freq, mag_ref, 'r--', lw=1.0, label='PSCAD')
            else:
                ax.semilogx(freq, mag_calc, 'b-', lw=1.3, label='Calculated')
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

    # =====================================================================
    # 相角对比图
    # =====================================================================
    fig_phase, axes_phase = plt.subplots(n, n, figsize=(4.2 * n, 3.2 * n), sharex=True)
    if n == 1:
        axes_phase = np.array([[axes_phase]])
    fig_phase.suptitle(f'Complete {matrix_label} Matrix — Phase Comparison\n'
                       f'(Blue solid = Calculated, Red dashed = PSCAD)',
                       fontsize=14, y=1.02)

    for i in range(n):
        for j in range(n):
            ax = axes_phase[i, j]
            phase_calc = np.angle(M_calc[:, i, j], deg=True)
            phase_ref = np.angle(M_ref[:, i, j], deg=True)

            ax.semilogx(freq, phase_calc, 'b-', lw=1.3, label='Calculated')
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


def plot_modal_H_comparison(freq: np.ndarray,
                             H_mode_calc: np.ndarray,
                             H_mode_ref: np.ndarray,
                             mode_names: List[str],
                             save_path: Optional[str] = None):
    """
    绘制模态传播函数 H 的对比图 (幅值 + 相角)

    支持 calc 和 ref 模态数不同，以最大模态数为准绘制所有子图。
    对于某一方缺少的模态，仅绘制有数据的一方。

    Parameters
    ----------
    H_mode_calc : (K, n_modes_calc)  自研计算
    H_mode_ref  : (K, n_modes_ref)   PSCAD 参考
    """
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

        # 幅值
        ax = axes[i, 0]
        if has_calc:
            ax.semilogx(freq, np.abs(H_mode_calc[:, i]), 'b-', lw=1.5, label='Calculated')
        if has_ref:
            ax.semilogx(freq, np.abs(H_mode_ref[:, i]), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel(f'|H| Mode {i+1}')
        suffix = ''
        if has_calc and not has_ref:
            suffix = ' (Calc only)'
        elif has_ref and not has_calc:
            suffix = ' (PSCAD only)'
        ax.set_title(f'{name} — Magnitude{suffix}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.set_ylim([0, 1.1])

        # 相角
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


def plot_lambda_comparison(freq: np.ndarray,
                           gamma_calc: np.ndarray,
                           lambda_ref: np.ndarray,
                           mode_names: List[str],
                           save_path: Optional[str] = None):
    """
    绘制传播常数 λ = γ² 对比图 (实部 + 虚部)

    将自研计算的 γ 平方后得到 λ_calc = γ²，与 PSCAD 输出的 Lambda 直接对比。
    PSCAD 的 Lambda 文件存储的是 ZY 的特征值 λ，而非 γ。

    Parameters
    ----------
    gamma_calc : (K, n_modes) 自研计算的模态传播常数 γ
    lambda_ref : (K, n_modes) PSCAD 输出的 Lambda (= γ²)
    mode_names : 模态名称列表
    save_path  : 保存路径
    """
    # 自研侧: λ = γ²
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

        # Re(λ)
        ax = axes[0, i]
        ax.loglog(freq, np.abs(np.real(lambda_calc[:, i])), 'b-', lw=1.5, label='Calc (γ²)')
        ax.loglog(freq, np.abs(np.real(lambda_ref[:, i])), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('|Re(λ)| [1/m²]')
        ax.set_title(f'{name} — Re(λ)')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        # Im(λ)
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


def plot_Ti_matrix_elements(freq: np.ndarray,
                             Ti_calc: np.ndarray,
                             Ti_ref: np.ndarray,
                             names: List[str],
                             save_path: Optional[str] = None):
    """
    绘制 Ti 矩阵每个元素的幅值和相角对比图

    Parameters
    ----------
    Ti_calc : (K, n_calc, n_calc)
    Ti_ref  : (K, n_ref, n_ref)
    """
    n_calc = Ti_calc.shape[1]
    n_ref = Ti_ref.shape[1]
    n = min(n_calc, n_ref)

    def get_name(idx):
        return names[idx] if idx < len(names) else f'C{idx+1}'

    # ---- 幅值 ----
    fig_mag, axes_mag = plt.subplots(n, n, figsize=(4.2 * n, 3.2 * n), sharex=True)
    if n == 1:
        axes_mag = np.array([[axes_mag]])
    fig_mag.suptitle(f'Ti Matrix — Magnitude Comparison (n={n})\n'
                     f'(Blue solid = Calculated, Red dashed = PSCAD)',
                     fontsize=14, y=1.02)

    for i in range(n):
        for j in range(n):
            ax = axes_mag[i, j]
            ax.semilogx(freq, np.abs(Ti_calc[:, i, j]), 'b-', lw=1.3, label='Calculated')
            ax.semilogx(freq, np.abs(Ti_ref[:, i, j]), 'r--', lw=1.0, label='PSCAD')
            ax.set_title(f'Ti({get_name(i)}, {get_name(j)})', fontsize=9, pad=3)
            ax.grid(True, which='both', ls=':', alpha=0.4)
            ax.tick_params(labelsize=7)
            if i == j:
                ax.set_facecolor('#f0f8ff')
            if j == 0:
                ax.set_ylabel('|Ti|', fontsize=8)
            if i == n - 1:
                ax.set_xlabel('Frequency [Hz]', fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best')

    fig_mag.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_Ti_full_magnitude.png')
        fig_mag.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    # ---- 相角 ----
    fig_phase, axes_phase = plt.subplots(n, n, figsize=(4.2 * n, 3.2 * n), sharex=True)
    if n == 1:
        axes_phase = np.array([[axes_phase]])
    fig_phase.suptitle(f'Ti Matrix — Phase Comparison (n={n})\n'
                       f'(Blue solid = Calculated, Red dashed = PSCAD)',
                       fontsize=14, y=1.02)

    for i in range(n):
        for j in range(n):
            ax = axes_phase[i, j]
            ax.semilogx(freq, np.angle(Ti_calc[:, i, j], deg=True), 'b-', lw=1.3, label='Calculated')
            ax.semilogx(freq, np.angle(Ti_ref[:, i, j], deg=True), 'r--', lw=1.0, label='PSCAD')
            ax.set_title(f'∠Ti({get_name(i)}, {get_name(j)})', fontsize=9, pad=3)
            ax.grid(True, which='both', ls=':', alpha=0.4)
            ax.tick_params(labelsize=7)
            if i == j:
                ax.set_facecolor('#f0f8ff')
            if j == 0:
                ax.set_ylabel('∠Ti [°]', fontsize=8)
            if i == n - 1:
                ax.set_xlabel('Frequency [Hz]', fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best')

    fig_phase.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_Ti_full_phase.png')
        fig_phase.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    return fig_mag, fig_phase


def plot_Ti_column_correlation(freq: np.ndarray,
                                mode_corr: np.ndarray,
                                col_errors: np.ndarray,
                                save_path: Optional[str] = None):
    """
    绘制 Ti 各模态列向量的相关系数和列误差随频率变化

    Parameters
    ----------
    mode_corr  : (K, n_modes) 相关系数
    col_errors : (K, n_modes) 列误差
    """
    n_modes = mode_corr.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Ti Column Vector Alignment Quality\n'
                 '(after normalization and sign alignment)',
                 fontsize=13, y=1.0)

    # 相关系数
    ax = axes[0]
    for j in range(n_modes):
        ax.semilogx(freq, mode_corr[:, j], lw=1.5, label=f'Mode {j+1}')
    ax.axhline(1.0, color='k', ls='--', alpha=0.3)
    ax.set_ylabel('Column Correlation |⟨c, r⟩|')
    ax.set_title('Normalized Column Inner Product (ideal = 1.0)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.set_ylim([max(0, np.min(mode_corr) - 0.05), 1.05])

    # 列误差
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


def plot_Yc_self_comparison(freq: np.ndarray,
                             Yc_calc: np.ndarray,
                             Yc_ref: np.ndarray,
                             names: List[str],
                             save_path: Optional[str] = None):
    """
    绘制 Yc 自导纳 (对角元素) 详细对比 + 等效 Zc
    """
    n_cond = Yc_calc.shape[1]
    n_show = min(n_cond, 4)

    fig, axes = plt.subplots(3, n_show, figsize=(4.5 * n_show, 10), sharex=True)
    if n_show == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle('Characteristic Admittance Yc — Self Elements Comparison', fontsize=13, y=1.0)

    for i in range(n_show):
        name = names[i] if i < len(names) else f'C{i+1}'

        # |Yc| (S)
        ax = axes[0, i]
        ax.semilogx(freq, np.abs(Yc_calc[:, i, i]) * 1e3, 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, np.abs(Yc_ref[:, i, i]) * 1e3, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('|Yc| [mS]')
        ax.set_title(f'{name}')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        # ∠Yc (°)
        ax = axes[1, i]
        ax.semilogx(freq, np.angle(Yc_calc[:, i, i], deg=True), 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, np.angle(Yc_ref[:, i, i], deg=True), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('∠Yc [°]')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        # 等效 |Zc| (Ω)
        with np.errstate(divide='ignore', invalid='ignore'):
            Zc_calc = 1.0 / np.abs(Yc_calc[:, i, i])
            Zc_ref = 1.0 / np.abs(Yc_ref[:, i, i])
            Zc_calc = np.where(np.isfinite(Zc_calc), Zc_calc, np.nan)
            Zc_ref = np.where(np.isfinite(Zc_ref), Zc_ref, np.nan)
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


def plot_error_vs_freq(freq: np.ndarray,
                        Yc_calc: np.ndarray, Yc_ref: np.ndarray,
                        H_calc: np.ndarray, H_ref: np.ndarray,
                        names: List[str],
                        save_path: Optional[str] = None):
    """
    绘制 Yc 和 H 的全矩阵误差随频率变化曲线（所有 n×n 元素）
    """
    n_cond = Yc_calc.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Yc / H Full Matrix Error vs Frequency', fontsize=13)

    # Yc 全矩阵幅值误差
    ax = axes[0, 0]
    for i in range(n_cond):
        for j in range(n_cond):
            mag_ref = np.abs(Yc_ref[:, i, j])
            mag_calc = np.abs(Yc_calc[:, i, j])
            threshold = np.max(mag_ref) * 0.01
            valid = mag_ref > threshold
            err = np.zeros_like(mag_ref)
            err[valid] = np.abs(mag_calc[valid] - mag_ref[valid]) / mag_ref[valid] * 100
            name_i = names[i] if i < len(names) else f'C{i+1}'
            name_j = names[j] if j < len(names) else f'C{j+1}'
            ls = '-' if i == j else '--'
            ax.semilogx(freq, err, ls=ls, lw=1.2, label=f'Yc({name_i},{name_j})')
    ax.set_ylabel('Magnitude Error [%]')
    ax.set_title('Yc Full Matrix Magnitude Error')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # Yc 全矩阵相角误差
    ax = axes[0, 1]
    for i in range(n_cond):
        for j in range(n_cond):
            ph_calc = np.angle(Yc_calc[:, i, j], deg=True)
            ph_ref = np.angle(Yc_ref[:, i, j], deg=True)
            ph_err = np.abs(ph_calc - ph_ref)
            ph_err = np.minimum(ph_err, 360 - ph_err)
            name_i = names[i] if i < len(names) else f'C{i+1}'
            name_j = names[j] if j < len(names) else f'C{j+1}'
            ls = '-' if i == j else '--'
            ax.semilogx(freq, ph_err, ls=ls, lw=1.2, label=f'Yc({name_i},{name_j})')
    ax.set_ylabel('Phase Error [°]')
    ax.set_title('Yc Full Matrix Phase Error')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # H 全矩阵幅值误差
    ax = axes[1, 0]
    for i in range(n_cond):
        for j in range(n_cond):
            mag_ref = np.abs(H_ref[:, i, j])
            mag_calc = np.abs(H_calc[:, i, j])
            threshold = np.max(mag_ref) * 0.01
            valid = mag_ref > threshold
            err = np.zeros_like(mag_ref)
            err[valid] = np.abs(mag_calc[valid] - mag_ref[valid]) / mag_ref[valid] * 100
            name_i = names[i] if i < len(names) else f'C{i+1}'
            name_j = names[j] if j < len(names) else f'C{j+1}'
            ls = '-' if i == j else '--'
            ax.semilogx(freq, err, ls=ls, lw=1.2, label=f'H({name_i},{name_j})')
    ax.set_ylabel('Magnitude Error [%]')
    ax.set_title('H_phase Full Matrix Magnitude Error')
    ax.set_xlabel('Frequency [Hz]')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # H 全矩阵相角误差
    ax = axes[1, 1]
    for i in range(n_cond):
        for j in range(n_cond):
            ph_calc = np.angle(H_calc[:, i, j], deg=True)
            ph_ref = np.angle(H_ref[:, i, j], deg=True)
            ph_err = np.abs(ph_calc - ph_ref)
            ph_err = np.minimum(ph_err, 360 - ph_err)
            name_i = names[i] if i < len(names) else f'C{i+1}'
            name_j = names[j] if j < len(names) else f'C{j+1}'
            ls = '-' if i == j else '--'
            ax.semilogx(freq, ph_err, ls=ls, lw=1.2, label=f'H({name_i},{name_j})')
    ax.set_ylabel('Phase Error [°]')
    ax.set_title('H_phase Full Matrix Phase Error')
    ax.set_xlabel('Frequency [Hz]')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    fig.tight_layout()

    if save_path:
        p = save_path.replace('.png', '_Yc_H_error.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    return fig


def plot_ZY_comparison(freq: np.ndarray,
                       Z_calc: np.ndarray, Z_ref: np.ndarray,
                       Y_calc: np.ndarray, Y_ref: np.ndarray,
                       names: List[str],
                       save_path: Optional[str] = None) -> List:
    """
    生成 Z/Y 全矩阵对比图（与 compare_zy_with_pscad_0211.py 风格一致，共 8 页）:
      1. 自阻抗幅值+相角
      2. 互阻抗幅值+相角
      3. 等效 R/L/C
      4. 误差随频率变化
      5. 完整 Z 矩阵幅值对比
      6. 完整 Z 矩阵相角对比
      7. 完整 Y 矩阵幅值对比
      8. 完整 Y 矩阵相角对比

    Parameters
    ----------
    freq   : (K,) 频率数组
    Z_calc : (K, n, n) 自研计算的 Z 矩阵
    Z_ref  : (K, n, n) PSCAD 参考 Z 矩阵
    Y_calc : (K, n, n) 自研计算的 Y 矩阵
    Y_ref  : (K, n, n) PSCAD 参考 Y 矩阵
    names  : 导体名称列表
    save_path : 保存路径基准名

    Returns
    -------
    figs : list of matplotlib Figure
    """
    n_cond = Z_calc.shape[1]
    omega = 2 * np.pi * freq

    all_figs = []

    # =========================================================================
    # 图 1：自阻抗对比 (幅值 + 相角)
    # =========================================================================
    fig1, axes1 = plt.subplots(n_cond, 2, figsize=(14, 3.5 * n_cond), sharex=True)
    if n_cond == 1:
        axes1 = axes1.reshape(1, -1)
    fig1.suptitle('Self Impedance Comparison: Calculated vs PSCAD', fontsize=13, y=1.0)

    for i in range(n_cond):
        name = names[i] if i < len(names) else f'Cond{i+1}'

        # 幅值
        ax = axes1[i, 0]
        ax.loglog(freq, np.abs(Z_calc[:, i, i]) * 1000, 'b-', lw=1.5, label='Calculated')
        ax.loglog(freq, np.abs(Z_ref[:, i, i]) * 1000, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel(f'|Z{i+1}{i+1}| [mΩ/m]')
        ax.set_title(f'{name} — Magnitude')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        # 相角
        ax = axes1[i, 1]
        ax.semilogx(freq, np.angle(Z_calc[:, i, i], deg=True), 'b-', lw=1.5, label='Calculated')
        ax.semilogx(freq, np.angle(Z_ref[:, i, i], deg=True), 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel(f'∠Z{i+1}{i+1} [°]')
        ax.set_title(f'{name} — Phase')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls=':', alpha=0.5)

    for ax in axes1[-1, :]:
        ax.set_xlabel('Frequency [Hz]')
    fig1.tight_layout()
    all_figs.append(fig1)

    if save_path:
        p = save_path.replace('.png', '_ZY_1_Zself.png')
        fig1.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    # =========================================================================
    # 图 2：互阻抗对比（选取代表性元素）
    # =========================================================================
    pairs = []
    for i in range(min(n_cond, 4)):
        for j in range(i + 1, min(n_cond, 4)):
            pairs.append((i, j))
    n_pairs = len(pairs)

    if n_pairs > 0:
        fig2, axes2 = plt.subplots(n_pairs, 2, figsize=(14, 3.5 * max(n_pairs, 1)), sharex=True)
        if n_pairs == 1:
            axes2 = axes2.reshape(1, -1)
        fig2.suptitle('Mutual Impedance Comparison: Calculated vs PSCAD', fontsize=13, y=1.0)

        for idx, (i, j) in enumerate(pairs):
            # 幅值
            ax = axes2[idx, 0]
            ax.loglog(freq, np.abs(Z_calc[:, i, j]) * 1000, 'b-', lw=1.5, label='Calculated')
            ax.loglog(freq, np.abs(Z_ref[:, i, j]) * 1000, 'r--', lw=1.2, label='PSCAD')
            ax.set_ylabel(f'|Z{i+1}{j+1}| [mΩ/m]')
            ax.set_title(f'Z{i+1}{j+1} — Magnitude')
            ax.legend(fontsize=8)
            ax.grid(True, which='both', ls=':', alpha=0.5)

            # 相角
            ax = axes2[idx, 1]
            ax.semilogx(freq, np.angle(Z_calc[:, i, j], deg=True), 'b-', lw=1.5, label='Calculated')
            ax.semilogx(freq, np.angle(Z_ref[:, i, j], deg=True), 'r--', lw=1.2, label='PSCAD')
            ax.set_ylabel(f'∠Z{i+1}{j+1} [°]')
            ax.set_title(f'Z{i+1}{j+1} — Phase')
            ax.legend(fontsize=8)
            ax.grid(True, which='both', ls=':', alpha=0.5)

        for ax in axes2[-1, :]:
            ax.set_xlabel('Frequency [Hz]')
        fig2.tight_layout()
        all_figs.append(fig2)

        if save_path:
            p = save_path.replace('.png', '_ZY_2_Zmutual.png')
            fig2.savefig(p, dpi=150, bbox_inches='tight')
            print(f"  已保存: {p}")

    # =========================================================================
    # 图 3：等效 R / L / C 对比
    # =========================================================================
    n_show = min(n_cond, 4)
    fig3, axes3 = plt.subplots(3, n_show, figsize=(4.5 * n_show, 10), sharex=True)
    if n_show == 1:
        axes3 = axes3.reshape(-1, 1)
    fig3.suptitle('Equivalent R / L / C Comparison', fontsize=13, y=1.0)

    for i in range(n_show):
        name = names[i] if i < len(names) else f'C{i+1}'

        # R (mΩ/m)
        R_calc = np.real(Z_calc[:, i, i]) * 1000
        R_ref = np.real(Z_ref[:, i, i]) * 1000
        ax = axes3[0, i]
        ax.loglog(freq, R_calc, 'b-', lw=1.5, label='Calc')
        ax.loglog(freq, R_ref, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('R [mΩ/m]')
        ax.set_title(f'{name}')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        # L (μH/m)
        with np.errstate(divide='ignore', invalid='ignore'):
            L_calc = np.imag(Z_calc[:, i, i]) / omega * 1e6
            L_ref = np.imag(Z_ref[:, i, i]) / omega * 1e6
        ax = axes3[1, i]
        ax.semilogx(freq, L_calc, 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, L_ref, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('L [μH/m]')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        # C (pF/m)
        with np.errstate(divide='ignore', invalid='ignore'):
            C_calc = np.imag(Y_calc[:, i, i]) / omega * 1e12
            C_ref = np.imag(Y_ref[:, i, i]) / omega * 1e12
        ax = axes3[2, i]
        ax.semilogx(freq, C_calc, 'b-', lw=1.5, label='Calc')
        ax.semilogx(freq, C_ref, 'r--', lw=1.2, label='PSCAD')
        ax.set_ylabel('C [pF/m]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

    fig3.tight_layout()
    all_figs.append(fig3)

    if save_path:
        p = save_path.replace('.png', '_ZY_3_RLC.png')
        fig3.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    # =========================================================================
    # 图 4：全矩阵误差随频率变化
    # =========================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('Z / Y Full Matrix Error vs Frequency', fontsize=13)

    # Z 全矩阵幅值误差
    ax = axes4[0, 0]
    for i in range(n_cond):
        for j in range(n_cond):
            mag_ref = np.abs(Z_ref[:, i, j])
            mag_calc = np.abs(Z_calc[:, i, j])
            threshold = np.max(mag_ref) * 0.01
            valid = mag_ref > threshold
            err = np.zeros_like(mag_ref)
            err[valid] = np.abs(mag_calc[valid] - mag_ref[valid]) / mag_ref[valid] * 100
            name_i = names[i] if i < len(names) else f'C{i+1}'
            name_j = names[j] if j < len(names) else f'C{j+1}'
            ls = '-' if i == j else '--'
            ax.semilogx(freq, err, ls=ls, lw=1.2, label=f'Z({name_i},{name_j})')
    ax.set_ylabel('Magnitude Error [%]')
    ax.set_title('Z Full Matrix Magnitude Error')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # Z 全矩阵相角误差
    ax = axes4[0, 1]
    for i in range(n_cond):
        for j in range(n_cond):
            ph_calc = np.angle(Z_calc[:, i, j], deg=True)
            ph_ref = np.angle(Z_ref[:, i, j], deg=True)
            ph_err = np.abs(ph_calc - ph_ref)
            ph_err = np.minimum(ph_err, 360 - ph_err)
            name_i = names[i] if i < len(names) else f'C{i+1}'
            name_j = names[j] if j < len(names) else f'C{j+1}'
            ls = '-' if i == j else '--'
            ax.semilogx(freq, ph_err, ls=ls, lw=1.2, label=f'Z({name_i},{name_j})')
    ax.set_ylabel('Phase Error [°]')
    ax.set_title('Z Full Matrix Phase Error')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # Y 全矩阵幅值误差
    ax = axes4[1, 0]
    for i in range(n_cond):
        for j in range(n_cond):
            mag_ref = np.abs(Y_ref[:, i, j])
            mag_calc = np.abs(Y_calc[:, i, j])
            threshold = np.max(mag_ref) * 0.01
            valid = mag_ref > threshold
            err = np.zeros_like(mag_ref)
            err[valid] = np.abs(mag_calc[valid] - mag_ref[valid]) / mag_ref[valid] * 100
            name_i = names[i] if i < len(names) else f'C{i+1}'
            name_j = names[j] if j < len(names) else f'C{j+1}'
            ls = '-' if i == j else '--'
            ax.semilogx(freq, err, ls=ls, lw=1.2, label=f'Y({name_i},{name_j})')
    ax.set_ylabel('Magnitude Error [%]')
    ax.set_title('Y Full Matrix Magnitude Error')
    ax.set_xlabel('Frequency [Hz]')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # Y 全矩阵相角误差
    ax = axes4[1, 1]
    for i in range(n_cond):
        for j in range(n_cond):
            ph_calc = np.angle(Y_calc[:, i, j], deg=True)
            ph_ref = np.angle(Y_ref[:, i, j], deg=True)
            ph_err = np.abs(ph_calc - ph_ref)
            ph_err = np.minimum(ph_err, 360 - ph_err)
            name_i = names[i] if i < len(names) else f'C{i+1}'
            name_j = names[j] if j < len(names) else f'C{j+1}'
            ls = '-' if i == j else '--'
            ax.semilogx(freq, ph_err, ls=ls, lw=1.2, label=f'Y({name_i},{name_j})')
    ax.set_ylabel('Phase Error [°]')
    ax.set_title('Y Full Matrix Phase Error')
    ax.set_xlabel('Frequency [Hz]')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    fig4.tight_layout()
    all_figs.append(fig4)

    if save_path:
        p = save_path.replace('.png', '_ZY_4_Error.png')
        fig4.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    # =========================================================================
    # 图 5-6：完整 Z 矩阵幅值 + 相角对比
    # =========================================================================
    fig5_mag, fig5_phase = plot_full_matrix_comparison(
        freq, Z_calc, Z_ref, names,
        matrix_label="Z", unit_mag="mΩ/m", mag_scale=1000.0,
        use_loglog_mag=True, save_path=save_path
    )
    all_figs.extend([fig5_mag, fig5_phase])

    # =========================================================================
    # 图 7-8：完整 Y 矩阵幅值 + 相角对比
    # =========================================================================
    fig6_mag, fig6_phase = plot_full_matrix_comparison(
        freq, Y_calc, Y_ref, names,
        matrix_label="Y", unit_mag="μS/m", mag_scale=1e6,
        use_loglog_mag=True, save_path=save_path
    )
    all_figs.extend([fig6_mag, fig6_phase])

    return all_figs


def print_ZY_spot_comparison(freq: np.ndarray,
                              Z_calc: np.ndarray, Z_ref: np.ndarray,
                              Y_calc: np.ndarray, Y_ref: np.ndarray,
                              names: List[str],
                              spot_freqs: List[float] = None):
    """在几个典型频率点打印 Z 和 Y 的详细对比值（与 compare_zy_with_pscad_0211 风格一致）"""

    if spot_freqs is None:
        spot_freqs = [0.5, 50, 500, 5000, 50000, 100000]

    n = Z_calc.shape[1]

    print("\n" + "=" * 90)
    print(" 典型频率点 Z / Y 详细对比")
    print("=" * 90)

    for f_target in spot_freqs:
        if f_target < freq[0] or f_target > freq[-1]:
            continue
        idx = np.argmin(np.abs(freq - f_target))
        f_actual = freq[idx]

        print(f"\n  ──── @ {f_actual:.2f} Hz ────")
        print(f"  {'元素':>10s} | {'|Calc|':>12s} | {'|PSCAD|':>12s} | "
              f"{'幅值误差%':>10s} | {'∠Calc°':>10s} | {'∠PSCAD°':>10s} | {'相角误差°':>10s}")
        print("  " + "-" * 88)

        # Z: 全矩阵 (mΩ/m)
        for i in range(n):
            for j in range(n):
                zc = Z_calc[idx, i, j]
                zr = Z_ref[idx, i, j]
                mag_c, mag_r = np.abs(zc) * 1000, np.abs(zr) * 1000
                ph_c, ph_r = np.angle(zc, deg=True), np.angle(zr, deg=True)
                mag_err = abs(mag_c - mag_r) / max(mag_r, 1e-30) * 100
                ph_err = abs(ph_c - ph_r)
                ph_err = min(ph_err, 360 - ph_err)
                print(f"  Z{i+1}{j+1:8d} | {mag_c:12.6f} | {mag_r:12.6f} | "
                      f"{mag_err:10.4f} | {ph_c:10.4f} | {ph_r:10.4f} | {ph_err:10.4f}")

        # Y 全矩阵 (μS/m)
        print()
        for i in range(n):
            for j in range(n):
                yc = Y_calc[idx, i, j]
                yr = Y_ref[idx, i, j]
                mag_c, mag_r = np.abs(yc) * 1e6, np.abs(yr) * 1e6
                ph_c, ph_r = np.angle(yc, deg=True), np.angle(yr, deg=True)
                mag_err = abs(mag_c - mag_r) / max(mag_r, 1e-30) * 100
                ph_err = abs(ph_c - ph_r)
                ph_err = min(ph_err, 360 - ph_err)
                print(f"  Y{i+1}{j+1:8d} | {mag_c:12.6f} | {mag_r:12.6f} | "
                      f"{mag_err:10.4f} | {ph_c:10.4f} | {ph_r:10.4f} | {ph_err:10.4f}")


def plot_VF_Yc_calc_vs_fitted(freq: np.ndarray,
                               calc_results: Dict,
                               names: List[str],
                               save_path: Optional[str] = None):
    """
    绘制 VF 自研 Yc 计算值 vs VF 拟合值对比图
    (对角元素幅值 + 相角, 类似 PSCAD 内部 Yc Calc vs Fit 图)

    采用"先画拟合 (粗虚线) → 再画计算 (细实线)"的策略确保虚线可见。
    """
    fitting_result = calc_results.get('fitting_result')
    Yc_calc = calc_results.get('Yc_phase')

    if fitting_result is None or Yc_calc is None:
        return None

    poles_Yc = getattr(fitting_result, 'poles_Yc', None)
    k0 = getattr(fitting_result, 'k0', None)
    k_residues = getattr(fitting_result, 'k_residues', None)

    if poles_Yc is None or k0 is None or k_residues is None:
        print("  ⚠ VF Yc 拟合数据不完整，跳过 Yc calc vs fitted 图")
        return None

    n_cond = Yc_calc.shape[1]
    K = len(freq)

    # ---- 从极点/留数重构完整 Yc 拟合矩阵 ----
    s = 1j * 2 * np.pi * freq
    Yc_fitted = np.zeros((K, n_cond, n_cond), dtype=complex)

    for k_idx in range(K):
        Yk = k0.astype(complex).copy()
        for n_p, p in enumerate(poles_Yc):
            if np.isreal(p):
                Yk += k_residues[n_p].real / (s[k_idx] - p)
            else:
                Yk += k_residues[n_p] / (s[k_idx] - p)
                Yk += k_residues[n_p].conj() / (s[k_idx] - p.conj())
        Yc_fitted[k_idx] = Yk

    # ---- 绘图: 对角元素 (先 Fit 粗虚线, 后 Calc 细实线) ----
    n_show = min(n_cond, 6)

    fig, axes = plt.subplots(2, n_show, figsize=(4.5 * n_show, 7), sharex=True)
    if n_show == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle('VF Yc: Calculated vs Vector Fitting Result\n'
                 f'(poles = {len(poles_Yc)}, '
                 f'RMSE = {fitting_result.Yc_trace_rmse * 100:.3f}%)',
                 fontsize=13)

    for i in range(n_show):
        name = names[i] if i < len(names) else f'C{i+1}'

        # 幅值 (mS) — 先 Fit 后 Calc
        ax = axes[0, i]
        ax.semilogx(freq, np.abs(Yc_fitted[:, i, i]) * 1e3,
                     'r--', lw=2.5, label='VF Fit')
        ax.semilogx(freq, np.abs(Yc_calc[:, i, i]) * 1e3,
                     'b-', lw=1.2, label='VF Calc')
        ax.set_ylabel(f'|Yc{i+1}{i+1}| [mS]')
        ax.set_title(f'Yc{i+1}{i+1}')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        # 相角 (°) — 先 Fit 后 Calc
        ax = axes[1, i]
        ax.semilogx(freq, np.angle(Yc_fitted[:, i, i], deg=True),
                     'r--', lw=2.5, label='VF Fit')
        ax.semilogx(freq, np.angle(Yc_calc[:, i, i], deg=True),
                     'b-', lw=1.2, label='VF Calc')
        ax.set_ylabel(f'∠Yc{i+1}{i+1} [°]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)

    fig.tight_layout()

    if save_path:
        p = save_path.replace('.png', '_Yc_VF_fitting.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    return fig

def plot_VF_H_mode_calc_vs_fitted(freq: np.ndarray,
                                    calc_results: Dict,
                                    save_path: Optional[str] = None):
    """
    绘制 VF 自研 P 模态 (去掉最小延时的 H) 计算值 vs VF 拟合值对比图
    (幅值 + 相角, 类似 PSCAD 内部 Pj Calc vs Fit 图)

    说明
    ----
    H_j(s) = P_j(s) * exp(-s * tau_j)
    本函数画的是 P_j(s) = H_j(s) * exp(+s * tau_j):
      - Calc 曲线:  H_mode_calc * exp(+j*omega*tau_m)
      - Fit  曲线:  sum_k r_k/(s-p_k) + d + e*s   (不再乘 exp(-s*tau_m))

    因为 |exp(-j*omega*tau)| = 1, 幅值图与 H 相同; 但相角图会去掉
    线性相位 -omega*tau, 从而更清楚地显示有理部分的相位行为。

    采用"先画拟合 (粗虚线) → 再画计算 (细实线)"的策略确保虚线可见。
    """
    fitting_result = calc_results.get('fitting_result')
    H_mode_calc = calc_results.get('H_mode')
    tau = calc_results.get('tau')

    if fitting_result is None or H_mode_calc is None:
        return None

    H_fits = getattr(fitting_result, 'H_modes_fits', None)
    if H_fits is None or len(H_fits) == 0:
        print("  ⚠ VF H 模态拟合数据不可用，跳过 P calc vs fitted 图")
        return None

    n_cond = H_mode_calc.shape[1]
    K = len(freq)
    s = 1j * 2 * np.pi * freq

    # ---- 从极点/留数重构各模态 P 拟合值 (不带最小延时) ----
    P_mode_fitted = np.zeros((K, n_cond), dtype=complex)
    # ---- 由 H_calc 剥离最小延时得到 P_calc ----
    P_mode_calc = np.zeros_like(H_mode_calc, dtype=complex)
    tau_per_mode = np.zeros(n_cond)

    fitted_modes = set()
    fit_info = []

    for fit in H_fits:
        if fit is None:
            continue
        m = fit.mode_index
        if m >= n_cond:
            continue

        poles = fit.poles
        residues = fit.residues
        tau_m = fit.tau if hasattr(fit, 'tau') else (tau[m] if tau is not None and m < len(tau) else 0.0)
        d = getattr(fit, 'd', 0.0)
        e = getattr(fit, 'e', 0.0)
        tau_per_mode[m] = tau_m

        for k_idx in range(K):
            val = complex(d) + complex(e) * s[k_idx] if d is not None else 0.0 + 0.0j
            for n_p, p in enumerate(poles):
                if np.isreal(p):
                    val += residues[n_p].real / (s[k_idx] - p)
                else:
                    val += residues[n_p] / (s[k_idx] - p)
                    val += residues[n_p].conj() / (s[k_idx] - p.conj())
            # 注意: 这里不再乘 exp(-s*tau_m), 直接保留 P_j(s) 的有理部分
            P_mode_fitted[k_idx, m] = val

        fitted_modes.add(m)
        fit_info.append((m, len(poles), fit.rmse * 100, tau_m * 1e6))

    # 对所有模态 (无论是否有 fit) 都计算 P_calc = H_calc * exp(+s*tau)
    # 对于没有 fit 的模态, tau_per_mode[m] 默认为 0 或从 tau 数组取
    if tau is not None:
        for m in range(n_cond):
            if m not in fitted_modes and m < len(tau):
                tau_per_mode[m] = tau[m]

    for m in range(n_cond):
        P_mode_calc[:, m] = H_mode_calc[:, m] * np.exp(s * tau_per_mode[m])

    # ---- 绘图: 先 Fit 粗虚线, 后 Calc 细实线 ----
    mode_colors = plt.cm.tab10(np.linspace(0, 1, max(n_cond, 1)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('VF P (Modal, min-delay removed): Calculated vs Vector Fitting Result',
                 fontsize=13)

    # 幅值 — 先 Fit 后 Calc
    ax = axes[0]
    for m in range(n_cond):
        if m in fitted_modes:
            ax.semilogx(freq, np.abs(P_mode_fitted[:, m]), '--',
                         color=mode_colors[m], lw=2.5, label=f'Mode {m+1} Fit')
    for m in range(n_cond):
        ax.semilogx(freq, np.abs(P_mode_calc[:, m]),
                     color=mode_colors[m], lw=1.2, label=f'Mode {m+1} Calc')
    ax.set_ylabel('|P|')
    ax.set_title('Magnitude  (|P| = |H|, 因为 |exp(-jωτ)|=1)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.set_ylim([0, 1.1])

    # 相角 — 先 Fit 后 Calc
    ax = axes[1]
    for m in range(n_cond):
        if m in fitted_modes:
            ax.semilogx(freq, np.angle(P_mode_fitted[:, m], deg=True), '--',
                         color=mode_colors[m], lw=2.5, label=f'Mode {m+1} Fit')
    for m in range(n_cond):
        ax.semilogx(freq, np.angle(P_mode_calc[:, m], deg=True),
                     color=mode_colors[m], lw=1.2, label=f'Mode {m+1} Calc')
    ax.set_ylabel('∠P [°]  (min-delay removed)')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_title('Phase  (线性相位 -ωτ 已剥离)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # 底部标注拟合信息
    if fit_info:
        info_text = '  '.join([f'Mode {m+1}: {np_}p, RMSE={rmse:.3f}%, τ={tau_us:.2f}μs'
                                for m, np_, rmse, tau_us in fit_info])
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=8, style='italic',
                 color='gray')

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        p = save_path.replace('.png', '_P_VF_fitting.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  已保存: {p}")

    return fig



def plot_all_comparisons(freq: np.ndarray,
                          calc_results: Dict,
                          pscad_results: Dict,
                          names: List[str],
                          save_path: Optional[str] = None) -> List:
    """
    生成全部对比图

    Returns
    -------
    figs : list of figures
    """
    figs = []

    Yc_calc = calc_results['Yc_phase']
    H_phase_calc = calc_results['H_phase']
    H_mode_calc = calc_results['H_mode']
    gamma_calc = calc_results['gamma_mode']

    # 从 calc_results 获取 Z/Y (自研计算值)
    Z_calc = calc_results.get('Z_calc')
    Y_calc = calc_results.get('Y_calc')

    # # =====================================================================
    # # 0. Z/Y 矩阵对比 (自研 vs PSCAD) — 与 compare_zy_with_pscad_0211 风格一致
    # # =====================================================================
    # Z_pscad = pscad_results.get('Z_pscad')
    # Y_pscad = pscad_results.get('Y_pscad')
    #
    # if Z_pscad is not None and Y_pscad is not None and Z_calc is not None and Y_calc is not None:
    #     zy_figs = plot_ZY_comparison(
    #         freq, Z_calc, Z_pscad, Y_calc, Y_pscad,
    #         names, save_path=save_path)
    #     figs.extend(zy_figs)
    # else:
    #     # 如果只有 Z 或只有 Y，则分别用全矩阵对比
    #     if Z_pscad is not None and Z_calc is not None:
    #         fig_z_mag, fig_z_phase = plot_full_matrix_comparison(
    #             freq, Z_calc, Z_pscad, names,
    #             matrix_label="Z", unit_mag="mΩ/m", mag_scale=1000.0,
    #             use_loglog_mag=True, save_path=save_path)
    #         figs.extend([fig_z_mag, fig_z_phase])
    #
    #     if Y_pscad is not None and Y_calc is not None:
    #         fig_y_mag, fig_y_phase = plot_full_matrix_comparison(
    #             freq, Y_calc, Y_pscad, names,
    #             matrix_label="Y", unit_mag="μS/m", mag_scale=1e6,
    #             use_loglog_mag=True, save_path=save_path)
    #         figs.extend([fig_y_mag, fig_y_phase])
    #
    # # =====================================================================
    # # 1. Yc 完整 n×n 矩阵对比 + 全矩阵误差曲线
    # # =====================================================================
    # if pscad_results.get('Yc_pscad') is not None:
    #     Yc_ref = pscad_results['Yc_pscad']
    #
    #     # Yc 完整矩阵幅值对比 (全矩阵 n×n)
    #     fig_yc_mag, fig_yc_phase = plot_full_matrix_comparison(
    #         freq, Yc_calc, Yc_ref, names,
    #         matrix_label="Yc", unit_mag="mS", mag_scale=1e3,
    #         use_loglog_mag=True, save_path=save_path)
    #     figs.extend([fig_yc_mag, fig_yc_phase])
    #
    #     # Yc/H 全矩阵误差 vs 频率
    #     if pscad_results.get('H_phase_pscad') is not None:
    #         H_ref = pscad_results['H_phase_pscad']
    #         fig_err = plot_error_vs_freq(
    #             freq, Yc_calc, Yc_ref, H_phase_calc, H_ref,
    #             names, save_path=save_path)
    #         figs.append(fig_err)

    # =====================================================================
    # 4. H 相域完整矩阵对比
    # =====================================================================
    if pscad_results.get('H_phase_pscad') is not None:
        H_phase_ref = pscad_results['H_phase_pscad']

        fig_h_mag, fig_h_phase = plot_full_matrix_comparison(
            freq, H_phase_calc, H_phase_ref, names,
            matrix_label="H_phase", unit_mag="—", mag_scale=1.0,
            use_loglog_mag=False, save_path=save_path)
        figs.extend([fig_h_mag, fig_h_phase])

    # =====================================================================
    # 5. H 模态域对比 (以最大模态数为准，全部绘制)
    # =====================================================================
    if pscad_results.get('H_mode_pscad') is not None:
        H_mode_ref = pscad_results['H_mode_pscad']
        # PSCAD 模态可能是 (K, n_modes) 或 (K, n_modes, n_modes)
        if H_mode_ref.ndim == 3:
            # 取对角 (假设模态 H 矩阵为对角)
            n_modes_ref = H_mode_ref.shape[1]
            H_mode_ref_vec = np.zeros((len(freq), n_modes_ref), dtype=complex)
            for m in range(n_modes_ref):
                H_mode_ref_vec[:, m] = H_mode_ref[:, m, m]
            H_mode_ref = H_mode_ref_vec

        n_modes_ref = H_mode_ref.shape[1]
        n_modes_calc = H_mode_calc.shape[1]
        n_modes = max(n_modes_ref, n_modes_calc)

        mode_names = [f'Mode {i+1}' for i in range(n_modes)]

        fig_h_mode = plot_modal_H_comparison(
            freq,
            H_mode_calc,
            H_mode_ref,
            mode_names, save_path=save_path)
        figs.append(fig_h_mode)

    # # =====================================================================
    # # 6. 传播常数 λ = γ² 对比 (自研 γ² vs PSCAD Lambda)
    # # =====================================================================
    # if pscad_results.get('Lambda_pscad') is not None:
    #     Lambda_ref = pscad_results['Lambda_pscad']
    #     # Lambda 可能是 (K, n_cond) 或 (K, n_cond, n_cond)
    #     if Lambda_ref.ndim == 3:
    #         n_modes_ref = Lambda_ref.shape[1]
    #         lambda_ref = np.zeros((len(freq), n_modes_ref), dtype=complex)
    #         for m in range(n_modes_ref):
    #             lambda_ref[:, m] = Lambda_ref[:, m, m]
    #     else:
    #         lambda_ref = Lambda_ref.copy()
    #
    #     n_modes_ref = lambda_ref.shape[1]
    #     n_modes_calc = gamma_calc.shape[1]
    #     n_modes = min(n_modes_ref, n_modes_calc)
    #
    #     mode_names = [f'Mode {i+1}' for i in range(n_modes)]
    #
    #     fig_lambda = plot_lambda_comparison(
    #         freq,
    #         gamma_calc[:, :n_modes],
    #         lambda_ref[:, :n_modes],
    #         mode_names, save_path=save_path)
    #     figs.append(fig_lambda)

    # =====================================================================
    # 6b. Ti 变换矩阵对比
    # =====================================================================
    Ti_pscad = pscad_results.get('Ti_pscad')
    Ti_calc_full = calc_results.get('T_I')  # (K, n, n) 全频率
    T_ref = calc_results.get('T_ref')       # (n, n) 参考频率

    if Ti_pscad is not None:
        # 确定要用的计算 Ti
        if Ti_calc_full is not None:
            Ti_for_plot = Ti_calc_full
        elif T_ref is not None:
            # 如果只有参考频率 Ti，扩展为全频率 (频率无关)
            Ti_for_plot = np.tile(T_ref[np.newaxis, :, :], (len(freq), 1, 1))
            print("  ⚠ Ti 计算值仅有参考频率 T_ref，扩展为频率无关矩阵进行对比")
        else:
            Ti_for_plot = None

        if Ti_for_plot is not None:
            n_calc = Ti_for_plot.shape[1]
            n_ref = Ti_pscad.shape[1]
            n_ti = min(n_calc, n_ref)

            if n_calc != n_ref:
                print(f"\n  ⚠ Ti 维度不匹配: 自研 {n_calc}×{n_calc}, "
                      f"PSCAD {n_ref}×{n_ref}")
                print(f"  → 对比前 {n_ti}×{n_ti} 子矩阵")

            ti_names = names[:n_ti] if len(names) >= n_ti else \
                [f'C{i+1}' for i in range(n_ti)]

            # 元素级幅值/相角对比图
            fig_ti_mag, fig_ti_phase = plot_Ti_matrix_elements(
                freq,
                Ti_for_plot[:, :n_ti, :n_ti],
                Ti_pscad[:, :n_ti, :n_ti],
                ti_names, save_path=save_path)
            figs.extend([fig_ti_mag, fig_ti_phase])

            # 列向量相关性/误差对比图
            ti_comparison = compute_Ti_comparison(
                Ti_for_plot, Ti_pscad, freq, ti_names, verbose=False)
            fig_ti_corr = plot_Ti_column_correlation(
                freq,
                ti_comparison['mode_correlations'],
                ti_comparison['column_errors'],
                save_path=save_path)
            figs.append(fig_ti_corr)

    # =====================================================================
    # 7. PSCAD 内部拟合对比 (Yc calc vs fitted)
    # =====================================================================
    if pscad_results.get('Yc_has_fitted') and pscad_results.get('Yc_fitted') is not None:
        Yc_pscad_calc = pscad_results['Yc_pscad']
        Yc_pscad_fit = pscad_results['Yc_fitted']
        n = Yc_pscad_calc.shape[1]
        n_show = min(n, 4)

        fig_fit, axes_fit = plt.subplots(2, n_show, figsize=(4.5 * n_show, 7), sharex=True)
        if n_show == 1:
            axes_fit = axes_fit.reshape(-1, 1)
        fig_fit.suptitle('PSCAD Yc: Calculated vs Fitted (PSCAD internal)', fontsize=13)

        for i in range(n_show):
            # 先画拟合 (粗虚线), 再画计算 (细实线) → 虚线间隙可见
            axes_fit[0, i].semilogx(freq, np.abs(Yc_pscad_fit[:, i, i]) * 1e3,
                                     'r--', lw=2.5, label='PSCAD Fit')
            axes_fit[0, i].semilogx(freq, np.abs(Yc_pscad_calc[:, i, i]) * 1e3,
                                     'b-', lw=1.2, label='PSCAD Calc')
            axes_fit[0, i].set_ylabel(f'|Yc{i+1}{i+1}| [mS]')
            axes_fit[0, i].set_title(f'Yc{i+1}{i+1}')
            axes_fit[0, i].legend(fontsize=7)
            axes_fit[0, i].grid(True, which='both', ls=':', alpha=0.5)

            axes_fit[1, i].semilogx(freq, np.angle(Yc_pscad_fit[:, i, i], deg=True),
                                     'r--', lw=2.5, label='PSCAD Fit')
            axes_fit[1, i].semilogx(freq, np.angle(Yc_pscad_calc[:, i, i], deg=True),
                                     'b-', lw=1.2, label='PSCAD Calc')
            axes_fit[1, i].set_ylabel(f'∠Yc{i+1}{i+1} [°]')
            axes_fit[1, i].set_xlabel('Frequency [Hz]')
            axes_fit[1, i].legend(fontsize=7)
            axes_fit[1, i].grid(True, which='both', ls=':', alpha=0.5)

        fig_fit.tight_layout()
        figs.append(fig_fit)

        if save_path:
            p = save_path.replace('.png', '_Yc_PSCAD_fitting.png')
            fig_fit.savefig(p, dpi=150, bbox_inches='tight')
            print(f"  已保存: {p}")

    # =====================================================================
    # 8. PSCAD 内部拟合对比 (H mode calc vs fitted)
    # =====================================================================
    if (pscad_results.get('H_mode_has_fitted') and
        pscad_results.get('H_mode_mag_fitted') is not None):

        h_mag = pscad_results['H_mode_pscad_mag']
        h_phase = pscad_results['H_mode_pscad_phase']
        h_mag_fit = pscad_results['H_mode_mag_fitted']
        h_phase_fit = pscad_results['H_mode_phase_fitted']

        if h_mag.ndim == 2:
            n_modes = h_mag.shape[1]
        else:
            n_modes = 1

        fig_hfit, axes_hfit = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig_hfit.suptitle('PSCAD H (Modal): Calculated vs Fitted', fontsize=13)

        # 定义颜色循环 (Calc 和 Fit 使用相同颜色, 靠线型区分)
        mode_colors = plt.cm.tab10(np.linspace(0, 1, max(n_modes, 1)))

        # ---- 幅值: 先画 Fit (粗虚线), 再画 Calc (细实线) ----
        if h_mag_fit is not None:
            for m in range(n_modes):
                if h_mag_fit.ndim == 2:
                    axes_hfit[0].semilogx(freq, h_mag_fit[:, m], '--',
                                           color=mode_colors[m], lw=2.5,
                                           label=f'Mode {m+1} Fit')
                else:
                    axes_hfit[0].semilogx(freq, h_mag_fit, '--',
                                           color=mode_colors[0], lw=2.5, label='Fit')
        for m in range(n_modes):
            if h_mag.ndim == 2:
                axes_hfit[0].semilogx(freq, h_mag[:, m],
                                       color=mode_colors[m], lw=1.2,
                                       label=f'Mode {m+1} Calc')
            else:
                axes_hfit[0].semilogx(freq, h_mag,
                                       color=mode_colors[0], lw=1.2, label='Calc')
        axes_hfit[0].set_ylabel('|H|')
        axes_hfit[0].set_title('Magnitude')
        axes_hfit[0].legend(fontsize=7, ncol=3)
        axes_hfit[0].grid(True, which='both', ls=':', alpha=0.5)
        axes_hfit[0].set_ylim([0, 1.1])

        # ---- 相角: 先画 Fit (粗虚线), 再画 Calc (细实线) ----
        if h_phase_fit is not None:
            for m in range(n_modes):
                if h_phase_fit.ndim == 2:
                    axes_hfit[1].semilogx(freq, h_phase_fit[:, m], '--',
                                           color=mode_colors[m], lw=2.5,
                                           label=f'Mode {m+1} Fit')
                else:
                    axes_hfit[1].semilogx(freq, h_phase_fit, '--',
                                           color=mode_colors[0], lw=2.5, label='Fit')
        for m in range(n_modes):
            if h_phase.ndim == 2:
                axes_hfit[1].semilogx(freq, h_phase[:, m],
                                       color=mode_colors[m], lw=1.2,
                                       label=f'Mode {m+1} Calc')
            else:
                axes_hfit[1].semilogx(freq, h_phase,
                                       color=mode_colors[0], lw=1.2)
        axes_hfit[1].set_ylabel('∠H [°]')
        axes_hfit[1].set_xlabel('Frequency [Hz]')
        axes_hfit[1].legend(fontsize=7, ncol=3)
        axes_hfit[1].grid(True, which='both', ls=':', alpha=0.5)

        fig_hfit.tight_layout()
        figs.append(fig_hfit)

        if save_path:
            p = save_path.replace('.png', '_H_PSCAD_fitting.png')
            fig_hfit.savefig(p, dpi=150, bbox_inches='tight')
            print(f"  已保存: {p}")

    # =====================================================================
    # 9. VF 拟合质量图 (tr(Yc) 拟合 + H 模态拟合 + 极点分布)
    # =====================================================================
    fitting_result = calc_results.get('fitting_result')
    ulm_params_obj = calc_results.get('ulm_params')

    if fitting_result is not None and ulm_params_obj is not None:
        n_cond = calc_results['Yc_phase'].shape[1]

        fig_vf, axes_vf = plt.subplots(2, 2, figsize=(14, 10))
        fig_vf.suptitle('Vector Fitting Quality: tr(Yc) + H Modes + Poles',
                        fontsize=13, y=1.0)

        # ---- 9a: tr(Yc) 拟合 ----
        ax = axes_vf[0, 0]
        ax.loglog(freq, np.abs(ulm_params_obj.Yc_trace), 'b-', lw=2, label='Calculated')
        # 评估拟合结果
        if hasattr(fitting_result, 'poles_Yc') and fitting_result.poles_Yc is not None:
            s = 1j * 2 * np.pi * freq
            Yc_fit_trace = np.zeros(len(freq), dtype=complex)
            k0 = fitting_result.k0
            k_residues = fitting_result.k_residues
            poles = fitting_result.poles_Yc
            for k in range(len(freq)):
                Yk = k0.astype(complex).copy()
                for n_p, p in enumerate(poles):
                    Yk += k_residues[n_p] / (s[k] - p)
                Yc_fit_trace[k] = np.trace(Yk)
            ax.loglog(freq, np.abs(Yc_fit_trace), 'r--', lw=1.5, label='VF Fitted')
            rmse = fitting_result.Yc_trace_rmse * 100
            ax.set_title(f'tr(Yc) Fitting (RMSE = {rmse:.3f}%)')
        else:
            ax.set_title('tr(Yc)')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('|tr(Yc)| [S]')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls=':', alpha=0.5)

        # ---- 9b: H 模态拟合 ----
        ax = axes_vf[0, 1]
        H_modes_calc = calc_results['H_mode']
        for i in range(n_cond):
            ax.semilogx(freq, np.abs(H_modes_calc[:, i]), lw=1.5,
                        label=f'Mode {i+1}')
        ax.axhline(1.0, color='k', ls='--', alpha=0.3)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('|H|')
        ax.set_title('Modal Propagation Functions')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.set_ylim([0, 1.1])

        # ---- 9c: 极点分布 ----
        ax = axes_vf[1, 0]
        # Yc 极点
        if hasattr(fitting_result, 'poles_Yc') and fitting_result.poles_Yc is not None:
            poles_Yc = fitting_result.poles_Yc
            ax.scatter(np.real(poles_Yc), np.imag(poles_Yc) / (2 * np.pi),
                       c='steelblue', s=60, marker='x', label='tr(Yc)', zorder=3)
        # H 极点
        H_fits = getattr(fitting_result, 'H_modes_fits', []) or []
        colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(H_fits))))
        for i, fit in enumerate(H_fits):
            if fit is None or fit.poles is None or len(fit.poles) == 0:
                continue
            ax.scatter(np.real(fit.poles), np.imag(fit.poles) / (2 * np.pi),
                       c=[colors[i]], s=40, marker='o', alpha=0.7,
                       label=f'H Mode {i+1}')
        ax.set_xlabel('Real Part [Np/s]')
        ax.set_ylabel('Imaginary Part [Hz]')
        ax.set_title('Pole Distribution')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, ls=':', alpha=0.5)

        # ---- 9d: H RMSE per mode (柱状图) ----
        ax = axes_vf[1, 1]
        if fitting_result.H_modes_fits:
            mode_indices = []
            rmses = []
            n_poles_list = []
            for fit in fitting_result.H_modes_fits:
                if fit is not None:
                    mode_indices.append(fit.mode_index + 1)
                    rmses.append(fit.rmse * 100)
                    n_poles_list.append(len(fit.poles))
            if mode_indices:
                bars = ax.bar(mode_indices, rmses, color='steelblue', alpha=0.7, edgecolor='navy')
                for bar, n_p in zip(bars, n_poles_list):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{n_p}p', ha='center', va='bottom', fontsize=8)
                ax.axhline(fitting_result.Yc_trace_rmse * 100, color='r', ls='--',
                           alpha=0.7, label=f'tr(Yc) RMSE')
                ax.set_xlabel('Mode Index')
                ax.set_ylabel('RMSE [%]')
                ax.set_title('H Mode Fitting RMSE (numbers = pole count)')
                ax.legend(fontsize=7)
                ax.grid(True, axis='y', ls=':', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'No H mode fits', ha='center', va='center',
                    transform=ax.transAxes)

        fig_vf.tight_layout()
        figs.append(fig_vf)

        if save_path:
            p = save_path.replace('.png', '_VF_fitting_quality.png')
            fig_vf.savefig(p, dpi=150, bbox_inches='tight')
            print(f"  已保存: {p}")

    # =====================================================================
    # 10. VF 自研 Yc 计算值 vs VF 拟合值 (类似 PSCAD 内部 Yc Calc vs Fit)
    # =====================================================================
    fig_vf_yc = plot_VF_Yc_calc_vs_fitted(
        freq, calc_results, names, save_path=save_path)
    if fig_vf_yc is not None:
        figs.append(fig_vf_yc)

    # =====================================================================
    # 11. VF 自研 H 模态计算值 vs VF 拟合值 (类似 PSCAD 内部 H Calc vs Fit)
    # =====================================================================
    fig_vf_h = plot_VF_H_mode_calc_vs_fitted(
        freq, calc_results, save_path=save_path)
    if fig_vf_h is not None:
        figs.append(fig_vf_h)

    return figs


# =============================================================================
# 第八部分：逐频率点打印对比表
# =============================================================================
def print_spot_comparison(freq: np.ndarray,
                           Yc_calc: np.ndarray, Yc_ref: np.ndarray,
                           H_calc: np.ndarray, H_ref: np.ndarray,
                           names: List[str],
                           spot_freqs: List[float] = None):
    """在典型频率点打印 Yc 和 H 的详细对比值"""

    if spot_freqs is None:
        spot_freqs = [0.5, 50, 500, 5000, 50000, 100000]

    n = Yc_calc.shape[1]

    print("\n" + "=" * 90)
    print(" 典型频率点 Yc / H 详细对比")
    print("=" * 90)

    for f_target in spot_freqs:
        if f_target < freq[0] or f_target > freq[-1]:
            continue
        idx = np.argmin(np.abs(freq - f_target))
        f_actual = freq[idx]

        print(f"\n  ──── @ {f_actual:.2f} Hz ────")
        print(f"  {'元素':>14s} | {'|Calc|':>14s} | {'|PSCAD|':>14s} | "
              f"{'幅值误差%':>10s} | {'∠Calc°':>10s} | {'∠PSCAD°':>10s} | {'相角误差°':>10s}")
        print("  " + "-" * 96)

        # Yc 全矩阵 (mS)
        for i in range(n):
            for j in range(n):
                yc_c = Yc_calc[idx, i, j]
                yc_r = Yc_ref[idx, i, j]
                mag_c, mag_r = np.abs(yc_c) * 1e3, np.abs(yc_r) * 1e3  # mS
                ph_c, ph_r = np.angle(yc_c, deg=True), np.angle(yc_r, deg=True)
                mag_err = abs(mag_c - mag_r) / max(mag_r, 1e-30) * 100
                ph_err = abs(ph_c - ph_r)
                ph_err = min(ph_err, 360 - ph_err)
                print(f"  Yc{i+1}{j+1:>9d} | {mag_c:>14.6f} | {mag_r:>14.6f} | "
                      f"{mag_err:>10.4f} | {ph_c:>10.4f} | {ph_r:>10.4f} | {ph_err:>10.4f}")

        print()
        # H 全矩阵
        for i in range(n):
            for j in range(n):
                h_c = H_calc[idx, i, j]
                h_r = H_ref[idx, i, j]
                mag_c, mag_r = np.abs(h_c), np.abs(h_r)
                ph_c, ph_r = np.angle(h_c, deg=True), np.angle(h_r, deg=True)
                mag_err = abs(mag_c - mag_r) / max(mag_r, 1e-30) * 100
                ph_err = abs(ph_c - ph_r)
                ph_err = min(ph_err, 360 - ph_err)
                print(f"  Hp{i+1}{j+1:>9d} | {mag_c:>14.6f} | {mag_r:>14.6f} | "
                      f"{mag_err:>10.4f} | {ph_c:>10.4f} | {ph_r:>10.4f} | {ph_err:>10.4f}")


# =============================================================================
# fitULM 文件导出 (模仿 ulm_ohl_calculation_deri_semlyen.py)
# =============================================================================
def export_fitulm_file(
        modules: Dict,
        fitting_result,
        fitulm_filename: str = "ohl_model.fitULM",
        fitulm_precision: int = 16,
        verbose: bool = True
) -> Optional[str]:
    """
    导出 fitULM 格式文件

    Parameters
    ----------
    modules : Dict
        已导入的模块 (需包含 'vf')
    fitting_result : ULMFittingResult
        Vector Fitting 拟合结果对象
    fitulm_filename : str
        输出文件名（相对于 TEST 目录，或绝对路径）
    fitulm_precision : int
        浮点数输出精度（有效位数）
    verbose : bool
        是否打印详细信息

    Returns
    -------
    fitulm_path : str or None
        成功时返回导出路径，失败返回 None
    """
    vf = modules.get('vf')
    if vf is None or not hasattr(vf, 'write_fitULM'):
        print("\n✗ 无法导出 fitULM: 模块不支持 write_fitULM 函数")
        return None

    if fitting_result is None:
        print("\n✗ 无法导出 fitULM: fitting_result 为 None")
        return None

    # 确定输出路径：放在 TEST 目录下
    if os.path.isabs(fitulm_filename):
        fitulm_path = fitulm_filename
    else:
        test_dir = os.path.join(str(PROJECT_DIR), 'TEST')
        os.makedirs(test_dir, exist_ok=True)
        fitulm_path = os.path.join(test_dir, fitulm_filename)

    if verbose:
        print("\n" + "=" * 60)
        print(" 导出 fitULM 文件")
        print("=" * 60)
        print(f"  输出路径: {fitulm_path}")
        print(f"  精度: {fitulm_precision} 位")

    try:
        vf.write_fitULM(
            result=fitting_result,
            filepath=fitulm_path,
            precision=fitulm_precision,
            verbose=verbose
        )

        # 可选：验证导出文件
        if hasattr(vf, 'verify_fitULM_file'):
            vf.verify_fitULM_file(fitulm_path, verbose=verbose)

        if verbose:
            print(f"\n  ✓ fitULM 文件已成功导出: {fitulm_path}")

        return fitulm_path

    except Exception as e:
        print(f"\n✗ fitULM 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# 主函数
# =============================================================================
def main(pscad_path: str = None,
         line_length: float = 900.0,
         verbose: bool = True,
         save_plots: bool = True,
         export_fitulm: bool = True,
         fitulm_filename: str = "ohl_model.fitULM",
         fitulm_precision: int = 16):
    """
    主流程

    Parameters
    ----------
    pscad_path : str
        PSCAD 输出文件路径前缀
    line_length : float
        线路长度 [m]，用于计算 H = exp(-γL)
    verbose : bool
        是否输出详细信息
    save_plots : bool
        是否保存图表
    export_fitulm : bool
        是否导出 fitULM 文件
    fitulm_filename : str
        fitULM 输出文件名（默认保存在 TEST 目录下）
    fitulm_precision : int
        fitULM 浮点数精度（默认 16 位）
    """
    print("\n" + "=" * 70)
    print(" 特性导纳 Yc / 传播函数 H 对比: Vector Fitting vs PSCAD")
    print("=" * 70)
    print(f"  线路长度: {line_length/1000:.1f} km")

    # ---- 导入模块 ----
    modules = import_all_modules()
    if modules['zy'] is None:
        print("✗ 缺少 Z/Y 计算模块，中止")
        return None
    if modules['PSCADFileReader'] is None:
        print("✗ 缺少 PSCAD 读取器，中止")
        return None
    if modules.get('vf') is None:
        print("✗ 缺少 Vector Fitting 模块 (vector_fitting_v47_fixed)，中止")
        return None

    # ---- 确定 PSCAD 路径 ----
    if pscad_path is None:
        candidates = [
            './TLine_2/TLine_2',
            './TLine_2',
\
        ]
        for c in candidates:
            test = c.rstrip('/') + '_zm.out'
            if os.path.exists(test):
                pscad_path = c
                break
            if os.path.isdir(c):
                for f in os.listdir(c):
                    if f.endswith('_zm.out'):
                        pscad_path = c
                        break
            if pscad_path:
                break

        if pscad_path is None:
            print("✗ 未找到 PSCAD 输出文件，请指定路径")
            print("  用法: python compare_yc_h_with_pscad.py <PSCAD文件前缀> [线路长度m]")
            return None

    print(f"  PSCAD 数据路径: {pscad_path}")

    # ---- 步骤 1: 读取 PSCAD 所有数据 ----
    pscad_results = load_Yc_H_pscad(modules, pscad_path, verbose)
    freq_pscad = pscad_results['freq_pscad']

    # ---- 步骤 2: 用自研代码计算 Z/Y ----
    pscad_config = PSCADLineConfig()
    print(f"\n  线路配置: {pscad_config.line_name}")
    print(f"  导体总数: {pscad_config.n_conductors} "
          f"({pscad_config.n_phases} 相导线 + {pscad_config.n_ground_wires} 地线)")
    print(f"  Kron 缩减: {'启用 (消去地线)' if pscad_config.kron_reduction else '不缩减 (保留全部导体)'}")

    Z_calc, Y_calc, line = compute_ZY_self(modules, pscad_config, freq_pscad, verbose)

    # ---- 步骤 3: 通过 Vector Fitting 模块计算 Yc 和 H ----
    calc_results = compute_Yc_H_via_VF(
        modules, freq_pscad, Z_calc, Y_calc, line_length,
        vf_config=None, use_freq_dependent='auto', verbose=verbose)

    # 将自研 Z/Y 也加入 calc_results，以便后续统一对比和绘图
    calc_results['Z_calc'] = Z_calc
    calc_results['Y_calc'] = Y_calc

    names = line.names
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
        Yc_calc_cmp = Yc_calc
        Yc_pscad_cmp = Yc_pscad
        names_cmp = names

    if H_phase_pscad is not None and H_phase_calc.shape[1] != H_phase_pscad.shape[1]:
        n_min = min(H_phase_calc.shape[1], H_phase_pscad.shape[1])
        H_phase_calc_cmp = H_phase_calc[:, :n_min, :n_min]
        H_phase_pscad_cmp = H_phase_pscad[:, :n_min, :n_min]
    else:
        H_phase_calc_cmp = H_phase_calc
        H_phase_pscad_cmp = H_phase_pscad

    # ---- Z/Y 维度对齐 ----
    Z_pscad = pscad_results.get('Z_pscad')
    Y_pscad = pscad_results.get('Y_pscad')

    if Z_pscad is not None and Z_calc.shape[1] != Z_pscad.shape[1]:
        n_min = min(Z_calc.shape[1], Z_pscad.shape[1])
        print(f"\n  ⚠ Z 维度不匹配: 自研 {Z_calc.shape[1]}, PSCAD {Z_pscad.shape[1]}")
        Z_calc_cmp = Z_calc[:, :n_min, :n_min]
        Z_pscad_cmp = Z_pscad[:, :n_min, :n_min]
        print(f"  → 截取为 {n_min}×{n_min} 进行对比")
    else:
        Z_calc_cmp = Z_calc
        Z_pscad_cmp = Z_pscad

    if Y_pscad is not None and Y_calc.shape[1] != Y_pscad.shape[1]:
        n_min = min(Y_calc.shape[1], Y_pscad.shape[1])
        print(f"\n  ⚠ Y 维度不匹配: 自研 {Y_calc.shape[1]}, PSCAD {Y_pscad.shape[1]}")
        Y_calc_cmp = Y_calc[:, :n_min, :n_min]
        Y_pscad_cmp = Y_pscad[:, :n_min, :n_min]
        print(f"  → 截取为 {n_min}×{n_min} 进行对比")
    else:
        Y_calc_cmp = Y_calc
        Y_pscad_cmp = Y_pscad

    # ---- 步骤 5: 误差统计 ----
    print("\n" + "=" * 65)
    print(" 误差统计")
    print("=" * 65)

    Yc_errors = None
    H_phase_errors = None
    H_mode_errors = None
    Z_errors = None
    Y_errors = None

    # Z/Y 矩阵误差
    if Z_pscad_cmp is not None:
        Z_errors = compute_matrix_error(
            Z_calc_cmp, Z_pscad_cmp, names_cmp, label='Z', verbose=verbose)

    if Y_pscad_cmp is not None:
        Y_errors = compute_matrix_error(
            Y_calc_cmp, Y_pscad_cmp, names_cmp, label='Y', verbose=verbose)

    if Yc_pscad_cmp is not None:
        Yc_errors = compute_matrix_error(
            Yc_calc_cmp, Yc_pscad_cmp, names_cmp, label='Yc', verbose=verbose)

    if H_phase_pscad_cmp is not None:
        H_phase_errors = compute_matrix_error(
            H_phase_calc_cmp, H_phase_pscad_cmp, names_cmp, label='H_phase', verbose=verbose)

    # H 模态域误差
    H_mode_pscad = pscad_results.get('H_mode_pscad')
    if H_mode_pscad is not None:
        H_mode_calc = calc_results['H_mode']
        if H_mode_pscad.ndim == 3:
            n_modes_ref = H_mode_pscad.shape[1]
            H_mode_ref_vec = np.zeros((len(freq_pscad), n_modes_ref), dtype=complex)
            for m in range(n_modes_ref):
                H_mode_ref_vec[:, m] = H_mode_pscad[:, m, m]
            H_mode_pscad_vec = H_mode_ref_vec
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
    Ti_errors = None

    if Ti_pscad is not None:
        if Ti_calc_full is not None:
            Ti_for_cmp = Ti_calc_full
        elif T_ref_mat is not None:
            Ti_for_cmp = np.tile(T_ref_mat[np.newaxis, :, :], (len(freq_pscad), 1, 1))
        else:
            Ti_for_cmp = None

        if Ti_for_cmp is not None:
            ti_names = names_cmp if len(names_cmp) >= min(Ti_for_cmp.shape[1], Ti_pscad.shape[1]) \
                else [f'C{i+1}' for i in range(min(Ti_for_cmp.shape[1], Ti_pscad.shape[1]))]
            Ti_errors = compute_Ti_comparison(
                Ti_for_cmp, Ti_pscad, freq_pscad, ti_names, verbose=verbose)

    # ---- 步骤 6: 典型频率点对比 ----
    if Z_pscad_cmp is not None and Y_pscad_cmp is not None:
        print_ZY_spot_comparison(
            freq_pscad, Z_calc_cmp, Z_pscad_cmp,
            Y_calc_cmp, Y_pscad_cmp, names_cmp)

    if Yc_pscad_cmp is not None and H_phase_pscad_cmp is not None:
        print_spot_comparison(
            freq_pscad, Yc_calc_cmp, Yc_pscad_cmp,
            H_phase_calc_cmp, H_phase_pscad_cmp, names_cmp)

    # # ---- 步骤 7: 绘图 ----
    # if save_plots:
    #     print("\n" + "=" * 65)
    #     print(" 生成对比图表")
    #     print("=" * 65)
    #
    #     # 为绘图准备 pscad_results 中的截取版数据
    #     pscad_for_plot = dict(pscad_results)
    #     pscad_for_plot['Yc_pscad'] = Yc_pscad_cmp
    #     pscad_for_plot['H_phase_pscad'] = H_phase_pscad_cmp
    #     pscad_for_plot['Z_pscad'] = Z_pscad_cmp
    #     pscad_for_plot['Y_pscad'] = Y_pscad_cmp
    #
    #     # 将 Z/Y 也写入 calc_results (截取后版本)
    #     calc_results_for_plot = dict(calc_results)
    #     calc_results_for_plot['Z_calc'] = Z_calc_cmp
    #     calc_results_for_plot['Y_calc'] = Y_calc_cmp
    #
    #     # 创建 TEST 输出目录
    #     test_dir = os.path.join(str(PROJECT_DIR), 'TEST')
    #     os.makedirs(test_dir, exist_ok=True)
    #     save_base = os.path.join(test_dir, 'compare_yc_h_results.png')
    #     figs = plot_all_comparisons(
    #         freq_pscad, calc_results_for_plot, pscad_for_plot,
    #         names_cmp, save_path=save_base)

    # ---- 步骤 8: 导出 fitULM 文件 ----
    fitulm_path = None
    if export_fitulm:
        fitting_result_obj = calc_results.get('fitting_result')
        fitulm_path = export_fitulm_file(
            modules=modules,
            fitting_result=fitting_result_obj,
            fitulm_filename=fitulm_filename,
            fitulm_precision=fitulm_precision,
            verbose=verbose
        )

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print(" 对比总结")
    print("=" * 70)

    # VF 拟合质量
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

    # vs PSCAD 误差
    print(f"\n  [与 PSCAD 的对比误差]")

    if Z_errors:
        print(f"  Z 矩阵最大幅值误差:  {Z_errors['_global']['max_mag_err']:.4f} %")
        print(f"  Z 矩阵平均幅值误差:  {Z_errors['_global']['mean_mag_err']:.4f} %")
        print(f"  Z 矩阵最大相角误差:  {Z_errors['_global']['max_phase_err']:.4f} °")

    if Y_errors:
        print(f"  Y 矩阵最大幅值误差:  {Y_errors['_global']['max_mag_err']:.4f} %")
        print(f"  Y 矩阵平均幅值误差:  {Y_errors['_global']['mean_mag_err']:.4f} %")
        print(f"  Y 矩阵最大相角误差:  {Y_errors['_global']['max_phase_err']:.4f} °")

    if Yc_errors:
        print(f"  Yc 矩阵最大幅值误差: {Yc_errors['_global']['max_mag_err']:.4f} %")
        print(f"  Yc 矩阵平均幅值误差: {Yc_errors['_global']['mean_mag_err']:.4f} %")
        print(f"  Yc 矩阵最大相角误差: {Yc_errors['_global']['max_phase_err']:.4f} °")

    if H_phase_errors:
        print(f"  H(相域) 最大幅值误差: {H_phase_errors['_global']['max_mag_err']:.4f} %")
        print(f"  H(相域) 平均幅值误差: {H_phase_errors['_global']['mean_mag_err']:.4f} %")
        print(f"  H(相域) 最大相角误差: {H_phase_errors['_global']['max_phase_err']:.4f} °")

    if H_mode_errors:
        print(f"  H(模态) 最大幅值误差: {H_mode_errors['_global']['max_mag_err']:.4f} %")
        print(f"  H(模态) 平均幅值误差: {H_mode_errors['_global']['mean_mag_err']:.4f} %")
        print(f"  H(模态) 最大相角误差: {H_mode_errors['_global']['max_phase_err']:.4f} °")

    if Ti_errors:
        s = Ti_errors['summary']
        print(f"  Ti 变换矩阵 (n={s['n_compared']}):")
        print(f"    平均列相关系数: {s['mean_correlation']:.6f}")
        print(f"    最小列相关系数: {s['min_correlation']:.6f}")
        print(f"    最大列误差:     {s['max_column_error']:.6f}")
        if s['n_calc'] != s['n_ref']:
            print(f"    ⚠ 维度不匹配 (自研 {s['n_calc']}×{s['n_calc']},"
                  f" PSCAD {s['n_ref']}×{s['n_ref']})")

    # 综合判断
    max_errs = []
    if Z_errors:
        max_errs.append(('Z', Z_errors['_global']['max_mag_err']))
    if Y_errors:
        max_errs.append(('Y', Y_errors['_global']['max_mag_err']))
    if Yc_errors:
        max_errs.append(('Yc', Yc_errors['_global']['max_mag_err']))
    if H_phase_errors:
        max_errs.append(('H_phase', H_phase_errors['_global']['max_mag_err']))

    if max_errs:
        overall_max = max(e[1] for e in max_errs)
        worst_item = max(max_errs, key=lambda x: x[1])
        if overall_max < 1.0:
            print(f"\n  ✓ 所有计算结果与 PSCAD 吻合良好 (最大误差 < 1%)")
        elif overall_max < 5.0:
            print(f"\n  ⚠ 存在一定差异 (最大误差 {worst_item[1]:.2f}% 在 {worst_item[0]})，可能原因:")
            if worst_item[0] in ('Z', 'Y'):
                print("    - 接地返回模型差异 (Deri-Semlyen vs PSCAD 内部模型)")
                print("    - 导体参数输入差异 (分裂导线等效半径、弧垂修正)")
                print("    - 土壤电阻率/介电常数参数差异")
            else:
                print("    - Z/Y 矩阵本身的微小差异在 Yc/H 中被放大")
                print("    - Newton-Raphson 特征求解器 vs PSCAD 内部求解器差异")
                print("    - 矩阵开方方法差异 (Schur vs 逐元素)")
                print("    - 模态排序或符号约定差异")
        else:
            print(f"\n  ✗ 差异较大 (最大误差 {worst_item[1]:.2f}% 在 {worst_item[0]})，建议排查:")
            if worst_item[0] in ('Z', 'Y'):
                print("    - 导线几何参数 (位置、半径、分裂参数) 是否与 PSCAD 一致")
                print("    - 土壤参数 (电阻率、相对介电常数) 是否与 PSCAD 一致")
                print("    - 接地返回计算方法是否匹配 (Deri-Semlyen / Carson)")
            else:
                print("    - 首先确认 Z/Y 矩阵对比是否通过")
                print("    - 线路长度 L 是否与 PSCAD 一致")
                print("    - 模态变换矩阵 Ti 的符号/归一化约定")
            print("    - Kron 缩减设置是否与 PSCAD 一致 "
                  f"(当前: {'启用' if pscad_config.kron_reduction else '不缩减'})")
    else:
        print("\n  ⚠ 无有效 PSCAD 参考数据 (Yc/H 文件缺失)")

    # 输出文件汇总
    print(f"\n  [输出文件]")
    if save_plots:
        print(f"  图表目录: {os.path.join(str(PROJECT_DIR), 'TEST')}")
    if fitulm_path:
        print(f"  fitULM 文件: {fitulm_path}")
    elif export_fitulm:
        print(f"  fitULM 导出: 失败")



    return {
        'freq': freq_pscad,
        'calc_results': calc_results,
        'pscad_results': pscad_results,
        'ulm_params': calc_results.get('ulm_params'),
        'fitting_result': fitting_result,
        'Z_errors': Z_errors,
        'Y_errors': Y_errors,
        'Yc_errors': Yc_errors,
        'H_phase_errors': H_phase_errors,
        'H_mode_errors': H_mode_errors,
        'Ti_errors': Ti_errors,
        'names': names_cmp,
        'Z_calc': Z_calc,
        'Y_calc': Y_calc,
        'Z_pscad': Z_pscad_cmp,
        'Y_pscad': Y_pscad_cmp,
        'line': line,
        'fitulm_path': fitulm_path,
    }


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    pscad_path = sys.argv[1] if len(sys.argv) > 1 else None
    line_length = float(sys.argv[2]) if len(sys.argv) > 2 else 900

    # 可选参数：fitULM 导出文件名 (第 3 个命令行参数)
    fitulm_filename = sys.argv[3] if len(sys.argv) > 3 else "ohl_model.fitULM"

    results = main(pscad_path=pscad_path,
                   line_length=line_length,
                   verbose=True,
                   save_plots=True,
                   export_fitulm=True,
                   fitulm_filename=fitulm_filename,
                   fitulm_precision=16)
