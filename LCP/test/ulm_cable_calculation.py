#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULM 电缆模型计算脚本 (简化版)
==============================
直接计算双回铠装电缆的 ULM 参数并输出结果

依赖模块:
- cable_model_0110.py    : 电缆阻抗/导纳计算
- vector_fitting.py      : ULM 拟合模块 (V4.5+ 含 fitULM 导出)
- vf_core.py             : Vector Fitting 核心

使用方法:
    python ulm_cable_calculation.py

作者: Claude
版本: 1.1 (添加 fitULM 导出功能)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from datetime import datetime

# =============================================================================
# 路径配置 (根据实际情况修改)
# =============================================================================
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent   # 脚本所在目录
sys.path.insert(0, str(PROJECT_DIR))            # 让同目录模块可 import


# =============================================================================
# 常量
# =============================================================================
C_LIGHT = 299792458.0  # 光速 m/s


# =============================================================================
# 计算参数配置
# =============================================================================
@dataclass
class CalculationConfig:
    """计算参数配置"""
    n_freq: int = 201           # 频率点数
    freq_min: float = 0.1       # 最小频率 Hz
    freq_max: float = 1e6       # 最大频率 Hz
    line_length: float = 100000  # 线路长度 m
    soil_rho: float = 100.0     # 土壤电阻率 Ω·m

    # Vector Fitting 参数
    Yc_poles_min: int = 12    # tr(Yc) 最小极点数
    Yc_poles_max: int = 20     # tr(Yc) 最大极点数
    Yc_target_error: float = 0.002  # tr(Yc) 目标误差
    H_poles_min: int = 12       # H 模态最小极点数
    H_poles_max: int = 20      # H 模态最大极点数
    H_target_error: float = 0.002   # H 模态目标误差
    
    # fitULM 导出配置
    export_fitulm: bool = True              # 是否导出 fitULM 文件
    fitulm_filename: str = "../emtp_simulator_v3_5/cable_model_100km.pch"  # fitULM 文件名
    fitulm_precision: int = 16              # 浮点精度


# =============================================================================
# 模块导入
# =============================================================================
def import_modules() -> Dict:
    """导入所需模块"""
    modules = {}

    # 1. 电缆模型
    try:
        import cable_model_0110 as cable_model
        modules['cable_model'] = cable_model
        print("✓ cable_model_0110 导入成功")
    except ImportError as e:
        print(f"✗ cable_model_0110 导入失败: {e}")
        modules['cable_model'] = None

    # 2. Vector Fitting / ULM
    try:
        import vector_fitting_v411_independent as vf
        modules['vf'] = vf
        print("✓ vector_fitting 导入成功")
        
        # 检查是否包含 fitULM 导出功能
        if hasattr(vf, 'write_fitULM'):
            print("  ✓ fitULM 导出功能可用")
        else:
            print("  ⚠ fitULM 导出功能不可用 (需要 V4.5+)")
            
    except ImportError as e:
        print(f"✗ vector_fitting 导入失败: {e}")
        modules['vf'] = None

    # 3. VF Core (可选，用于评估拟合)
    try:
        import vf_core
        modules['vf_core'] = vf_core
        print("✓ vf_core 导入成功")
    except ImportError:
        modules['vf_core'] = None

    # 4. 诊断绘图模块 (可选)
    try:
        from vf_diagnostic_plots import plot_vf_diagnostics, export_vf_data_to_txt
        modules['plot_vf_diagnostics'] = plot_vf_diagnostics
        modules['export_vf_data_to_txt'] = export_vf_data_to_txt
        print("✓ vf_diagnostic_plots 导入成功")
    except ImportError:
        modules['plot_vf_diagnostics'] = None
        modules['export_vf_data_to_txt'] = None

    return modules


# =============================================================================
# 电缆几何定义
# =============================================================================
def create_cable_geometries(cable_model):
    """
    创建双回铠装电缆几何配置

    电缆结构 (由内到外):
    1. 芯线 (Core)        - 铜导体
    2. 主绝缘 (Insulation) - XLPE
    3. 金属护套 (Sheath)   - 铅或铝
    4. 护套绝缘            - PE
    5. 铠装层 (Armor)      - 钢丝或钢带
    6. 外护套 (Jacket)     - PE

    Returns
    -------
    cable1, cable2 : ArmoredCableGeometry
        两根电缆的几何参数
    """

    # 电缆1:
    cable1 = cable_model.ArmoredCableGeometry(
        # 芯线
        core_radius=0.0286,           # 芯线半径 28.6mm
        core_rho=3.6e-8,              # 铜电阻率 Ω·m
        core_mu_r=1.0,                # 相对磁导率
        # 主绝缘
        insulation_radius=0.061,      # 绝缘外半径 61mm
        insulation_epsilon_r=2.5,     # XLPE 相对介电常数
        insulation_tan_delta=0.0001,  # 损耗角正切
        insulation_mu_r=1.0,
        # 金属护套
        sheath_inner_radius=0.061,    # 护套内半径
        sheath_outer_radius=0.0677,   # 护套外半径 67.7mm
        sheath_rho=2.14e-7,           # 铅电阻率 Ω·m
        sheath_mu_r=1.0,
        # 护套绝缘
        sheath_insulation_radius=0.0803,  # 护套绝缘外半径 80.3mm
        sheath_insulation_epsilon_r=2.5,
        sheath_insulation_mu_r=1.0,
        # 铠装层
        armor_inner_radius=0.0803,
        armor_outer_radius=0.0868,    # 铠装外半径 86.8mm
        armor_rho=1.38e-7,            # 钢电阻率 Ω·m
        armor_mu_r=400.0,               # 铠装
        # 外护套
        jacket_radius=0.0908,         # 外护套半径 90.8mm
        jacket_epsilon_r=2.5,
        jacket_tan_delta=0.0001,
        jacket_mu_r=1.0,
        # 埋设参数
        burial_depth=2.0,             # 埋深 2m
        horizontal_pos=-6.0           # 水平位置 -6m
    )

    # 电缆2: 磁性钢铠装 (μr=400)
    cable2 = cable_model.ArmoredCableGeometry(
        core_radius=0.0286,
        core_rho=3.6e-8,
        core_mu_r=1.0,
        insulation_radius=0.061,
        insulation_epsilon_r=2.5,
        insulation_tan_delta=0.0001,
        insulation_mu_r=1.0,
        sheath_inner_radius=0.061,
        sheath_outer_radius=0.0677,
        sheath_rho=2.14e-7,
        sheath_mu_r=1.0,
        sheath_insulation_radius=0.0803,
        sheath_insulation_epsilon_r=2.5,
        sheath_insulation_mu_r=1.0,
        armor_inner_radius=0.0803,
        armor_outer_radius=0.0868,
        armor_rho=1.38e-7,
        armor_mu_r=400.0,             # 磁性钢铠装 μr=400
        jacket_radius=0.0908,
        jacket_epsilon_r=2.5,
        jacket_tan_delta=0.0001,
        jacket_mu_r=1.0,
        burial_depth=2.0,
        horizontal_pos=6.0            # 水平位置 +6m
    )

    return cable1, cable2


# =============================================================================
# 步骤 1: Z/Y 矩阵计算
# =============================================================================
def compute_ZY_matrices(modules: Dict, config: CalculationConfig,
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算电缆的阻抗矩阵 Z 和导纳矩阵 Ye

    对于 N 根三层铠装电缆，每根有 3 个导体 (芯、护套、铠装)，
    总矩阵维度为 3N × 3N。

    Parameters
    ----------
    modules : Dict
        已导入的模块字典
    config : CalculationConfig
        计算配置
    verbose : bool
        是否打印详细信息

    Returns
    -------
    freq : np.ndarray
        频率向量 (Hz)
    Z_matrix : np.ndarray
        阻抗矩阵 (n_freq, 6, 6)，单位 Ω/m
    Y_matrix : np.ndarray
        导纳矩阵 (n_freq, 6, 6)，单位 S/m
    """
    cable_model = modules['cable_model']

    if verbose:
        print("\n" + "=" * 60)
        print(" 步骤 1: 计算 Z/Y 矩阵")
        print("=" * 60)

    # 生成对数分布的频率向量
    freq = cable_model.generate_frequency_vector(
        config.freq_min, config.freq_max, config.n_freq
    )
    n_freq = len(freq)

    # 土壤参数
    soil = cable_model.SoilParameters(
        rho=config.soil_rho,  # 土壤电阻率
        epsilon_r=1.0,        # 相对介电常数
        mu_r=1.0              # 相对磁导率
    )
    gamma_soil = soil.get_gamma(freq)  # 土壤传播常数

    # 创建电缆几何
    cable1, cable2 = create_cable_geometries(cable_model)
    cables = [cable1, cable2]

    if verbose:
        print(f"  频率范围: {freq[0]:.2f} Hz ~ {freq[-1]:.0e} Hz ({n_freq} 点)")
        print(f"  土壤电阻率: {soil.rho} Ω·m")
        print(f"  电缆间距: {abs(cable1.horizontal_pos - cable2.horizontal_pos):.1f} m")

    # 计算阻抗矩阵 Z (包含自阻抗和互阻抗)
    if verbose:
        print("\n  计算阻抗矩阵 Z...")
    Z_matrix = cable_model.compute_multi_cable_impedance(freq, cables, gamma_soil)

    # 计算导纳矩阵 Y (各电缆独立，无互导纳)
    if verbose:
        print("  计算导纳矩阵 Y...")
    Y_matrix = np.zeros((n_freq, 6, 6), dtype=complex)
    Y1 = cable_model.compute_armored_cable_admittance(freq, cable1)
    Y2 = cable_model.compute_armored_cable_admittance(freq, cable2)
    Y_matrix[:, 0:3, 0:3] = Y1  # 电缆1: [Core1, Sheath1, Armor1]
    Y_matrix[:, 3:6, 3:6] = Y2  # 电缆2: [Core2, Sheath2, Armor2]

    if verbose:
        print(f"\n  Z_matrix 形状: {Z_matrix.shape}")
        print(f"  Y_matrix 形状: {Y_matrix.shape}")

        # 显示 50Hz 时的典型值
        idx_50 = np.argmin(np.abs(freq - 50))
        print(f"\n  @ {freq[idx_50]:.1f} Hz 典型值:")
        print(f"    Z_core1_self  = {Z_matrix[idx_50, 0, 0]*1000:.4f} mΩ/m")
        print(f"    Z_sheath1_self= {Z_matrix[idx_50, 1, 1]*1000:.4f} mΩ/m")
        print(f"    Y_core1_self  = {np.abs(Y_matrix[idx_50, 0, 0])*1e6:.4f} µS/m")

    return freq, Z_matrix, Y_matrix


# =============================================================================
# 步骤 2: ULM 参数计算
# =============================================================================
def compute_ulm_parameters(modules: Dict, freq: np.ndarray,
                           Z_matrix: np.ndarray, Y_matrix: np.ndarray,
                           config: CalculationConfig,
                           verbose: bool = True):
    """
    使用 Vector Fitting 计算 ULM (Universal Line Model) 参数

    ULM 将频域传输线方程转换为时域状态空间形式，
    通过有理函数逼近特征导纳 Yc 和传播函数 H。

    Parameters
    ----------
    modules : Dict
        已导入的模块
    freq : np.ndarray
        频率向量
    Z_matrix, Y_matrix : np.ndarray
        阻抗和导纳矩阵
    config : CalculationConfig
        计算配置
    verbose : bool
        是否打印详细信息

    Returns
    -------
    ulm_params : object
        ULM 参数对象，包含:
        - Yc_trace: tr(Yc) 特征导纳迹
        - H_modes: 各模态传播函数
        - tau: 传播时间延迟
        - T_ref: 参考变换矩阵
    fitting_result : object
        拟合结果对象，包含极点、留数等
    """
    vf = modules['vf']

    if verbose:
        print("\n" + "=" * 60)
        print(" 步骤 2: ULM 参数计算")
        print("=" * 60)

    # 配置 Vector Fitting 参数
    vf_config = vf.IterativePoleFindingConfig(
        Ymin=config.Yc_poles_min,
        Ymax=config.Yc_poles_max,
        epsY=config.Yc_target_error,
        Hmin=config.H_poles_min,
        Hmax=config.H_poles_max,
        epsH=config.H_target_error,
        pole_step=2,
        eps_deg=10  # 进行模态合并
    )

    if verbose:
        print(f"  Vector Fitting 配置:")
        print(f"    tr(Yc): {vf_config.Ymin}-{vf_config.Ymax} 极点, 目标误差 {vf_config.epsY*100}%")
        print(f"    H 模态: {vf_config.Hmin}-{vf_config.Hmax} 极点, 目标误差 {vf_config.epsH*100}%")

    # 执行 ULM 完整拟合
    ulm_params, fitting_result = vf.ulm_complete_fitting(
        freq=freq,
        Z_matrix=Z_matrix,
        Y_matrix=Y_matrix,
        length=config.line_length,
        velocity_freq=1e5,              # 参考频率 1MHz
        config=vf_config,
        enforce_passivity_flag=False,    # 强制被动性
        verbose=verbose
    )

    return ulm_params, fitting_result


# =============================================================================
# 步骤 3: 结果输出
# =============================================================================
def print_results(freq: np.ndarray, ulm_params, fitting_result,
                  config: CalculationConfig):
    """
    打印计算结果摘要

    Parameters
    ----------
    freq : np.ndarray
        频率向量
    ulm_params : object
        ULM 参数
    fitting_result : object
        拟合结果
    config : CalculationConfig
        计算配置
    """
    print("\n" + "=" * 60)
    print(" 计算结果摘要")
    print("=" * 60)

    # 1. 传播时间 τ
    print("\n[1] 传播时间 τ (模态延迟)")
    print("-" * 40)
    tau_ms = ulm_params.tau * 1000  # 转换为 ms
    for i, tau in enumerate(tau_ms):
        # 计算等效传播速度与光速的比值
        v_ratio = config.line_length / (tau / 1000) / C_LIGHT if tau > 0 else float('inf')

        # 判断模态类型
        if v_ratio > 1.0:
            mode_type = "接地模态 (Ground)"
        elif v_ratio > 0.5:
            mode_type = "芯线模态 (Core)"
        elif v_ratio > 0.2:
            mode_type = "护套模态 (Sheath)"
        else:
            mode_type = "铠装模态 (Armor)"

        print(f"  τ{i+1}: {tau:10.6f} ms  (v/c = {v_ratio:.4f})  {mode_type}")

    if hasattr(ulm_params, 'tau_min'):
        print(f"\n  最小延迟 τ_min = {ulm_params.tau_min*1000:.6f} ms")
        print(f"  建议时间步长 Δt ≤ {ulm_params.tau_min*1000/10:.6f} ms")

    # 2. 变换矩阵 Ti
    print("\n[2] 模态变换矩阵 T_i (参考频率处)")
    print("-" * 40)
    Ti = ulm_params.T_ref
    print(f"  矩阵维度: {Ti.shape}")
    print(f"  归一化检查 (每列最大值位置):")
    for col in range(Ti.shape[1]):
        max_idx = np.argmax(np.abs(Ti[:, col]))
        max_val = Ti[max_idx, col]
        conductor = ['Core1', 'Sheath1', 'Armor1', 'Core2', 'Sheath2', 'Armor2'][max_idx]
        print(f"    列{col+1}: 行{max_idx+1} ({conductor}), 值={max_val:.4f}")

    # 3. 拟合质量
    print("\n[3] Vector Fitting 拟合质量")
    print("-" * 40)

    # tr(Yc)
    if hasattr(fitting_result, "Yc_trace_rmse") and hasattr(fitting_result, "poles_Yc"):
        n_poles = len(fitting_result.poles_Yc)
        rmse = fitting_result.Yc_trace_rmse
        status = "✓" if rmse < 0.02 else "⚠"
        print(f"  tr(Yc): {n_poles} 极点, RMSE = {rmse * 100:.3f}% {status}")

    # H modes
    if hasattr(fitting_result, "H_modes_fits") and fitting_result.H_modes_fits:
        h_info = []
        rmses = []
        total_poles = 0
        for fit in fitting_result.H_modes_fits:
            if fit is None:
                continue
            total_poles += len(fit.poles)
            rmses.append(fit.rmse)
            h_info.append(f"mode{fit.mode_index}:{len(fit.poles)}p({fit.rmse * 100:.2f}%)")
        if h_info:
            print(f"  H 模态: {', '.join(h_info)}")
            print(f"  H 总极点: {total_poles}, 平均 RMSE = {np.mean(rmses) * 100:.2f}%")

    # H matrix reconstruction
    if hasattr(fitting_result, "H_matrix_rmse"):
        print(f"  H 矩阵重构 RMSE = {fitting_result.H_matrix_rmse * 100:.3f}%")

    # 4. passivity
    print("\n[4] 被动性")
    print("-" * 40)
    status = "✓ 满足" if getattr(fitting_result, "is_passive", False) else "✗ 违反"
    print(f"  状态: {status}")

    # 5. 模态 Yc 值
    if hasattr(ulm_params, 'Yc_modal_diag'):
        print("\n[5] 模态特征导纳 Yc @ 50Hz")
        print("-" * 40)
        idx_50 = np.argmin(np.abs(freq - 50))
        Yc_50 = ulm_params.Yc_modal_diag[idx_50]
        for i, yc in enumerate(Yc_50):
            mag = np.abs(yc)
            ang = np.angle(yc, deg=True)
            print(f"  Yc{i+1}{i+1} = {mag:.4e} ∠{ang:+.2f}° S")


# =============================================================================
# 步骤 4: 绘制结果图表
# =============================================================================
def plot_results(freq: np.ndarray, Z_matrix: np.ndarray, Y_matrix: np.ndarray,
                 ulm_params, fitting_result, modules: Dict,
                 save_path: Optional[str] = None):
    """
    绘制计算结果图表

    Parameters
    ----------
    freq : np.ndarray
        频率向量
    Z_matrix, Y_matrix : np.ndarray
        阻抗和导纳矩阵
    ulm_params : object
        ULM 参数
    fitting_result : object
        拟合结果
    modules : Dict
        模块字典
    save_path : str, optional
        图片保存路径
    """
    print("\n" + "=" * 60)
    print(" 步骤 4: 生成结果图表")
    print("=" * 60)

    vf_core = modules.get('vf_core')

    def _eval_Yc_trace_from_matrix_fit(freq_hz, poles, k_residues, k0):
        """用 (poles_Yc, k_residues, k0) 计算拟合后的 tr(Yc)(f)。"""
        if poles is None or k_residues is None or k0 is None:
            return None
        s = 1j * 2 * np.pi * freq_hz
        tr = np.zeros(len(freq_hz), dtype=complex)
        k0c = k0.astype(complex)

        for k in range(len(freq_hz)):
            Yk = k0c.copy()
            for n, p in enumerate(poles):
                Yk += k_residues[n] / (s[k] - p)
            tr[k] = np.trace(Yk)
        return tr

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('ULM Cable Model Calculation Results', fontsize=14, fontweight='bold')

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # ----- 图1: 自阻抗幅值 -----
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Core1', 'Sheath1', 'Armor1', 'Core2', 'Sheath2', 'Armor2']
    for i, label in enumerate(labels):
        ax1.loglog(freq, np.abs(Z_matrix[:, i, i]) * 1000, lw=1.5, label=label)
    ax1.axvline(50, color='gray', ls=':', alpha=0.5, label='50 Hz')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('|Z| [mΩ/m]')
    ax1.set_title('Self Impedance')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, which='both', ls=':', alpha=0.5)

    # ----- 图2: 等效电容 -----
    ax2 = fig.add_subplot(gs[0, 1])
    omega = 2 * np.pi * freq
    for i in range(3):
        C_eq = np.imag(Y_matrix[:, i, i]) / omega * 1e12  # pF/m
        ax2.semilogx(freq, C_eq, lw=1.5, label=f'C{i+1}{i+1}')
    ax2.axvline(50, color='gray', ls=':', alpha=0.5)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('C [pF/m]')
    ax2.set_title('Equivalent Capacitance (Cable 1)')
    ax2.legend(fontsize=8)
    ax2.grid(True, which='both', ls=':', alpha=0.5)

    # ----- 图3: 模态 Yc -----
    ax3 = fig.add_subplot(gs[0, 2])
    if hasattr(ulm_params, 'Yc_modal_diag'):
        for i in range(min(6, ulm_params.Yc_modal_diag.shape[1])):
            ax3.semilogx(freq, np.abs(ulm_params.Yc_modal_diag[:, i]), lw=1.5, label=f'Yc{i+1}')
    ax3.axvline(50, color='gray', ls=':', alpha=0.5)
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('|Yc| [S]')
    ax3.set_title('Modal Characteristic Admittance')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, which='both', ls=':', alpha=0.5)

    # ----- 图4: 传播函数 H -----
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(min(6, ulm_params.H_modes.shape[1])):
        ax4.semilogx(freq, np.abs(ulm_params.H_modes[:, i]), lw=1.5, label=f'H{i+1}')
    ax4.axvline(50, color='gray', ls=':', alpha=0.5)
    ax4.axhline(1.0, color='k', ls='--', alpha=0.3)
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('|H|')
    ax4.set_title('Modal Propagation Functions')
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, which='both', ls=':', alpha=0.5)
    ax4.set_ylim([0, 1.1])

    # ----- 图5: tr(Yc) 拟合 -----
    ax5 = fig.add_subplot(gs[1, 1])

    poles_Yc = getattr(fitting_result, 'poles_Yc', None)
    k_residues = getattr(fitting_result, 'k_residues', None)
    k0 = getattr(fitting_result, 'k0', None)
    Yc_trace_rmse = getattr(fitting_result, 'Yc_trace_rmse', None)

    ax5.loglog(freq, np.abs(ulm_params.Yc_trace), 'b-', lw=2, label='Calculated')

    Yc_trace_fit = _eval_Yc_trace_from_matrix_fit(freq, poles_Yc, k_residues, k0)
    if Yc_trace_fit is not None:
        ax5.loglog(freq, np.abs(Yc_trace_fit), 'r--', lw=1.5, label='Fitted')
        n_poles = len(poles_Yc) if poles_Yc is not None else 0
        rmse_pct = (Yc_trace_rmse * 100) if Yc_trace_rmse is not None else float('nan')
        ax5.set_title(f'tr(Yc) Fitting (n={n_poles}, RMSE={rmse_pct:.2f}%)')
    else:
        ax5.set_title('tr(Yc)')

    ax5.set_xlabel('Frequency [Hz]')
    ax5.set_ylabel('|tr(Yc)| [S]')
    ax5.legend(fontsize=8)
    ax5.grid(True, which='both', ls=':', alpha=0.5)

    # ----- 图6: 极点分布 -----
    ax6 = fig.add_subplot(gs[1, 2])

    # tr(Yc) poles
    if poles_Yc is not None:
        ax6.scatter(np.real(poles_Yc), np.imag(poles_Yc) / (2 * np.pi),
                    c='steelblue', s=50, marker='x', label='tr(Yc)')

    # H poles（新版：H_modes_fits）
    H_fits = getattr(fitting_result, 'H_modes_fits', []) or []
    colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(H_fits))))

    for i, fit in enumerate(H_fits):
        if fit is None or getattr(fit, 'poles', None) is None:
            continue
        poles = fit.poles
        if len(poles) == 0:
            continue
        ax6.scatter(np.real(poles), np.imag(poles) / (2 * np.pi),
                    c=[colors[i]], s=30, marker='o', alpha=0.6,
                    label=f'H_group{i + 1}')

    ax6.axvline(0, color='k', ls='-', lw=0.5)
    ax6.set_xlabel('Real Part [rad/s]')
    ax6.set_ylabel('Imag Part / 2π [Hz]')
    ax6.set_title('Pole Distribution')
    ax6.legend(fontsize=7, loc='upper left')
    ax6.grid(True, ls=':', alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图表已保存: {save_path}")

    plt.show()


# =============================================================================
# 步骤 5: 导出 fitULM 文件 (新增)
# =============================================================================
def export_fitulm_file(modules: Dict, ulm_params, fitting_result,
                       config: CalculationConfig, verbose: bool = True) -> Optional[str]:
    """
    导出 fitULM 格式文件
    
    fitULM 是 ULM-ATP 手册规定的纯文本文件格式，用于在 ATP-EMTP 中
    导入线路模型参数。
    
    Parameters
    ----------
    modules : Dict
        已导入的模块字典
    ulm_params : ULMParameters
        ULM 参数对象
    fitting_result : ULMFittingResult
        拟合结果对象
    config : CalculationConfig
        计算配置
    verbose : bool
        是否打印详细信息
    
    Returns
    -------
    filepath : str or None
        导出的文件路径，失败时返回 None
    """
    if not config.export_fitulm:
        if verbose:
            print("\n  fitULM 导出已禁用 (config.export_fitulm = False)")
        return None
    
    vf = modules.get('vf')
    
    if vf is None:
        print("\n✗ 无法导出 fitULM: vector_fitting 模块未加载")
        return None
    
    # 检查是否有 write_fitULM 函数
    if not hasattr(vf, 'write_fitULM'):
        print("\n✗ 无法导出 fitULM: vector_fitting 模块版本过低 (需要 V4.5+)")
        print("  请更新 vector_fitting.py 以获得 fitULM 导出功能")
        return None
    
    if verbose:
        print("\n" + "=" * 60)
        print(" 步骤 5: 导出 fitULM 文件")
        print("=" * 60)
    
    # 构建输出文件路径
    fitulm_path = os.path.join(PROJECT_DIR, config.fitulm_filename)
    
    try:
        # 调用 write_fitULM 导出文件
        vf.write_fitULM(
            result=fitting_result,
            filepath=fitulm_path,
            precision=config.fitulm_precision,
            verbose=verbose
        )
        
        # 验证导出的文件
        if hasattr(vf, 'verify_fitULM_file'):
            if verbose:
                print("\n  验证导出的文件...")
            is_valid = vf.verify_fitULM_file(fitulm_path, verbose=verbose)
            
            if not is_valid:
                print("  ⚠ 文件验证失败，但文件已生成")
        
        return fitulm_path
        
    except Exception as e:
        print(f"\n✗ fitULM 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# 主函数
# =============================================================================
def main(verbose: bool = True, save_plot: bool = True, export_fitulm: bool = True):
    """
    主计算流程

    Parameters
    ----------
    verbose : bool
        是否打印详细信息
    save_plot : bool
        是否保存图表
    export_fitulm : bool
        是否导出 fitULM 文件

    Returns
    -------
    dict
        包含所有计算结果的字典
    """
    print("\n" + "=" * 60)
    print(" ULM 电缆模型计算")
    print("=" * 60)
    print(f" 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载模块
    modules = import_modules()

    if modules['cable_model'] is None or modules['vf'] is None:
        print("\n✗ 计算中止: 必要模块未导入")
        return None

    # 创建配置
    config = CalculationConfig()
    config.export_fitulm = export_fitulm  # 使用函数参数覆盖配置
    
    print(f"\n 计算配置:")
    print(f"   线路长度: {config.line_length/1000:.1f} km")
    print(f"   频率范围: {config.freq_min} Hz ~ {config.freq_max/1e6} MHz")
    print(f"   频率点数: {config.n_freq}")
    print(f"   导出 fitULM: {'是' if config.export_fitulm else '否'}")

    try:
        # 步骤 1: 计算 Z/Y 矩阵
        freq, Z_matrix, Y_matrix = compute_ZY_matrices(modules, config, verbose=verbose)

        # 步骤 2: 计算 ULM 参数
        ulm_params, fitting_result = compute_ulm_parameters(
            modules, freq, Z_matrix, Y_matrix, config, verbose=verbose
        )

        # 步骤 3: 输出结果
        print_results(freq, ulm_params, fitting_result, config)

        # 步骤 4: 绘制图表
        if save_plot:
            plot_path = os.path.join(PROJECT_DIR, 'ulm_results.png')
            plot_results(freq, Z_matrix, Y_matrix, ulm_params, fitting_result,
                        modules, save_path=plot_path)

        # 步骤 5: 导出 fitULM 文件 (新增)
        fitulm_path = None
        if config.export_fitulm:
            fitulm_path = export_fitulm_file(
                modules, ulm_params, fitting_result, config, verbose=verbose
            )

        # 步骤 6: 诊断图和数据导出 (可选)
        if modules.get('plot_vf_diagnostics') is not None:
            try:
                modules['plot_vf_diagnostics'](
                    freq, ulm_params, fitting_result, modules, save_dir=PROJECT_DIR
                )
            except Exception as e:
                print(f"  ⚠ 诊断图生成失败: {e}")

        if modules.get('export_vf_data_to_txt') is not None:
            try:
                modules['export_vf_data_to_txt'](
                    freq, ulm_params, fitting_result, modules, save_dir=PROJECT_DIR
                )
            except Exception as e:
                print(f"  ⚠ 数据导出失败: {e}")

        print("\n" + "=" * 60)
        print(" 计算完成")
        print("=" * 60)
        
        # 打印输出文件汇总
        print("\n 输出文件:")
        if save_plot:
            print(f"   - 结果图表: {os.path.join(PROJECT_DIR, 'ulm_results.png')}")
        if fitulm_path:
            print(f"   - fitULM 文件: {fitulm_path}")

        return {
            'freq': freq,
            'Z_matrix': Z_matrix,
            'Y_matrix': Y_matrix,
            'ulm_params': ulm_params,
            'fitting_result': fitting_result,
            'config': config,
            'fitulm_path': fitulm_path,
        }

    except Exception as e:
        print(f"\n✗ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    # 运行主程序，启用 fitULM 导出
    results = main(verbose=True, save_plot=True, export_fitulm=True)
