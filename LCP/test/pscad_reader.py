#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSCAD Line Constants Program 输出文件读取器
============================================

用于读取和分析PSCAD线路常数计算程序生成的各类输出文件。

文件命名规则：
- 第2个字母: m = magnitude幅值, p = phase相角
- 第3个字母: m = mode模态域, p = phase相域

支持的文件类型：
| 数据类型 | 幅值合文件 | 相角文件 | 数据结构 | 有拟值 |
|---------|---------|---------|---------|---------|
| 阻抗Z | _zm.out | _zp.out | 6×6矩阵 | 否 |
| 导纳Y | _ym.out | _yp.out | 6×6矩阵 | 否 |
| 特性导纳Yc | _ycmp.out | _ycpp.out | 6×6矩阵 | 是 |
| 模态传播函数Hmode | _hmm.out | _hpm.out | N-1个模态 | 是 |
| 相域传播函数Hphase | _hmp.out | _hpp.out | 6×6矩阵 | 是 |
| 传播常数Lambda | _lamdam.out | _lamdap.out | N个对角元素 | 否 |
| 变换矩阵Ti | _timp.out | _tipp.out | 6×6矩阵 | 否 |

作者: Claude
日期: 2026-01-14
版本: 2.0
"""

import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Union
import warnings


@dataclass
class PSCADMatrixData:
    """PSCAD矩阵数据容器"""
    frequency: np.ndarray  # 频率数组 (Hz)
    magnitude: np.ndarray  # 幅值数组
    phase_deg: np.ndarray  # 相角数组 (度)
    complex_data: np.ndarray  # 复数数组
    unit: str  # 单位
    description: str  # 描述
    has_fitted: bool = False  # 是否包含拟合值
    magnitude_fitted: Optional[np.ndarray] = None  # 拟合幅值
    phase_fitted: Optional[np.ndarray] = None  # 拟合相角
    complex_fitted: Optional[np.ndarray] = None  # 拟合复数值


class PSCADFileReader:
    """
    PSCAD线路常数输出文件读取器

    支持两种使用方式：
    1. 指定文件夹路径：reader = PSCADFileReader('/path/to/Cable_1/')
    2. 指定文件前缀：reader = PSCADFileReader('/path/to/Cable_1')

    使用示例
    --------
    >>> reader = PSCADFileReader('/path/to/Cable_1')
    >>> Z = reader.get_impedance()
    >>> print(Z.complex_data[0, 0, 0])  # f=0.5Hz时的Z11

    >>> # 获取特定频率的数据
    >>> Z_50hz = reader.get_impedance_at_freq(50)
    """

    def __init__(self, path: str):
        """
        初始化读取器

        Parameters
        ----------
        path : str
            可以是以下两种形式：
            - 文件夹路径: '/path/to/Cable_1/' (会自动在里面查找Cable_1_*.out文件)
            - 文件前缀: '/path/to/Cable_1' (会查找Cable_1_zm.out等文件)
        """
        self.base_path, self.file_prefix = self._resolve_path(path)
        self._cache: Dict[str, PSCADMatrixData] = {}

        # 自动检测导体数量
        self.n_conductors = self._detect_n_conductors()
        self.n_modes = self.n_conductors - 1  # 模态数 = 导体数 - 1

        # 文件配置（根据实际文件结构）
        self._file_config = {
            'impedance': {
                'mag_suffix': '_zm.out',
                'arg_suffix': '_zp.out',
                'unit': 'ohms/m',
                'description': '串联阻抗矩阵 Z',
                'shape': 'matrix',
                'has_fitted': False
            },
            'admittance': {
                'mag_suffix': '_ym.out',
                'arg_suffix': '_yp.out',
                'unit': 'mhos/m',
                'description': '并联导纳矩阵 Y',
                'shape': 'matrix',
                'has_fitted': False
            },
            'char_admittance': {
                'mag_suffix': '_ycmp.out',
                'arg_suffix': '_ycpp.out',
                'unit': 'mho',
                'description': '特性导纳矩阵 Yc',
                'shape': 'matrix',
                'has_fitted': True
            },
            'h_mode': {
                'mag_suffix': '_hmm.out',  # H Magnitude Mode
                'arg_suffix': '_hpm.out',  # H Phase Mode
                'unit': '-',
                'description': '传播函数(模态域) Hmode',
                'shape': 'vector',
                'has_fitted': True
            },
            'h_phase': {
                'mag_suffix': '_hmp.out',  # H Magnitude Phase
                'arg_suffix': '_hpp.out',  # H Phase Phase
                'unit': '-',
                'description': '传播函数(相域) Hphase',
                'shape': 'matrix',
                'has_fitted': True
            },
            'propagation': {
                'mag_suffix': '_lamdam.out',
                'arg_suffix': '_lamdap.out',
                'unit': '-',
                'description': '传播常数 λ',
                'shape': 'diagonal',
                'has_fitted': False
            },
            'transform': {
                'mag_suffix': '_timp.out',
                'arg_suffix': '_tipp.out',
                'unit': '-',
                'description': '电流变换矩阵 Ti',
                'shape': 'matrix',
                'has_fitted': False
            }
        }

    def _resolve_path(self, path: str) -> Tuple[str, str]:
        """
        解析路径，支持文件夹或文件前缀

        Returns
        -------
        base_path : str
            基础目录路径
        file_prefix : str
            文件名前缀（不含后缀）
        """
        path = path.rstrip('/').rstrip('\\')

        # 检查是否是目录
        if os.path.isdir(path):
            # 是目录，在里面查找文件
            base_path = path
            # 尝试查找_zm.out文件来确定前缀
            for f in os.listdir(path):
                if f.endswith('_zm.out'):
                    file_prefix = os.path.join(path, f[:-7])  # 去掉'_zm.out'
                    return base_path, file_prefix
            # 如果没找到，假设前缀是目录名
            dir_name = os.path.basename(path)
            file_prefix = os.path.join(path, dir_name)
            return base_path, file_prefix
        else:
            # 不是目录，作为文件前缀处理
            base_path = os.path.dirname(path) or '.'
            file_prefix = path
            return base_path, file_prefix

    def _get_file_path(self, suffix: str) -> str:
        """获取完整文件路径"""
        return self.file_prefix + suffix

    def _detect_n_conductors(self) -> int:
        """自动检测导体数量"""
        zm_file = self._get_file_path('_zm.out')
        if os.path.exists(zm_file):
            with open(zm_file, 'r') as f:
                f.readline()  # 跳过表头
                line = f.readline()
                parts = line.split()
                n_values = len(parts) - 2  # 减去频率的两列
                n_cond = int(np.sqrt(n_values))
                return n_cond
        return 6  # 默认值

    def _read_raw_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取原始数据文件

        Returns
        -------
        frequency : ndarray
            频率数组
        data : ndarray
            数据数组（不含频率列）
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件不存在: {filename}")

        raw_data = np.loadtxt(filename, skiprows=1)

        frequency = raw_data[:, 1]  # 第2列是频率(Hz)
        data = raw_data[:, 2:]  # 第3列起是数据

        return frequency, data

    def _parse_matrix_data(self, data: np.ndarray, has_fitted: bool,
                           shape: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        解析矩阵数据

        Parameters
        ----------
        data : ndarray
            原始数据，shape = (n_freq, n_values)
        has_fitted : bool
            是否包含拟合值（calc和fitted交替）
        shape : str
            数据形状: 'matrix', 'vector', 'diagonal'

        Returns
        -------
        calc_data : ndarray
            计算值
        fitted_data : ndarray or None
            拟合值（如果有）
        """
        n_freq = data.shape[0]
        n_cols = data.shape[1]

        if shape == 'matrix':
            if has_fitted:
                # 每个元素有两个值：calc, fitted
                n_elements = n_cols // 2
                n_cond = int(np.sqrt(n_elements))

                calc_data = np.zeros((n_freq, n_cond, n_cond))
                fitted_data = np.zeros((n_freq, n_cond, n_cond))

                for i in range(n_cond):
                    for j in range(n_cond):
                        idx = (i * n_cond + j) * 2
                        calc_data[:, i, j] = data[:, idx]
                        fitted_data[:, i, j] = data[:, idx + 1]
            else:
                n_elements = n_cols
                n_cond = int(np.sqrt(n_elements))
                calc_data = data.reshape(n_freq, n_cond, n_cond)
                fitted_data = None

        elif shape == 'vector':
            if has_fitted:
                n_modes = n_cols // 2
                calc_data = data[:, 0::2]  # 偶数索引列（0,2,4,...）
                fitted_data = data[:, 1::2]  # 奇数索引列（1,3,5,...）
            else:
                calc_data = data
                fitted_data = None

        elif shape == 'diagonal':
            # 对角元素向量
            calc_data = data
            fitted_data = None
        else:
            raise ValueError(f"未知的shape类型: {shape}")

        return calc_data, fitted_data

    def _load_data(self, data_type: str) -> PSCADMatrixData:
        """加载指定类型的数据"""
        if data_type in self._cache:
            return self._cache[data_type]

        config = self._file_config[data_type]

        mag_file = self._get_file_path(config['mag_suffix'])
        arg_file = self._get_file_path(config['arg_suffix'])

        # 读取幅值和相角
        freq, mag_raw = self._read_raw_data(mag_file)
        _, arg_raw = self._read_raw_data(arg_file)

        # 解析数据
        mag_calc, mag_fitted = self._parse_matrix_data(
            mag_raw, config['has_fitted'], config['shape']
        )
        arg_calc, arg_fitted = self._parse_matrix_data(
            arg_raw, config['has_fitted'], config['shape']
        )

        # 计算复数值
        arg_rad = np.deg2rad(arg_calc)
        complex_calc = mag_calc * np.exp(1j * arg_rad)

        complex_fitted = None
        if config['has_fitted'] and mag_fitted is not None:
            arg_fitted_rad = np.deg2rad(arg_fitted)
            complex_fitted = mag_fitted * np.exp(1j * arg_fitted_rad)

        result = PSCADMatrixData(
            frequency=freq,
            magnitude=mag_calc,
            phase_deg=arg_calc,
            complex_data=complex_calc,
            unit=config['unit'],
            description=config['description'],
            has_fitted=config['has_fitted'],
            magnitude_fitted=mag_fitted,
            phase_fitted=arg_fitted,
            complex_fitted=complex_fitted
        )

        self._cache[data_type] = result
        return result

    def _find_freq_index(self, freq: np.ndarray, target_freq: float) -> int:
        """找到最接近目标频率的索引"""
        idx = np.argmin(np.abs(freq - target_freq))
        actual_freq = freq[idx]
        if abs(actual_freq - target_freq) / max(target_freq, 1e-10) > 0.1:
            warnings.warn(f"请求频率 {target_freq} Hz，实际最近频率 {actual_freq:.4f} Hz")
        return idx

    # ==================== 数据获取接口 ====================

    def get_impedance(self) -> PSCADMatrixData:
        """获取阻抗矩阵 Z [ohms/m]"""
        return self._load_data('impedance')

    def get_admittance(self) -> PSCADMatrixData:
        """获取导纳矩阵 Y [mhos/m]"""
        return self._load_data('admittance')

    def get_char_admittance(self) -> PSCADMatrixData:
        """获取特性导纳矩阵 Yc [mho]"""
        return self._load_data('char_admittance')

    def get_h_mode(self) -> PSCADMatrixData:
        """获取模态域传播函数 Hmode"""
        return self._load_data('h_mode')

    def get_h_phase(self) -> PSCADMatrixData:
        """获取相域传播函数 Hphase"""
        return self._load_data('h_phase')

    def get_propagation_constant(self) -> PSCADMatrixData:
        """获取传播常数 λ (Lambda)"""
        return self._load_data('propagation')

    def get_transform_matrix(self) -> PSCADMatrixData:
        """获取电流变换矩阵 Ti"""
        return self._load_data('transform')

    # ==================== 便捷方法 ====================

    def get_impedance_at_freq(self, target_freq: float) -> np.ndarray:
        """获取指定频率的阻抗矩阵"""
        data = self.get_impedance()
        idx = self._find_freq_index(data.frequency, target_freq)
        return data.complex_data[idx]

    def get_admittance_at_freq(self, target_freq: float) -> np.ndarray:
        """获取指定频率的导纳矩阵"""
        data = self.get_admittance()
        idx = self._find_freq_index(data.frequency, target_freq)
        return data.complex_data[idx]

    def get_char_impedance(self) -> PSCADMatrixData:
        """获取特性阻抗矩阵 Zc = Yc^(-1)"""
        yc_data = self.get_char_admittance()

        n_freq = len(yc_data.frequency)
        zc = np.zeros_like(yc_data.complex_data)
        zc_fitted = None

        for i in range(n_freq):
            try:
                zc[i] = np.linalg.inv(yc_data.complex_data[i])
            except np.linalg.LinAlgError:
                zc[i] = np.full_like(yc_data.complex_data[i], np.nan)

        if yc_data.has_fitted and yc_data.complex_fitted is not None:
            zc_fitted = np.zeros_like(yc_data.complex_fitted)
            for i in range(n_freq):
                try:
                    zc_fitted[i] = np.linalg.inv(yc_data.complex_fitted[i])
                except np.linalg.LinAlgError:
                    zc_fitted[i] = np.full_like(yc_data.complex_fitted[i], np.nan)

        return PSCADMatrixData(
            frequency=yc_data.frequency,
            magnitude=np.abs(zc),
            phase_deg=np.rad2deg(np.angle(zc)),
            complex_data=zc,
            unit='ohm',
            description='特性阻抗矩阵 Zc',
            has_fitted=yc_data.has_fitted,
            magnitude_fitted=np.abs(zc_fitted) if zc_fitted is not None else None,
            phase_fitted=np.rad2deg(np.angle(zc_fitted)) if zc_fitted is not None else None,
            complex_fitted=zc_fitted
        )

    def get_total_impedance(self, length_m: float) -> PSCADMatrixData:
        """获取整条线路的总阻抗矩阵"""
        z_data = self.get_impedance()

        return PSCADMatrixData(
            frequency=z_data.frequency,
            magnitude=z_data.magnitude * length_m,
            phase_deg=z_data.phase_deg,
            complex_data=z_data.complex_data * length_m,
            unit='ohms',
            description=f'总阻抗矩阵 Z×L (L={length_m}m)',
            has_fitted=False
        )

    def get_frequency_array(self) -> np.ndarray:
        """获取频率数组"""
        data = self.get_impedance()
        return data.frequency.copy()

    def get_fitting_error(self, data_type: str = 'char_admittance',
                          elements: str = 'diagonal',
                          min_magnitude_ratio: float = 0.01) -> Dict[str, any]:
        """
        计算拟合误差

        Parameters
        ----------
        data_type : str
            数据类型: 'char_admittance', 'h_mode', 'h_phase'
        elements : str
            计算哪些元素的误差:
            - 'diagonal': 只计算对角元素（矩阵数据时默认）
            - 'all': 计算所有元素
        min_magnitude_ratio : float
            最小幅值比例阈值，幅值小于(最大值×此比例)的元素不参与误差计算

        Returns
        -------
        dict
            包含各种误差统计量
        """
        data = self._load_data(data_type)

        if not data.has_fitted:
            raise ValueError(f"{data_type} 没有拟合值")

        # 选择要分析的元素
        if data.magnitude.ndim == 3:
            n = data.magnitude.shape[1]

            if elements == 'diagonal':
                mag_calc = np.array([data.magnitude[:, i, i] for i in range(n)]).T
                mag_fitted = np.array([data.magnitude_fitted[:, i, i] for i in range(n)]).T
                phase_calc = np.array([data.phase_deg[:, i, i] for i in range(n)]).T
                phase_fitted = np.array([data.phase_fitted[:, i, i] for i in range(n)]).T
            else:
                mag_calc = data.magnitude.reshape(data.magnitude.shape[0], -1)
                mag_fitted = data.magnitude_fitted.reshape(data.magnitude_fitted.shape[0], -1)
                phase_calc = data.phase_deg.reshape(data.phase_deg.shape[0], -1)
                phase_fitted = data.phase_fitted.reshape(data.phase_fitted.shape[0], -1)
        else:
            # 向量数据 (h_mode)
            mag_calc = data.magnitude
            mag_fitted = data.magnitude_fitted
            phase_calc = data.phase_deg
            phase_fitted = data.phase_fitted

        # 使用相对阈值过滤小值
        max_mag = np.max(mag_calc)
        threshold = max_mag * min_magnitude_ratio
        valid_mask = mag_calc > threshold

        # 计算幅值相对误差 (%)
        mag_error = np.zeros_like(mag_calc)
        mag_error[valid_mask] = np.abs(mag_calc[valid_mask] - mag_fitted[valid_mask]) / mag_calc[valid_mask] * 100

        # 计算相角误差 (度)，处理±180°跳变
        phase_error = np.abs(phase_calc - phase_fitted)
        phase_error = np.minimum(phase_error, 360 - phase_error)
        phase_error[~valid_mask] = 0

        valid_mag_errors = mag_error[valid_mask]
        valid_phase_errors = phase_error[valid_mask]

        return {
            'frequency': data.frequency,
            'mag_error_percent': mag_error,
            'phase_error_deg': phase_error,
            'valid_mask': valid_mask,
            'max_mag_error': np.max(valid_mag_errors) if len(valid_mag_errors) > 0 else 0.0,
            'mean_mag_error': np.mean(valid_mag_errors) if len(valid_mag_errors) > 0 else 0.0,
            'max_phase_error': np.max(valid_phase_errors) if len(valid_phase_errors) > 0 else 0.0,
            'mean_phase_error': np.mean(valid_phase_errors) if len(valid_phase_errors) > 0 else 0.0,
            'n_valid_points': int(np.sum(valid_mask)),
            'n_total_points': int(valid_mask.size),
            'threshold_used': threshold,
            'elements_analyzed': elements
        }

    def check_files(self) -> Dict[str, bool]:
        """检查各个数据文件是否存在"""
        result = {}
        for name, config in self._file_config.items():
            mag_file = self._get_file_path(config['mag_suffix'])
            arg_file = self._get_file_path(config['arg_suffix'])
            result[name] = os.path.exists(mag_file) and os.path.exists(arg_file)
        return result

    def summary(self) -> str:
        """返回数据摘要"""
        lines = [
            "=" * 65,
            "PSCAD 线路常数数据摘要",
            "=" * 65,
            f"文件路径: {self.file_prefix}",
            f"导体数量: {self.n_conductors}",
            f"模态数量: {self.n_modes}",
        ]

        # 尝试读取频率信息
        try:
            z_data = self.get_impedance()
            lines.append(f"频率范围: {z_data.frequency[0]:.4f} Hz ~ {z_data.frequency[-1]:.2f} Hz")
            lines.append(f"频率点数: {len(z_data.frequency)}")
        except:
            lines.append("频率信息: 无法读取")

        lines.append("")
        lines.append("文件状态:")

        file_status = self.check_files()
        for name, config in self._file_config.items():
            exists = "✓" if file_status[name] else "✗"
            fitted = "(含拟合值)" if config['has_fitted'] else ""
            lines.append(f"  {exists} {config['description']:20s} {fitted}")

        lines.append("=" * 65)
        return "\n".join(lines)

    def verify_all_data(self) -> Dict[str, dict]:
        """验证所有数据的读取"""
        results = {}

        for name in self._file_config.keys():
            try:
                data = self._load_data(name)
                results[name] = {
                    'success': True,
                    'shape': data.complex_data.shape,
                    'has_fitted': data.has_fitted,
                    'freq_range': (data.frequency[0], data.frequency[-1]),
                    'n_freq': len(data.frequency)
                }
            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e)
                }

        return results


def setup_chinese_font():
    """配置matplotlib中文字体支持"""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontManager

    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'STHeiti', 'STSong',
        'PingFang SC', 'Noto Sans CJK SC', 'Noto Sans CJK',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Droid Sans Fallback', 'AR PL UMing CN',
    ]

    fm = FontManager()
    available_fonts = set([f.name for f in fm.ttflist])

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return True
    return False


class PSCADPlotter:
    """PSCAD数据可视化工具"""

    def __init__(self, reader: PSCADFileReader, use_chinese: bool = True):
        self.reader = reader
        self.use_chinese = use_chinese
        self._chinese_available = False

        if use_chinese:
            self._chinese_available = setup_chinese_font()
            if not self._chinese_available:
                warnings.warn("未找到中文字体，将使用英文标签")
                self.use_chinese = False

        self._labels = {
            'frequency': ('频率 (Hz)', 'Frequency (Hz)'),
            'magnitude': ('幅值', 'Magnitude'),
            'phase': ('相角 (°)', 'Phase (°)'),
            'impedance': ('阻抗', 'Impedance'),
            'admittance': ('导纳', 'Admittance'),
            'char_admittance': ('特性导纳', 'Char. Admittance'),
            'calculated': ('计算值', 'Calculated'),
            'fitted': ('拟合值', 'Fitted'),
            'fitting_result': ('拟合效果', 'Fitting Result'),
            'attenuation': ('衰减因子', 'Attenuation'),
            'mode_propagation': ('模态传播函数', 'Modal Propagation'),
        }

    def _label(self, key: str) -> str:
        if key in self._labels:
            return self._labels[key][0 if self.use_chinese else 1]
        return key

    def plot_impedance_vs_freq(self, i: int = 0, j: int = 0, ax=None):
        """绘制阻抗随频率变化曲线"""
        import matplotlib.pyplot as plt

        data = self.reader.get_impedance()
        freq = data.frequency
        z_ij = data.complex_data[:, i, j]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_xlabel(self._label('frequency'))
        ax.set_ylabel(f'|Z{i + 1}{j + 1}| (Ω/m)', color='blue')
        line1 = ax.semilogx(freq, np.abs(z_ij), 'b-', label=self._label('magnitude'))
        ax.tick_params(axis='y', labelcolor='blue')

        ax2 = ax.twinx()
        ax2.set_ylabel(f'∠Z{i + 1}{j + 1} (°)', color='red')
        line2 = ax2.semilogx(freq, np.rad2deg(np.angle(z_ij)), 'r--', label=self._label('phase'))
        ax2.tick_params(axis='y', labelcolor='red')

        lines = line1 + line2
        ax.legend(lines, [l.get_label() for l in lines], loc='best')
        ax.set_title(f'{self._label("impedance")} Z{i + 1}{j + 1}')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        return ax

    def plot_char_admittance_fitting(self, i: int = 0, j: int = 0):
        """绘制特性导纳拟合效果"""
        import matplotlib.pyplot as plt

        data = self.reader.get_char_admittance()
        freq = data.frequency

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 幅值
        axes[0].semilogx(freq, data.magnitude[:, i, j], 'b-',
                         label=self._label('calculated'), linewidth=2)
        axes[0].semilogx(freq, data.magnitude_fitted[:, i, j], 'r--',
                         label=self._label('fitted'), linewidth=1.5)
        axes[0].set_ylabel(f'|Yc{i + 1}{j + 1}| (mho)')
        axes[0].legend()
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{self._label("char_admittance")} Yc{i + 1}{j + 1} {self._label("fitting_result")}')

        # 相角
        axes[1].semilogx(freq, data.phase_deg[:, i, j], 'b-',
                         label=self._label('calculated'), linewidth=2)
        axes[1].semilogx(freq, data.phase_fitted[:, i, j], 'r--',
                         label=self._label('fitted'), linewidth=1.5)
        axes[1].set_xlabel(self._label('frequency'))
        axes[1].set_ylabel(f'∠Yc{i + 1}{j + 1} (°)')
        axes[1].legend()
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig, axes

    def plot_propagation_modes(self):
        """绘制模态传播特性"""
        import matplotlib.pyplot as plt

        data = self.reader.get_h_mode()
        freq = data.frequency
        n_modes = data.complex_data.shape[1]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 幅值
        for m in range(n_modes):
            axes[0].semilogx(freq, data.magnitude[:, m], label=f'Mode {m + 1}')
        axes[0].set_ylabel(f'|H| ({self._label("attenuation")})')
        axes[0].legend(ncol=3, loc='lower left')
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{self._label("mode_propagation")} - {self._label("magnitude")}')
        axes[0].set_ylim([0, 1.05])

        # 相角
        for m in range(n_modes):
            axes[1].semilogx(freq, data.phase_deg[:, m], label=f'Mode {m + 1}')
        axes[1].set_xlabel(self._label('frequency'))
        axes[1].set_ylabel('∠H (°)')
        axes[1].legend(ncol=3, loc='lower left')
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
        axes[1].set_title(f'{self._label("mode_propagation")} - {self._label("phase")}')

        plt.tight_layout()
        return fig, axes

    def plot_impedance_matrix_heatmap(self, target_freq: float = 50.0):
        """绘制阻抗矩阵热力图"""
        import matplotlib.pyplot as plt

        Z = self.reader.get_impedance_at_freq(target_freq)
        n = Z.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 幅值
        im1 = axes[0].imshow(np.abs(Z) * 1000, cmap='YlOrRd')
        axes[0].set_title(f'|Z| @ {target_freq} Hz (mΩ/m)')
        axes[0].set_xticks(range(n))
        axes[0].set_yticks(range(n))
        axes[0].set_xticklabels([str(i + 1) for i in range(n)])
        axes[0].set_yticklabels([str(i + 1) for i in range(n)])
        plt.colorbar(im1, ax=axes[0])

        for i in range(n):
            for j in range(n):
                val = np.abs(Z[i, j]) * 1000
                axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7)

        # 相角
        im2 = axes[1].imshow(np.rad2deg(np.angle(Z)), cmap='RdBu_r', vmin=-90, vmax=90)
        axes[1].set_title(f'∠Z @ {target_freq} Hz (°)')
        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_xticklabels([str(i + 1) for i in range(n)])
        axes[1].set_yticklabels([str(i + 1) for i in range(n)])
        plt.colorbar(im2, ax=axes[1])

        for i in range(n):
            for j in range(n):
                val = np.rad2deg(np.angle(Z[i, j]))
                axes[1].text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7)

        plt.tight_layout()
        return fig, axes


def read_pscad_summary(out_file: str) -> dict:
    """读取 .out 摘要文件"""
    result = {
        'rxb_metallic': {},
        'rxb_ground': {},
        'min_time_delay_ms': None,
        'recommended_timestep_ms': None
    }

    if not os.path.exists(out_file):
        return result

    with open(out_file, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()

        if 'Metallic Mode' in line:
            current_section = 'metallic'
        elif 'Ground Mode' in line:
            current_section = 'ground'
        elif 'Resistance' in line and 'Rsq' in line:
            try:
                val = float(line.split()[-1].replace('E', 'e'))
                if current_section == 'metallic':
                    result['rxb_metallic']['R_pu'] = val
                elif current_section == 'ground':
                    result['rxb_ground']['R_pu'] = val
            except:
                pass
        elif 'Reactance' in line and 'Xsq' in line:
            try:
                val = float(line.split()[-1].replace('E', 'e'))
                if current_section == 'metallic':
                    result['rxb_metallic']['X_pu'] = val
                elif current_section == 'ground':
                    result['rxb_ground']['X_pu'] = val
            except:
                pass
        elif 'Susceptance' in line and 'Bsq' in line:
            try:
                val = float(line.split()[-1].replace('E', 'e'))
                if current_section == 'metallic':
                    result['rxb_metallic']['B_pu'] = val
                elif current_section == 'ground':
                    result['rxb_ground']['B_pu'] = val
            except:
                pass
        elif 'Surge Impedance' in line:
            try:
                val = float(line.split()[-1].replace('E', 'e'))
                if current_section == 'metallic':
                    result['rxb_metallic']['Zc_pu'] = val
                elif current_section == 'ground':
                    result['rxb_ground']['Zc_pu'] = val
            except:
                pass
        elif 'Minimum Time Delay' in line:
            try:
                result['min_time_delay_ms'] = float(line.split()[-1].replace('E', 'e'))
            except:
                pass
        elif 'Recommended Time Step' in line:
            try:
                result['recommended_timestep_ms'] = float(line.split()[-1].replace('E', 'e'))
            except:
                pass

    return result


def export_to_csv(reader: PSCADFileReader, output_dir: str,
                  data_types: List[str] = None):
    """导出数据到CSV文件"""
    if data_types is None:
        data_types = ['impedance', 'admittance', 'char_admittance', 'h_mode', 'propagation']

    os.makedirs(output_dir, exist_ok=True)

    for dtype in data_types:
        try:
            data = reader._load_data(dtype)
            freq = data.frequency
            n_freq = len(freq)

            if data.complex_data.ndim == 3:
                n1, n2 = data.complex_data.shape[1], data.complex_data.shape[2]
                cols = ['frequency_Hz']
                for i in range(n1):
                    for j in range(n2):
                        cols.extend([f'Re_{i + 1}{j + 1}', f'Im_{i + 1}{j + 1}'])

                rows = []
                for f_idx in range(n_freq):
                    row = [freq[f_idx]]
                    for i in range(n1):
                        for j in range(n2):
                            z = data.complex_data[f_idx, i, j]
                            row.extend([z.real, z.imag])
                    rows.append(row)
            else:
                n_elem = data.complex_data.shape[1]
                cols = ['frequency_Hz']
                for i in range(n_elem):
                    cols.extend([f'Re_{i + 1}', f'Im_{i + 1}'])

                rows = []
                for f_idx in range(n_freq):
                    row = [freq[f_idx]]
                    for i in range(n_elem):
                        z = data.complex_data[f_idx, i]
                        row.extend([z.real, z.imag])
                    rows.append(row)

            csv_path = os.path.join(output_dir, f'{dtype}.csv')
            with open(csv_path, 'w') as f:
                f.write(','.join(cols) + '\n')
                for row in rows:
                    f.write(','.join([f'{v:.10e}' for v in row]) + '\n')

            print(f"已导出: {csv_path}")
        except Exception as e:
            print(f"导出 {dtype} 失败: {e}")


# ==================== 主程序 ====================

if __name__ == '__main__':
    import sys

    print("PSCAD 线路常数文件读取器 v2.0")
    print("=" * 65)

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = './Cable_1'

    print(f"输入路径: {path}")

    # 创建读取器
    reader = PSCADFileReader(path)

    # 打印摘要
    print(reader.summary())

    # 验证所有数据
    print("\n验证所有数据读取:")
    print("-" * 65)
    results = reader.verify_all_data()
    for name, info in results.items():
        if info['success']:
            shape_str = str(info['shape'])
            fitted_str = "有拟合" if info['has_fitted'] else "无拟合"
            print(f"  ✓ {name:18s} shape={shape_str:18s} {fitted_str}")
        else:
            print(f"  ✗ {name:18s} 错误: {info['error']}")

    # 示例：拟合误差
    print("\n拟合误差统计:")
    print("-" * 65)
    for dtype in ['char_admittance', 'h_mode', 'h_phase']:
        try:
            err = reader.get_fitting_error(dtype, min_magnitude_ratio=0.01)
            print(f"  {dtype:18s}: 最大幅值误差={err['max_mag_error']:.4f}%, "
                  f"最大相角误差={err['max_phase_error']:.4f}° "
                  f"(有效点:{err['n_valid_points']}/{err['n_total_points']})")
        except Exception as e:
            print(f"  {dtype:18s}: {e}")

    print("\n读取完成!")
