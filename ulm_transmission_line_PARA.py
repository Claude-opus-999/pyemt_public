"""ULM (Universal Line Model) 频变参数传输线,Numba JIT 加速实现。

所有递归卷积热路径提取为 ``@njit`` 函数,状态数组打包为连续 3D/4D 数组
以兼容 Numba。

数据布局(packed arrays)
------------------------
    yc_residues_3d : (n_yc, nc, nc)              complex128
    qn_yc_3d       : (n_yc, nc, nc)              complex128
    xk_2d / xm_2d  : (n_yc, nc)                  complex128
    h_poles_2d     : (nmod, max_n_h)             complex128   (padded)
    h_residues_4d  : (nmod, max_n_h, nc, nc)     complex128   (padded)
    pi_h_2d        : (nmod, max_n_h)             complex128
    qij_h_4d       : (nmod, max_n_h, nc, nc)     complex128
    bk_3d / bm_3d  : (nmod, max_n_h, nc)         complex128
    n_h_poles_arr  : (nmod,)                     int64        (各模态实际极点数)

参考
----
- Zanon et al., "Implementation of the universal line model in ATP."
- Morched et al., "A universal model for accurate calculation of electromagnetic
  transients on overhead lines and underground cables."
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numba import njit


logger = logging.getLogger(__name__)


# ===========================================================================
# 第一部分:FitULM 数据结构与 I/O
# ===========================================================================

@dataclass
class FitULMData:
    """FitULM 拟合数据。

    数学表示:
        Yc(s)     = yc_d + Σ yc_residues[n] / (s - yc_poles[n])
        H_mode(s) = exp(-s·τ_mode) · Σ h_residues[mode][i] / (s - h_poles[mode][i])
    """

    nf: int
    nmod: int
    n_poles_yc: int
    n_poles_h: List[int]
    time_delays: np.ndarray
    yc_poles: np.ndarray
    yc_residues: List[np.ndarray]
    yc_d: np.ndarray
    h_poles: List[np.ndarray]
    h_residues: List[List[np.ndarray]]

    def validate(self) -> bool:
        """验证数据一致性。"""
        try:
            assert len(self.time_delays) == self.nmod
            assert len(self.yc_poles) == self.n_poles_yc
            assert len(self.yc_residues) == self.n_poles_yc
            assert self.yc_d.shape == (self.nf, self.nf)
            assert len(self.h_poles) == self.nmod
            assert len(self.h_residues) == self.nmod
            for n in range(self.n_poles_yc):
                assert self.yc_residues[n].shape == (self.nf, self.nf)
            for mode in range(self.nmod):
                assert len(self.h_poles[mode]) == self.n_poles_h[mode]
                assert len(self.h_residues[mode]) == self.n_poles_h[mode]
            return True
        except AssertionError:
            return False

    def print_summary(self, precision: int = 15) -> None:
        print("FitULM 数据摘要")
        print("-" * 60)
        print(f"  导体数 nf    = {self.nf}")
        print(f"  模态数 nmod  = {self.nmod}")
        print(f"  Yc 极点数    = {self.n_poles_yc}")
        print(f"  H 极点数     = {self.n_poles_h}")
        print("  传播时延:")
        for mode, tau in enumerate(self.time_delays):
            print(f"    模态 {mode}: {tau:.{precision}E} s "
                  f"({tau * 1e6:.{precision}E} μs)")

    def print_detailed_info(self, precision: int = 15) -> None:
        """打印 FitULMData 完整明细。"""
        self.print_summary(precision)

        print("\n【Yc 极点】")
        for i, pole in enumerate(self.yc_poles):
            if np.imag(pole) < 0:
                continue
            if np.imag(pole) == 0:
                print(f"  [{i}] {np.real(pole):.{precision}E}")
            else:
                print(f"  [{i}] {np.real(pole):.{precision}E} "
                      f"+ {np.imag(pole):.{precision}E}j")

        print("\n【Yc 常数项 k0】")
        for row in range(self.nf):
            print("  [ " + " ".join(
                f"{np.real(self.yc_d[row, col]):.{precision}E}"
                for col in range(self.nf)
            ) + " ]")

        print("\n【H 极点】")
        for mode in range(self.nmod):
            print(f"  模态 {mode}:")
            for i, pole in enumerate(self.h_poles[mode]):
                if np.imag(pole) < 0:
                    continue
                if np.imag(pole) == 0:
                    print(f"    [{i}] {np.real(pole):.{precision}E}")
                else:
                    print(f"    [{i}] {np.real(pole):.{precision}E} "
                          f"+ {np.imag(pole):.{precision}E}j")


class FitULMReader:
    """fitULM 文件读取器。

    格式规则:
        - '#' / '$' 开头为注释,行尾 '$' 截除
        - 极点:负数=实极点;正数=复极点的 |Re|(下一行为虚部)
        - Yc 留数上三角、H 留数完整矩阵,共轭极点对自动生成
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data_lines: List[str] = []
        self.current_line: int = 0

    def _read_file(self) -> None:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"文件未找到: {self.filepath}")

        cleaned: List[str] = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('$'):
                    continue
                if line.endswith('$'):
                    line = line[:-1].strip()
                if line:
                    cleaned.append(line)
        self.data_lines = cleaned
        self.current_line = 0

    def _read_value(self) -> float:
        if self.current_line >= len(self.data_lines):
            raise ValueError(f"文件读取越界,行号 {self.current_line}")
        value = float(self.data_lines[self.current_line])
        self.current_line += 1
        return value

    def _read_int(self) -> int:
        return int(self._read_value())

    def _read_pole(self) -> complex:
        pole_indicator = self._read_value()
        if pole_indicator < 0:
            return complex(pole_indicator, 0)
        return complex(-abs(pole_indicator), self._read_value())

    def _read_upper_triangular(self, n: int, is_complex_pole: bool) -> np.ndarray:
        matrix = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(i, n):
                re = self._read_value()
                im = self._read_value() if is_complex_pole else 0.0
                matrix[i, j] = complex(re, im)
                if i != j:
                    matrix[j, i] = matrix[i, j]
        return matrix

    def _read_full_matrix(self, n: int, is_complex_pole: bool) -> np.ndarray:
        matrix = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                re = self._read_value()
                im = self._read_value() if is_complex_pole else 0.0
                matrix[i, j] = complex(re, im)
        return matrix

    def read(self, verbose: bool = True, precision: int = 15) -> FitULMData:
        self._read_file()

        nf = self._read_int()
        nmod = self._read_int()
        n_poles_yc_file = self._read_int()
        n_poles_h_file = [self._read_int() for _ in range(nmod)]
        time_delays = np.array([self._read_value() for _ in range(nmod)])

        if verbose:
            logger.info("fitULM: nf=%d, nmod=%d, Yc极点=%d, H极点=%s",
                        nf, nmod, n_poles_yc_file, n_poles_h_file)

        # Yc
        yc_poles: List[complex] = []
        yc_residues: List[np.ndarray] = []
        for _ in range(n_poles_yc_file):
            pole = self._read_pole()
            is_complex = (np.imag(pole) != 0)
            residue = self._read_upper_triangular(nf, is_complex)
            yc_poles.append(pole)
            yc_residues.append(residue)
            if is_complex:
                yc_poles.append(np.conj(pole))
                yc_residues.append(np.conj(residue))

        # H
        h_poles: List[np.ndarray] = []
        h_residues: List[List[np.ndarray]] = []
        for mode in range(nmod):
            mode_poles: List[complex] = []
            mode_residues: List[np.ndarray] = []
            for _ in range(n_poles_h_file[mode]):
                pole = self._read_pole()
                is_complex = (np.imag(pole) != 0)
                residue = self._read_full_matrix(nf, is_complex)
                mode_poles.append(pole)
                mode_residues.append(residue)
                if is_complex:
                    mode_poles.append(np.conj(pole))
                    mode_residues.append(np.conj(residue))
            h_poles.append(np.array(mode_poles))
            h_residues.append(mode_residues)

        yc_d = self._read_upper_triangular(nf, is_complex_pole=False)

        remaining = len(self.data_lines) - self.current_line
        if remaining > 0:
            logger.warning("文件末尾还有 %d 行未读取", remaining)

        return FitULMData(
            nf=nf, nmod=nmod,
            n_poles_yc=len(yc_poles),
            n_poles_h=[len(p) for p in h_poles],
            time_delays=time_delays,
            yc_poles=np.array(yc_poles), yc_residues=yc_residues,
            yc_d=yc_d, h_poles=h_poles, h_residues=h_residues,
        )


class FitULMWriter:
    """fitULM 格式文件写入器。"""

    def __init__(self, data: FitULMData):
        self.data = data
        self.output_lines: List[str] = []

    def _write_value(self, value: float) -> None:
        self.output_lines.append(f"{value:.15E}")

    def _write_int(self, value: int) -> None:
        self.output_lines.append(str(value))

    def _write_pole(self, pole: complex) -> None:
        """实极点写实部;复极点(Im>0 的那个)写 |Re| 和 Im。"""
        if np.imag(pole) == 0:
            self._write_value(np.real(pole))
        elif np.imag(pole) > 0:
            self._write_value(abs(np.real(pole)))
            self._write_value(np.imag(pole))

    def _write_upper_triangular(self, matrix: np.ndarray,
                                is_complex_pole: bool) -> None:
        n = matrix.shape[0]
        for i in range(n):
            for j in range(i, n):
                self._write_value(np.real(matrix[i, j]))
                if is_complex_pole:
                    self._write_value(np.imag(matrix[i, j]))

    def _write_full_matrix(self, matrix: np.ndarray,
                           is_complex_pole: bool) -> None:
        n = matrix.shape[0]
        for i in range(n):
            for j in range(n):
                self._write_value(np.real(matrix[i, j]))
                if is_complex_pole:
                    self._write_value(np.imag(matrix[i, j]))

    @staticmethod
    def _get_file_pole_count(poles: np.ndarray) -> int:
        """文件中的极点条目数:共轭对记为 1 个。"""
        count = 0
        i = 0
        while i < len(poles):
            count += 1
            i += 2 if np.imag(poles[i]) != 0 else 1
        return count

    def write(self, filepath: str) -> None:
        self.output_lines = []
        data = self.data

        self._write_int(data.nf)
        self._write_int(data.nmod)
        self._write_int(self._get_file_pole_count(data.yc_poles))

        for mode in range(data.nmod):
            self._write_int(self._get_file_pole_count(data.h_poles[mode]))
        for tau in data.time_delays:
            self._write_value(tau)

        # Yc
        i = 0
        while i < len(data.yc_poles):
            pole = data.yc_poles[i]
            is_complex = (np.imag(pole) != 0)
            self._write_pole(pole)
            self._write_upper_triangular(data.yc_residues[i], is_complex)
            i += 2 if is_complex else 1

        # H
        for mode in range(data.nmod):
            i = 0
            while i < len(data.h_poles[mode]):
                pole = data.h_poles[mode][i]
                is_complex = (np.imag(pole) != 0)
                self._write_pole(pole)
                self._write_full_matrix(data.h_residues[mode][i], is_complex)
                i += 2 if is_complex else 1

        # k0
        self._write_upper_triangular(data.yc_d, is_complex_pole=False)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.output_lines) + '\n')
        logger.info("fitULM 文件已写入: %s", filepath)


def convert_to_ulm_format(fit_data: FitULMData) -> Dict[str, Any]:
    """FitULMData → ULMModel 构造字典。"""
    return {
        'nc':          fit_data.nf,
        'nmod':        fit_data.nmod,
        'yc_poles':    fit_data.yc_poles,
        'yc_residues': fit_data.yc_residues,
        'yc_d':        fit_data.yc_d,
        'h_poles':     fit_data.h_poles,
        'h_residues':  fit_data.h_residues,
        'time_delays': fit_data.time_delays,
    }


def validate_fitting(
    fit_data: FitULMData, freq: np.ndarray,
    verbose: bool = True, precision: int = 15,
) -> Dict[str, Any]:
    """验证拟合结果:重建 Yc、H,并检查极点稳定性。"""
    n_freq = len(freq)
    nf = fit_data.nf
    nmod = fit_data.nmod
    s = 1j * 2 * np.pi * freq

    Yc_fit = np.zeros((n_freq, nf, nf), dtype=complex)
    for f_idx in range(n_freq):
        Yc_fit[f_idx] = fit_data.yc_d.copy()
        for n, pole in enumerate(fit_data.yc_poles):
            Yc_fit[f_idx] += fit_data.yc_residues[n] / (s[f_idx] - pole)

    H_fit: List[np.ndarray] = []
    for mode in range(nmod):
        H_mode = np.zeros((n_freq, nf, nf), dtype=complex)
        tau = fit_data.time_delays[mode]
        for f_idx in range(n_freq):
            prop_factor = np.exp(-s[f_idx] * tau)
            H_rational = np.zeros((nf, nf), dtype=complex)
            for i, pole in enumerate(fit_data.h_poles[mode]):
                H_rational += fit_data.h_residues[mode][i] / (s[f_idx] - pole)
            H_mode[f_idx] = prop_factor * H_rational
        H_fit.append(H_mode)

    poles_stability: Dict[str, Any] = {
        'yc_stable': True, 'h_stable': [True] * nmod,
        'yc_unstable_poles': [],
        'h_unstable_poles': [[] for _ in range(nmod)],
    }
    for pole in fit_data.yc_poles:
        if np.real(pole) > 0:
            poles_stability['yc_stable'] = False
            poles_stability['yc_unstable_poles'].append(pole)
    for mode in range(nmod):
        for pole in fit_data.h_poles[mode]:
            if np.real(pole) > 0:
                poles_stability['h_stable'][mode] = False
                poles_stability['h_unstable_poles'][mode].append(pole)

    if verbose:
        logger.info("拟合验证: 频率 %.*E-%.*E Hz, Yc稳定=%s, H稳定=%s",
                    precision, freq[0], precision, freq[-1],
                    poles_stability['yc_stable'], poles_stability['h_stable'])

    return {
        'freq': freq, 'Yc_fit': Yc_fit,
        'H_fit': H_fit, 'poles_stability': poles_stability,
    }


def check_passivity(
    fit_data: FitULMData, freq: np.ndarray,
    verbose: bool = True, precision: int = 15,
) -> Dict[str, Any]:
    """无源性检查:各频点 Re(Yc) 应为半正定。"""
    validation = validate_fitting(fit_data, freq, verbose=False)
    Yc_fit = validation['Yc_fit']

    violations: List[Dict[str, float]] = []
    for f_idx, f in enumerate(freq):
        min_eig = float(np.min(np.linalg.eigvalsh(np.real(Yc_fit[f_idx]))))
        if min_eig < -1e-10:
            violations.append({'frequency': f, 'min_eigenvalue': min_eig})

    is_passive = not violations
    if verbose:
        if is_passive:
            logger.info("Yc 满足无源性")
        else:
            logger.warning("Yc 在 %d 个频点违反无源性", len(violations))

    return {
        'is_passive': is_passive,
        'violations': violations,
        'n_violations': len(violations),
    }


def print_poles_info(fit_data: FitULMData, precision: int = 15) -> None:
    """打印极点明细。"""
    print(f"Yc 极点 ({fit_data.n_poles_yc} 个):")
    for i, pole in enumerate(fit_data.yc_poles):
        re, im = np.real(pole), np.imag(pole)
        if im == 0:
            print(f"  [{i:2d}] 实:   {re:.{precision}E}")
        elif im > 0:
            print(f"  [{i:2d}] 复:   {re:.{precision}E} + {im:.{precision}E}j")
        else:
            print(f"  [{i:2d}] 共轭: {re:.{precision}E} {im:.{precision}E}j")

    print("\nH 极点:")
    for mode in range(fit_data.nmod):
        print(f"  模态 {mode} ({fit_data.n_poles_h[mode]} 个):")
        for i, pole in enumerate(fit_data.h_poles[mode]):
            re, im = np.real(pole), np.imag(pole)
            if im == 0:
                print(f"    [{i:2d}] 实:   {re:.{precision}E}")
            elif im > 0:
                print(f"    [{i:2d}] 复:   {re:.{precision}E} + {im:.{precision}E}j")
            else:
                print(f"    [{i:2d}] 共轭: {re:.{precision}E} {im:.{precision}E}j")


def print_residues_info(fit_data: FitULMData, precision: int = 15) -> None:
    """打印留数矩阵明细。"""
    print("Yc 留数矩阵:")
    for i, residue in enumerate(fit_data.yc_residues):
        pole = fit_data.yc_poles[i]
        if np.imag(pole) < 0:
            continue
        print(f"  极点 {i} ({np.real(pole):.6E}+{np.imag(pole):.6E}j):")
        _print_matrix(residue, fit_data.nf, precision, indent=4)

    print("\nYc 常数项 k0:")
    _print_matrix(fit_data.yc_d, fit_data.nf, precision, indent=2,
                  real_only=True)

    print("\nH 留数矩阵:")
    for mode in range(fit_data.nmod):
        print(f"  模态 {mode}:")
        for i, residue in enumerate(fit_data.h_residues[mode]):
            pole = fit_data.h_poles[mode][i]
            if np.imag(pole) < 0:
                continue
            print(f"    极点 {i} ({np.real(pole):.6E}+{np.imag(pole):.6E}j):")
            _print_matrix(residue, fit_data.nf, precision, indent=6)


def _print_matrix(matrix: np.ndarray, n: int, precision: int,
                  indent: int = 2, real_only: bool = False) -> None:
    pad = " " * indent
    for row in range(n):
        parts: List[str] = []
        for col in range(n):
            val = matrix[row, col]
            if real_only or np.imag(val) == 0:
                parts.append(f"{np.real(val):.{precision}E}")
            else:
                parts.append(
                    f"({np.real(val):.{precision}E}{np.imag(val):+.{precision}E}j)"
                )
        print(f"{pad}[ " + " ".join(parts) + " ]")


def create_test_fitulm_data(nf: int = 3, n_poles: int = 4) -> FitULMData:
    """生成合成 FitULM 数据(用于单元测试或 JIT 预热)。"""
    nmod = nf
    yc_poles = np.array(
        [-1000 + 0j, -5000 + 2000j, -5000 - 2000j, -20000 + 0j][:n_poles]
    )

    yc_residues: List[np.ndarray] = []
    for _ in range(n_poles):
        diag = np.random.uniform(0.001, 0.01, nf)
        off_diag = np.random.uniform(0.0001, 0.001, (nf, nf))
        matrix = off_diag + np.diag(diag)
        matrix = (matrix + matrix.T) / 2
        yc_residues.append(matrix.astype(complex))

    yc_d = np.diag(np.random.uniform(0.003, 0.01, nf))
    yc_d = yc_d + yc_d.T - np.diag(np.diag(yc_d))

    time_delays = np.array([1e-5 * (i + 1) for i in range(nmod)])

    h_poles: List[np.ndarray] = []
    h_residues: List[List[np.ndarray]] = []
    for mode in range(nmod):
        n_h = 2 + mode % 3
        poles_list: List[complex] = [
            -2000 * (mode + 1) + 1000j * (mode + 1),
            -2000 * (mode + 1) - 1000j * (mode + 1),
        ]
        if n_h > 2:
            poles_list.append(complex(-5000 * (mode + 1)))
        h_poles.append(np.array(poles_list[:n_h]))

        residues: List[np.ndarray] = []
        for _ in range(n_h):
            diag = np.random.uniform(0.3, 0.5, nf)
            off_diag = np.random.uniform(0.01, 0.05, (nf, nf))
            residues.append((off_diag + np.diag(diag)).astype(complex))
        h_residues.append(residues)

    return FitULMData(
        nf=nf, nmod=nmod, n_poles_yc=n_poles,
        n_poles_h=[len(p) for p in h_poles],
        time_delays=time_delays, yc_poles=yc_poles,
        yc_residues=yc_residues, yc_d=yc_d,
        h_poles=h_poles, h_residues=h_residues,
    )


# ===========================================================================
# 第二部分:Numba JIT 核心计算函数
# ===========================================================================

@njit(cache=True)
def _jit_matvec(A, x, nc):
    """手工 y = A @ x,小矩阵优于 BLAS。"""
    y = np.zeros(nc, dtype=np.complex128)
    for i in range(nc):
        s = 0.0 + 0.0j
        for j in range(nc):
            s += A[i, j] * x[j]
        y[i] = s
    return y


@njit(cache=True)
def _jit_interpolate_history(history, delay_steps, step, history_length, nc):
    """从循环历史缓冲区线性插值取 delay_steps 步之前的值。"""
    idx_low = int(np.floor(delay_steps))
    idx_high = idx_low + 1
    frac = delay_steps - idx_low

    if idx_low < 0 or idx_high < 0:
        return np.zeros(nc, dtype=np.complex128)

    idx_low_buf = (step - idx_low + 1) % history_length
    idx_high_buf = (step - idx_high + 1) % history_length

    result = np.empty(nc, dtype=np.complex128)
    for i in range(nc):
        result[i] = ((1.0 - frac) * history[i, idx_low_buf]
                     + frac * history[i, idx_high_buf])
    return result


@njit(cache=True)
def _jit_compute_yc_history_term(pn_yc, qn_yc_3d, xk_2d, v_prev, n_yc, nc):
    """Yc 历史项 i_hist_yc = Σ (p_n · x_n + q_n @ v_prev)。"""
    i_hist = np.zeros(nc, dtype=np.complex128)
    for n in range(n_yc):
        pn = pn_yc[n]
        for i in range(nc):
            term = pn * xk_2d[n, i]
            qv = 0.0 + 0.0j
            for j in range(nc):
                qv += qn_yc_3d[n, i, j] * v_prev[j]
            i_hist[i] += term + qv
    return i_hist


@njit(cache=True)
def _jit_compute_f(i_vec, v_vec, yc_d, xk_2d, n_yc, nc):
    """行波 f = i + Yc · v = i + k₀ · v + Σ x_n。"""
    f = np.empty(nc, dtype=np.complex128)
    for i in range(nc):
        k0v = 0.0 + 0.0j
        for j in range(nc):
            k0v += yc_d[i, j] * v_vec[j]
        x_sum = 0.0 + 0.0j
        for n in range(n_yc):
            x_sum += xk_2d[n, i]
        f[i] = i_vec[i] + k0v + x_sum
    return f


@njit(cache=True)
def _jit_update_x_states(pn_yc, qn_yc_3d, xk_2d, v_curr, v_prev, n_yc, nc):
    """就地更新 Yc 状态:x_n(t) = p_n · x_n + q_n @ (v_curr + v_prev)。"""
    v_sum = np.empty(nc, dtype=np.complex128)
    for i in range(nc):
        v_sum[i] = v_curr[i] + v_prev[i]

    for n in range(n_yc):
        pn = pn_yc[n]
        for i in range(nc):
            qv = 0.0 + 0.0j
            for j in range(nc):
                qv += qn_yc_3d[n, i, j] * v_sum[j]
            xk_2d[n, i] = pn * xk_2d[n, i] + qv


@njit(cache=True)
def _jit_compute_b(pi_h_2d, qij_h_4d, b_states_3d, history,
                   tau_arr, dt, step, history_length,
                   nmod, n_h_poles_arr, nc):
    """传播项 b(t) = Σ_mode Σ_i [p_i · b_ij + q_ij @ (f(t-τ) + f(t-τ-Δt))]。"""
    result = np.zeros(nc, dtype=np.complex128)
    for mode in range(nmod):
        delay_steps = tau_arr[mode] / dt
        f_tau = _jit_interpolate_history(history, delay_steps,
                                         step, history_length, nc)
        f_tau_dt = _jit_interpolate_history(history,
                                            (tau_arr[mode] + dt) / dt,
                                            step, history_length, nc)

        f_sum = np.empty(nc, dtype=np.complex128)
        for k in range(nc):
            f_sum[k] = f_tau[k] + f_tau_dt[k]

        n_h = n_h_poles_arr[mode]
        for i in range(n_h):
            pi_val = pi_h_2d[mode, i]
            for k in range(nc):
                pb = pi_val * b_states_3d[mode, i, k]
                qf = 0.0 + 0.0j
                for j in range(nc):
                    qf += qij_h_4d[mode, i, k, j] * f_sum[j]
                result[k] += pb + qf
    return result


@njit(cache=True)
def _jit_update_b_states(pi_h_2d, qij_h_4d, b_states_3d, history,
                         tau_arr, dt, step, history_length,
                         nmod, n_h_poles_arr, nc):
    """就地更新 b 状态。"""
    for mode in range(nmod):
        delay_steps = tau_arr[mode] / dt
        f_tau = _jit_interpolate_history(history, delay_steps,
                                         step, history_length, nc)
        f_tau_dt = _jit_interpolate_history(history,
                                            (tau_arr[mode] + dt) / dt,
                                            step, history_length, nc)

        f_sum = np.empty(nc, dtype=np.complex128)
        for k in range(nc):
            f_sum[k] = f_tau[k] + f_tau_dt[k]

        n_h = n_h_poles_arr[mode]
        for i in range(n_h):
            pi_val = pi_h_2d[mode, i]
            for k in range(nc):
                qf = 0.0 + 0.0j
                for j in range(nc):
                    qf += qij_h_4d[mode, i, k, j] * f_sum[j]
                b_states_3d[mode, i, k] = pi_val * b_states_3d[mode, i, k] + qf


@njit(cache=True)
def _jit_calculate_history_and_update(
    nc, n_yc, nmod, max_n_h, history_length,
    n_h_poles_arr,
    pn_yc, qn_yc_3d, yc_d,
    pi_h_2d, qij_h_4d, tau_arr, dt,
    xk_2d, xm_2d,
    bk_3d, bm_3d,
    fk_history, fm_history,
    vk_prev, vm_prev, ik_prev, im_prev,
    step,
):
    """合并 calculate_history + update 为单次 JIT 调用。

    Returns
    -------
    (ik_hist_real, im_hist_real) : (nc,) float64
        状态数组就地更新。
    """
    # 1) 存储 f(t-Δt)
    buf_idx = step % history_length
    fk_prev = _jit_compute_f(im_prev, vm_prev, yc_d, xm_2d, n_yc, nc)
    fm_prev = _jit_compute_f(ik_prev, vk_prev, yc_d, xk_2d, n_yc, nc)
    for i in range(nc):
        fk_history[i, buf_idx] = fk_prev[i]
        fm_history[i, buf_idx] = fm_prev[i]

    # 2) Yc 历史项
    ik_yc_hist = _jit_compute_yc_history_term(
        pn_yc, qn_yc_3d, xk_2d, vk_prev, n_yc, nc
    )
    im_yc_hist = _jit_compute_yc_history_term(
        pn_yc, qn_yc_3d, xm_2d, vm_prev, n_yc, nc
    )

    # 3) 传播项 b
    bk = _jit_compute_b(pi_h_2d, qij_h_4d, bk_3d, fk_history,
                        tau_arr, dt, step, history_length,
                        nmod, n_h_poles_arr, nc)
    bm = _jit_compute_b(pi_h_2d, qij_h_4d, bm_3d, fm_history,
                        tau_arr, dt, step, history_length,
                        nmod, n_h_poles_arr, nc)

    # 4) i_hist = i_yc_hist - b
    ik_hist_real = np.empty(nc, dtype=np.float64)
    im_hist_real = np.empty(nc, dtype=np.float64)
    for i in range(nc):
        ik_hist_real[i] = (ik_yc_hist[i] - bk[i]).real
        im_hist_real[i] = (im_yc_hist[i] - bm[i]).real

    return ik_hist_real, im_hist_real


@njit(cache=True)
def _jit_do_update_states(
    nc, n_yc, nmod, max_n_h, history_length,
    n_h_poles_arr,
    pn_yc, qn_yc_3d,
    pi_h_2d, qij_h_4d, tau_arr, dt,
    xk_2d, xm_2d,
    bk_3d, bm_3d,
    fk_history, fm_history,
    vk_curr, vm_curr,
    vk_prev, vm_prev,
    step,
):
    """求解 v(t)、i(t) 后就地更新所有内部状态。"""
    _jit_update_x_states(pn_yc, qn_yc_3d, xk_2d, vk_curr, vk_prev, n_yc, nc)
    _jit_update_x_states(pn_yc, qn_yc_3d, xm_2d, vm_curr, vm_prev, n_yc, nc)

    _jit_update_b_states(pi_h_2d, qij_h_4d, bk_3d, fk_history,
                         tau_arr, dt, step, history_length,
                         nmod, n_h_poles_arr, nc)
    _jit_update_b_states(pi_h_2d, qij_h_4d, bm_3d, fm_history,
                         tau_arr, dt, step, history_length,
                         nmod, n_h_poles_arr, nc)


@njit(cache=True)
def _jit_full_step(
    nc, n_yc, nmod, max_n_h, history_length,
    n_h_poles_arr,
    pn_yc, qn_yc_3d, yc_d,
    pi_h_2d, qij_h_4d, tau_arr, dt,
    xk_2d, xm_2d,
    bk_3d, bm_3d,
    fk_history, fm_history,
    G_real,
    vk_real, vm_real,
    I_hist_k_real, I_hist_m_real,
    vk_prev, vm_prev,
    step,
):
    """单次 JIT 调用完成 EMTPSolver 每步的全部传输线计算。

    合并内容:
    0. i = G·v + I_hist    (原 Python numpy matmul,现内联)
    1. update_x_states     (梯形递推 x 状态更新)
    2. store f             (行波存入历史缓冲区)
    3. Yc 历史项            (特征导纳历史贡献)
    4. update_b + compute_b (单次遍历完成 b 状态更新 + b 结果累加)
    5. I_hist_new          (返回下一步所需的历史电流源)

    关键优化:
    - 消除 2 次 Python→JIT 过渡 + 2 次 numpy matmul
    - 内联插值:共享 step/step+1 的缓冲区索引计算 (n_hi == s_lo)
    - (mode, pole, nc, nc) 四重循环单次遍历,q_val 读一次用四次
    - b_states 读一次、更新后立即用于 compute_b,数据留在寄存器

    Parameters
    ----------
    G_real : (nc, nc) float64 — 导纳矩阵
    vk_real, vm_real : (nc,) float64 — 当前步电压
    I_hist_k_real, I_hist_m_real : (nc,) float64 — 当前 I_hist
    vk_prev, vm_prev : (nc,) complex128 — 上一步电压 (就地更新为当前步)
    step : int — 当前步号

    Returns
    -------
    (ik_hist_new, im_hist_new, ik_real, im_real) : 4 × (nc,) float64
    """
    next_step = step + 1

    # =============== 0. 内联 i = G·v + I_hist ===============
    vk_c = np.empty(nc, dtype=np.complex128)
    vm_c = np.empty(nc, dtype=np.complex128)
    ik_c = np.empty(nc, dtype=np.complex128)
    im_c = np.empty(nc, dtype=np.complex128)
    for i in range(nc):
        vk_c[i] = vk_real[i]
        vm_c[i] = vm_real[i]
        gvk = I_hist_k_real[i]
        gvm = I_hist_m_real[i]
        for j in range(nc):
            gvk += G_real[i, j] * vk_real[j]
            gvm += G_real[i, j] * vm_real[j]
        ik_c[i] = gvk
        im_c[i] = gvm

    # =============== 1. 更新 x 状态 ===============
    _jit_update_x_states(pn_yc, qn_yc_3d, xk_2d, vk_c, vk_prev, n_yc, nc)
    _jit_update_x_states(pn_yc, qn_yc_3d, xm_2d, vm_c, vm_prev, n_yc, nc)

    # =============== 2. 存储 f(step+1) ===============
    buf_idx = next_step % history_length
    fk = _jit_compute_f(im_c, vm_c, yc_d, xm_2d, n_yc, nc)
    fm = _jit_compute_f(ik_c, vk_c, yc_d, xk_2d, n_yc, nc)
    for i in range(nc):
        fk_history[i, buf_idx] = fk[i]
        fm_history[i, buf_idx] = fm[i]

    # =============== 3. Yc 历史项 ===============
    ik_yc_hist = _jit_compute_yc_history_term(
        pn_yc, qn_yc_3d, xk_2d, vk_c, n_yc, nc)
    im_yc_hist = _jit_compute_yc_history_term(
        pn_yc, qn_yc_3d, xm_2d, vm_c, n_yc, nc)

    # =============== 4. 合并 update_b(step) + compute_b(step+1) ===============
    bk_result = np.zeros(nc, dtype=np.complex128)
    bm_result = np.zeros(nc, dtype=np.complex128)

    for mode in range(nmod):
        # ---- 内联插值: 预计算共享索引 ----
        delay = tau_arr[mode] / dt
        delay_dt = (tau_arr[mode] + dt) / dt

        idx_lo = int(np.floor(delay))
        frac = delay - idx_lo
        one_m_frac = 1.0 - frac

        idx_lo_dt = int(np.floor(delay_dt))
        frac_dt = delay_dt - idx_lo_dt
        one_m_frac_dt = 1.0 - frac_dt

        # step S 的 4 个缓冲区位置
        s_lo   = (step - idx_lo + 1)     % history_length
        s_hi   = (step - idx_lo)         % history_length
        s_lo_d = (step - idx_lo_dt + 1)  % history_length
        s_hi_d = (step - idx_lo_dt)      % history_length

        # step S+1 的 4 个位置 (n_hi == s_lo, n_hi_d == s_lo_d)
        n_lo   = (next_step - idx_lo + 1)    % history_length
        n_lo_d = (next_step - idx_lo_dt + 1) % history_length

        # 对 nc 个分量直接内联插值,无临时数组分配
        fk_sum_curr = np.empty(nc, dtype=np.complex128)
        fm_sum_curr = np.empty(nc, dtype=np.complex128)
        fk_sum_next = np.empty(nc, dtype=np.complex128)
        fm_sum_next = np.empty(nc, dtype=np.complex128)

        for k in range(nc):
            # 共享读取: fk_history[k, s_lo] 同时用于 step-S 的 lo 和 step-N 的 hi
            fkh_s_lo = fk_history[k, s_lo]
            fkh_s_hi = fk_history[k, s_hi]
            fkh_n_lo = fk_history[k, n_lo]
            fkh_s_lo_d = fk_history[k, s_lo_d]
            fkh_s_hi_d = fk_history[k, s_hi_d]
            fkh_n_lo_d = fk_history[k, n_lo_d]

            fk_tau_s  = one_m_frac * fkh_s_lo + frac * fkh_s_hi
            fk_tau_n  = one_m_frac * fkh_n_lo + frac * fkh_s_lo      # hi of N == lo of S
            fk_tdt_s  = one_m_frac_dt * fkh_s_lo_d + frac_dt * fkh_s_hi_d
            fk_tdt_n  = one_m_frac_dt * fkh_n_lo_d + frac_dt * fkh_s_lo_d

            fk_sum_curr[k] = fk_tau_s + fk_tdt_s
            fk_sum_next[k] = fk_tau_n + fk_tdt_n

            fmh_s_lo = fm_history[k, s_lo]
            fmh_s_hi = fm_history[k, s_hi]
            fmh_n_lo = fm_history[k, n_lo]
            fmh_s_lo_d = fm_history[k, s_lo_d]
            fmh_s_hi_d = fm_history[k, s_hi_d]
            fmh_n_lo_d = fm_history[k, n_lo_d]

            fm_tau_s  = one_m_frac * fmh_s_lo + frac * fmh_s_hi
            fm_tau_n  = one_m_frac * fmh_n_lo + frac * fmh_s_lo
            fm_tdt_s  = one_m_frac_dt * fmh_s_lo_d + frac_dt * fmh_s_hi_d
            fm_tdt_n  = one_m_frac_dt * fmh_n_lo_d + frac_dt * fmh_s_lo_d

            fm_sum_curr[k] = fm_tau_s + fm_tdt_s
            fm_sum_next[k] = fm_tau_n + fm_tdt_n

        # ---- 单次遍历: update_b(step) + compute_b(step+1) ----
        n_h = n_h_poles_arr[mode]
        for i in range(n_h):
            pi_val = pi_h_2d[mode, i]
            for k in range(nc):
                b_old_k = bk_3d[mode, i, k]
                b_old_m = bm_3d[mode, i, k]
                qf_upd_k = 0.0 + 0.0j
                qf_cmp_k = 0.0 + 0.0j
                qf_upd_m = 0.0 + 0.0j
                qf_cmp_m = 0.0 + 0.0j

                for j in range(nc):
                    q_val = qij_h_4d[mode, i, k, j]   # 读一次,用四次
                    qf_upd_k += q_val * fk_sum_curr[j]
                    qf_cmp_k += q_val * fk_sum_next[j]
                    qf_upd_m += q_val * fm_sum_curr[j]
                    qf_cmp_m += q_val * fm_sum_next[j]

                # update_b: 写入新 b
                b_new_k = pi_val * b_old_k + qf_upd_k
                b_new_m = pi_val * b_old_m + qf_upd_m
                bk_3d[mode, i, k] = b_new_k
                bm_3d[mode, i, k] = b_new_m

                # compute_b: 用刚更新的 b 累加
                bk_result[k] += pi_val * b_new_k + qf_cmp_k
                bm_result[k] += pi_val * b_new_m + qf_cmp_m

    # =============== 5. 输出 ===============
    ik_hist_new = np.empty(nc, dtype=np.float64)
    im_hist_new = np.empty(nc, dtype=np.float64)
    ik_out = np.empty(nc, dtype=np.float64)
    im_out = np.empty(nc, dtype=np.float64)
    for i in range(nc):
        ik_hist_new[i] = (ik_yc_hist[i] - bk_result[i]).real
        im_hist_new[i] = (im_yc_hist[i] - bm_result[i]).real
        ik_out[i] = ik_c[i].real
        im_out[i] = im_c[i].real

    # 就地更新 prev (下次调用时用)
    for i in range(nc):
        vk_prev[i] = vk_c[i]
        vm_prev[i] = vm_c[i]

    return ik_hist_new, im_hist_new, ik_out, im_out


@njit(cache=True)
def _jit_simulate_loop(
    n_steps, nc, n_yc, nmod, max_n_h, history_length,
    n_h_poles_arr, dt,
    pn_yc, qn_yc_3d, yc_d, G,
    pi_h_2d, qij_h_4d, tau_arr,
    xk_2d, xm_2d, bk_3d, bm_3d,
    fk_history, fm_history,
    vs_k_all, vs_m_all,
    A_inv_k, A_inv_m,
    z_source_k, z_source_m,
    vk_out, vm_out, ik_out, im_out,
):
    """完全 JIT 化的仿真主循环。"""
    vk_prev = np.zeros(nc, dtype=np.complex128)
    vm_prev = np.zeros(nc, dtype=np.complex128)
    ik_prev = np.zeros(nc, dtype=np.complex128)
    im_prev = np.zeros(nc, dtype=np.complex128)

    for step in range(n_steps):
        # 1) 源电压
        vs_k = np.empty(nc, dtype=np.complex128)
        vs_m = np.empty(nc, dtype=np.complex128)
        for i in range(nc):
            vs_k[i] = vs_k_all[step, i]
            vs_m[i] = vs_m_all[step, i]

        # 2) 历史电流源
        ik_hist, im_hist = _jit_calculate_history_and_update(
            nc, n_yc, nmod, max_n_h, history_length,
            n_h_poles_arr,
            pn_yc, qn_yc_3d, yc_d,
            pi_h_2d, qij_h_4d, tau_arr, dt,
            xk_2d, xm_2d, bk_3d, bm_3d,
            fk_history, fm_history,
            vk_prev, vm_prev, ik_prev, im_prev,
            step,
        )

        # 3) 求解 v = A_inv @ (vs - Zs @ i_hist)
        rhs_k = np.empty(nc, dtype=np.float64)
        rhs_m = np.empty(nc, dtype=np.float64)
        for i in range(nc):
            zih_k = 0.0
            zih_m = 0.0
            for j in range(nc):
                zih_k += z_source_k[i, j] * ik_hist[j]
                zih_m += z_source_m[i, j] * im_hist[j]
            rhs_k[i] = vs_k[i].real - zih_k
            rhs_m[i] = vs_m[i].real - zih_m

        vk_curr_f = np.empty(nc, dtype=np.float64)
        vm_curr_f = np.empty(nc, dtype=np.float64)
        for i in range(nc):
            sk = 0.0
            sm = 0.0
            for j in range(nc):
                sk += A_inv_k[i, j] * rhs_k[j]
                sm += A_inv_m[i, j] * rhs_m[j]
            vk_curr_f[i] = sk
            vm_curr_f[i] = sm

        # 4) i = G @ v + i_hist
        ik_curr_f = np.empty(nc, dtype=np.float64)
        im_curr_f = np.empty(nc, dtype=np.float64)
        for i in range(nc):
            gv_k = 0.0
            gv_m = 0.0
            for j in range(nc):
                gv_k += G[i, j] * vk_curr_f[j]
                gv_m += G[i, j] * vm_curr_f[j]
            ik_curr_f[i] = gv_k + ik_hist[i]
            im_curr_f[i] = gv_m + im_hist[i]

        # 5) 更新状态
        vk_curr_c = np.empty(nc, dtype=np.complex128)
        vm_curr_c = np.empty(nc, dtype=np.complex128)
        ik_curr_c = np.empty(nc, dtype=np.complex128)
        im_curr_c = np.empty(nc, dtype=np.complex128)
        for i in range(nc):
            vk_curr_c[i] = vk_curr_f[i]
            vm_curr_c[i] = vm_curr_f[i]
            ik_curr_c[i] = ik_curr_f[i]
            im_curr_c[i] = im_curr_f[i]

        _jit_do_update_states(
            nc, n_yc, nmod, max_n_h, history_length,
            n_h_poles_arr,
            pn_yc, qn_yc_3d,
            pi_h_2d, qij_h_4d, tau_arr, dt,
            xk_2d, xm_2d, bk_3d, bm_3d,
            fk_history, fm_history,
            vk_curr_c, vm_curr_c,
            vk_prev, vm_prev,
            step,
        )

        # 6) 输出
        for i in range(nc):
            vk_out[i, step] = vk_curr_f[i]
            vm_out[i, step] = vm_curr_f[i]
            ik_out[i, step] = ik_curr_f[i]
            im_out[i, step] = im_curr_f[i]

        # 7) 推进 prev
        for i in range(nc):
            vk_prev[i] = vk_curr_c[i]
            vm_prev[i] = vm_curr_c[i]
            ik_prev[i] = ik_curr_c[i]
            im_prev[i] = im_curr_c[i]


# ===========================================================================
# 第三部分:ULM 核心模型
# ===========================================================================

class ULMModel:
    """ULM 核心模型(JIT 加速)。"""

    def __init__(
        self, fit_data: FitULMData, line_length: float, dt: float,
        verbose: bool = True,
    ):
        self.nc = fit_data.nf
        self.nmod = fit_data.nmod
        self.length = line_length
        self.dt = dt
        self.verbose = verbose

        self.yc_poles = fit_data.yc_poles
        self.yc_residues = fit_data.yc_residues
        self.yc_d = fit_data.yc_d
        self.h_poles = fit_data.h_poles
        self.h_residues = fit_data.h_residues
        self.tau = fit_data.time_delays

        self._compute_and_pack_constants()
        self._initialize_state()

    def _compute_and_pack_constants(self) -> None:
        """梯形递推常数预计算并打包。

        递推关系(梯形法):
            p_n = (2 + a_n·dt) / (2 - a_n·dt),
            q_n = R_n · dt / (2 - a_n·dt)
        电导矩阵: G = Re(k₀ + Σ q_n)
        """
        dt = self.dt
        nc = self.nc
        n_yc = len(self.yc_poles)

        # Yc
        self.pn_yc = np.zeros(n_yc, dtype=np.complex128)
        qn_yc_list: List[np.ndarray] = []
        for n in range(n_yc):
            an = self.yc_poles[n]
            denom = 2.0 - an * dt
            self.pn_yc[n] = (2.0 + an * dt) / denom
            qn_yc_list.append(self.yc_residues[n] * dt / denom)

        self.qn_yc_3d = np.zeros((n_yc, nc, nc), dtype=np.complex128)
        for n in range(n_yc):
            self.qn_yc_3d[n] = qn_yc_list[n]

        self.G = np.real(self.yc_d.copy())
        for q in qn_yc_list:
            self.G += np.real(q)

        self.n_yc = n_yc

        # H (padded)
        n_h_poles_list = [len(self.h_poles[mode]) for mode in range(self.nmod)]
        self.max_n_h = max(n_h_poles_list) if n_h_poles_list else 0
        self.n_h_poles_arr = np.array(n_h_poles_list, dtype=np.int64)

        self.pi_h_2d = np.zeros((self.nmod, self.max_n_h), dtype=np.complex128)
        self.qij_h_4d = np.zeros(
            (self.nmod, self.max_n_h, nc, nc), dtype=np.complex128
        )
        for mode in range(self.nmod):
            for i in range(n_h_poles_list[mode]):
                ai = self.h_poles[mode][i]
                denom = 2.0 - ai * dt
                self.pi_h_2d[mode, i] = (2.0 + ai * dt) / denom
                self.qij_h_4d[mode, i] = self.h_residues[mode][i] * dt / denom

        # 历史缓冲区
        max_delay = float(np.max(self.tau)) if len(self.tau) > 0 else 0.0
        self.history_length = int(np.ceil(max_delay / self.dt)) + 10

        # 预计算 complex128 版 yc_d，避免每步 astype 分配
        self._yc_d_c128 = self.yc_d.astype(np.complex128)

        self._check_stability()

        if self.verbose:
            logger.info(
                "ULMModel: nc=%d, nmod=%d, dt=%.3e s, max_tau=%.3e s, "
                "hist=%d, n_yc=%d, max_n_h=%d",
                self.nc, self.nmod, self.dt, max_delay,
                self.history_length, n_yc, self.max_n_h,
            )

    def _check_stability(self) -> None:
        """极点稳定性与步长合理性检查。"""
        warnings_list: List[str] = []
        for pole in self.yc_poles:
            if np.real(pole) > 0:
                warnings_list.append(f"Yc 不稳定极点: {pole}")
            if np.abs(pole) * self.dt > 0.5:
                warnings_list.append(f"dt 可能过大,极点 {pole}")
        for mode, poles in enumerate(self.h_poles):
            for pole in poles:
                if np.real(pole) > 0:
                    warnings_list.append(f"H[{mode}] 不稳定极点: {pole}")

        for w in warnings_list[:5]:
            logger.warning(w)
        if len(warnings_list) > 5:
            logger.warning("...还有 %d 个稳定性警告", len(warnings_list) - 5)

    def _initialize_state(self) -> None:
        """初始化所有状态数组。"""
        nc = self.nc
        self.xk_2d = np.zeros((self.n_yc, nc), dtype=np.complex128)
        self.xm_2d = np.zeros((self.n_yc, nc), dtype=np.complex128)

        self.bk_3d = np.zeros((self.nmod, self.max_n_h, nc), dtype=np.complex128)
        self.bm_3d = np.zeros((self.nmod, self.max_n_h, nc), dtype=np.complex128)

        self.fk_history = np.zeros((nc, self.history_length), dtype=np.complex128)
        self.fm_history = np.zeros((nc, self.history_length), dtype=np.complex128)

        self.vk_prev = np.zeros(nc, dtype=np.complex128)
        self.vm_prev = np.zeros(nc, dtype=np.complex128)

        self.current_step = 0

        # 预分配 calculate_history / update_states 的输入缓冲区
        self._buf_vk = np.zeros(nc, dtype=np.complex128)
        self._buf_vm = np.zeros(nc, dtype=np.complex128)
        self._buf_ik = np.zeros(nc, dtype=np.complex128)
        self._buf_im = np.zeros(nc, dtype=np.complex128)
        self._buf_vk_curr = np.zeros(nc, dtype=np.complex128)
        self._buf_vm_curr = np.zeros(nc, dtype=np.complex128)

    def reset(self) -> None:
        self._initialize_state()

    def calculate_history(
        self, vk_prev, vm_prev, ik_prev, im_prev, step,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算当前步的历史电流源,并存储 f(t-Δt)。

        优化: 使用预分配缓冲区避免每步创建临时数组。
        """
        # 就地写入预分配缓冲区（避免 np.asarray + ravel 分配）
        self._buf_vk[:] = vk_prev
        self._buf_vm[:] = vm_prev
        self._buf_ik[:] = ik_prev
        self._buf_im[:] = im_prev

        self.current_step = step
        self._vk_prev_temp = self._buf_vk  # JIT 只读,无需 copy
        self._vm_prev_temp = self._buf_vm

        ik_hist, im_hist = _jit_calculate_history_and_update(
            self.nc, self.n_yc, self.nmod, self.max_n_h, self.history_length,
            self.n_h_poles_arr,
            self.pn_yc, self.qn_yc_3d, self._yc_d_c128,
            self.pi_h_2d, self.qij_h_4d, self.tau, self.dt,
            self.xk_2d, self.xm_2d, self.bk_3d, self.bm_3d,
            self.fk_history, self.fm_history,
            self._buf_vk, self._buf_vm, self._buf_ik, self._buf_im,
            step,
        )
        return ik_hist, im_hist  # 已经是 (nc,) float64,无需 reshape

    def update_states(self, vk_curr, vm_curr, ik_curr, im_curr) -> None:
        """求解完成后更新内部状态。

        优化: 使用预分配缓冲区避免每步 asarray + copy。
        """
        self._buf_vk_curr[:] = vk_curr
        self._buf_vm_curr[:] = vm_curr

        _jit_do_update_states(
            self.nc, self.n_yc, self.nmod, self.max_n_h, self.history_length,
            self.n_h_poles_arr,
            self.pn_yc, self.qn_yc_3d,
            self.pi_h_2d, self.qij_h_4d, self.tau, self.dt,
            self.xk_2d, self.xm_2d, self.bk_3d, self.bm_3d,
            self.fk_history, self.fm_history,
            self._buf_vk_curr, self._buf_vm_curr,
            self._vk_prev_temp, self._vm_prev_temp,
            self.current_step,
        )

        self.vk_prev[:] = self._buf_vk_curr
        self.vm_prev[:] = self._buf_vm_curr

    def full_step(
        self, vk_real, vm_real, I_hist_k, I_hist_m, G_real, step,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """单次 JIT 调用完成全部 ULM 计算 (G@v + update + history)。

        Returns (ik_hist_new, im_hist_new, ik_real, im_real), all float64 (nc,)。
        """
        return _jit_full_step(
            self.nc, self.n_yc, self.nmod, self.max_n_h, self.history_length,
            self.n_h_poles_arr,
            self.pn_yc, self.qn_yc_3d, self._yc_d_c128,
            self.pi_h_2d, self.qij_h_4d, self.tau, self.dt,
            self.xk_2d, self.xm_2d, self.bk_3d, self.bm_3d,
            self.fk_history, self.fm_history,
            G_real, vk_real, vm_real, I_hist_k, I_hist_m,
            self.vk_prev, self.vm_prev,
            step,
        )

    def get_conductance_matrix(self) -> np.ndarray:
        return self.G

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'nc':          self.nc,
            'nmod':        self.nmod,
            'dt':          self.dt,
            'length':      self.length,
            'time_delays': self.tau.tolist(),
            'n_yc_poles':  len(self.yc_poles),
            'n_h_poles':   [len(p) for p in self.h_poles],
            'G_diagonal':  np.diag(self.G).tolist(),
        }


# ===========================================================================
# 第四部分:暂态仿真器
# ===========================================================================

class TransientSimulator:
    """ULM 暂态仿真器,核心循环在 JIT 函数中执行。"""

    def __init__(self, model: ULMModel):
        self.model = model
        self.nc = model.nc
        self.dt = model.dt

    def simulate(
        self, t_end: float,
        source_k: Callable[[float], np.ndarray],
        source_m: Callable[[float], np.ndarray],
        z_source_k: np.ndarray,
        z_source_m: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """运行暂态仿真。

        预计算所有源电压以及求解矩阵 A_inv = (I + Zs·G)⁻¹,
        主循环在 @njit 函数内执行。
        """
        n_steps = int(t_end / self.dt) + 1
        nc = self.nc
        model = self.model

        # 预计算源电压
        vs_k_all = np.zeros((n_steps, nc), dtype=np.float64)
        vs_m_all = np.zeros((n_steps, nc), dtype=np.float64)
        for step in range(n_steps):
            t = step * self.dt
            vs_k_all[step] = source_k(t).ravel()
            vs_m_all[step] = source_m(t).ravel()

        # 预计算求解矩阵
        G = model.G
        z_k = np.asarray(z_source_k, dtype=np.float64)
        z_m = np.asarray(z_source_m, dtype=np.float64)

        A_k = np.eye(nc) + z_k @ G
        A_m = np.eye(nc) + z_m @ G

        try:
            A_inv_k = np.linalg.inv(A_k)
        except np.linalg.LinAlgError:
            A_inv_k = np.linalg.pinv(A_k)
        try:
            A_inv_m = np.linalg.inv(A_m)
        except np.linalg.LinAlgError:
            A_inv_m = np.linalg.pinv(A_m)

        # 分配输出
        vk_out = np.zeros((nc, n_steps), dtype=np.float64)
        vm_out = np.zeros((nc, n_steps), dtype=np.float64)
        ik_out = np.zeros((nc, n_steps), dtype=np.float64)
        im_out = np.zeros((nc, n_steps), dtype=np.float64)

        model.reset()

        if verbose:
            logger.info("开始仿真(JIT): %d 步, dt=%.3e s", n_steps, self.dt)

        _jit_simulate_loop(
            n_steps, nc, model.n_yc, model.nmod, model.max_n_h,
            model.history_length, model.n_h_poles_arr,
            model.dt,
            model.pn_yc, model.qn_yc_3d,
            model._yc_d_c128, G,
            model.pi_h_2d, model.qij_h_4d, model.tau,
            model.xk_2d, model.xm_2d,
            model.bk_3d, model.bm_3d,
            model.fk_history, model.fm_history,
            vs_k_all, vs_m_all,
            A_inv_k, A_inv_m,
            z_k, z_m,
            vk_out, vm_out, ik_out, im_out,
        )

        return {
            'time': np.arange(n_steps) * self.dt,
            'vk': vk_out, 'vm': vm_out,
            'ik': ik_out, 'im': im_out,
        }


# ===========================================================================
# 第五部分:ULMLine (求解器接口)
# ===========================================================================

class ULMLine:
    """ULM 传输线模型(兼容单相/多相 & EMTPSolver)。"""

    def __init__(
        self, name: str, ulm_model: ULMModel,
        node_k: int = 0, node_m: int = 0,
    ):
        self._name = name
        self._ulm = ulm_model

        self.node_k = node_k
        self.node_m = node_m

        self._G_matrix = self._ulm.get_conductance_matrix()
        self._G_real = np.ascontiguousarray(self._G_matrix, dtype=np.float64)
        self._is_multiphase = (ulm_model.nc > 1)

        self.initialize(ulm_model.dt)
        self._compute_equiv_params()

    def _compute_equiv_params(self) -> None:
        """从 G_eq 和 time_delays 计算等效 Zc 与 τ。"""
        G_eq = self.G_eq
        if isinstance(G_eq, np.ndarray):
            G_diag = np.diag(G_eq) if G_eq.ndim == 2 else G_eq.flatten()
            G_avg = np.mean(G_diag[G_diag > 0]) if np.any(G_diag > 0) else 1e-6
            self.Zc = 1.0 / G_avg
        elif G_eq > 0:
            self.Zc = 1.0 / G_eq
        else:
            self.Zc = 300.0

        tau_arr = self._ulm.tau
        if len(tau_arr) > 0:
            self.tau = float(tau_arr[0])
        else:
            self.tau = self._ulm.length / 3e8

    @property
    def name(self) -> str:
        return self._name

    @property
    def nc(self) -> int:
        return self._ulm.nc

    @property
    def G_eq(self) -> Union[float, np.ndarray]:
        if self._is_multiphase:
            return self._G_matrix
        return float(np.real(self._G_matrix[0, 0]))

    @property
    def I_hist_k(self) -> Union[float, np.ndarray]:
        if self._is_multiphase:
            return self._I_hist_k_vec
        return float(self._I_hist_k_vec[self._sp_idx])

    @property
    def I_hist_m(self) -> Union[float, np.ndarray]:
        if self._is_multiphase:
            return self._I_hist_m_vec
        return float(self._I_hist_m_vec[self._sp_idx])

    def initialize(self, dt: float) -> None:
        self._ulm.reset()
        self._step = 0

        nc = self.nc
        self._I_hist_k_vec = np.zeros(nc)
        self._I_hist_m_vec = np.zeros(nc)
        self._V_k_vec_prev = np.zeros(nc)
        self._V_m_vec_prev = np.zeros(nc)
        self._I_k_vec_prev = np.zeros(nc)
        self._I_m_vec_prev = np.zeros(nc)

        self._I_hist_k_scalar = 0.0
        self._I_hist_m_scalar = 0.0

        # 预分配历史数组（初始容量，按需扩展）
        self._hist_capacity = 1024
        self._hist_count = 0
        if self._is_multiphase:
            self._I_k_hist_buf = np.zeros((self._hist_capacity, nc))
            self._I_m_hist_buf = np.zeros((self._hist_capacity, nc))
            self._V_k_hist_buf = np.zeros((self._hist_capacity, nc))
            self._V_m_hist_buf = np.zeros((self._hist_capacity, nc))
        else:
            self._I_k_hist_buf = np.zeros(self._hist_capacity)
            self._I_m_hist_buf = np.zeros(self._hist_capacity)
            self._V_k_hist_buf = np.zeros(self._hist_capacity)
            self._V_m_hist_buf = np.zeros(self._hist_capacity)

        # 预分配 update_state / full_step 的临时向量
        if self._is_multiphase:
            self._vk_work = np.zeros(nc)
            self._vm_work = np.zeros(nc)
        else:
            # 单相模式也预分配，避免每步 np.zeros(nc)
            self._vk_work_sp = np.zeros(nc)
            self._vm_work_sp = np.zeros(nc)
            self._sp_idx = 0

    def update_history_sources(self) -> None:
        ik_hist, im_hist = self._ulm.calculate_history(
            self._V_k_vec_prev,
            self._V_m_vec_prev,
            self._I_k_vec_prev,
            self._I_m_vec_prev,
            self._step,
        )

        self._I_hist_k_vec[:self.nc] = ik_hist
        self._I_hist_m_vec[:self.nc] = im_hist

        if not self._is_multiphase:
            idx = self._sp_idx
            self._I_hist_k_scalar = float(self._I_hist_k_vec[idx])
            self._I_hist_m_scalar = float(self._I_hist_m_vec[idx])

    def _ensure_hist_capacity(self) -> None:
        """按需倍增历史缓冲区容量。"""
        if self._hist_count >= self._hist_capacity:
            new_cap = self._hist_capacity * 2
            nc = self.nc
            if self._is_multiphase:
                for attr in ('_I_k_hist_buf', '_I_m_hist_buf',
                             '_V_k_hist_buf', '_V_m_hist_buf'):
                    old = getattr(self, attr)
                    new = np.zeros((new_cap, nc))
                    new[:self._hist_capacity] = old
                    setattr(self, attr, new)
            else:
                for attr in ('_I_k_hist_buf', '_I_m_hist_buf',
                             '_V_k_hist_buf', '_V_m_hist_buf'):
                    old = getattr(self, attr)
                    new = np.zeros(new_cap)
                    new[:self._hist_capacity] = old
                    setattr(self, attr, new)
            self._hist_capacity = new_cap

    def update_state(
        self, V_k: Union[float, np.ndarray],
        V_m: Union[float, np.ndarray],
    ) -> None:
        nc = self.nc
        if self._is_multiphase:
            vk_vec = self._vk_work
            vm_vec = self._vm_work
            vk_vec[:] = V_k
            vm_vec[:] = V_m
        else:
            vk_vec = self._vk_work_sp
            vm_vec = self._vm_work_sp
            vk_vec[:] = 0.0
            vm_vec[:] = 0.0
            idx = self._sp_idx
            vk_vec[idx] = float(V_k)
            vm_vec[idx] = float(V_m)

        ik_vec = self._G_matrix @ vk_vec + self._I_hist_k_vec
        im_vec = self._G_matrix @ vm_vec + self._I_hist_m_vec

        self._ulm.update_states(vk_vec, vm_vec, ik_vec, im_vec)

        self._V_k_vec_prev[:] = vk_vec
        self._V_m_vec_prev[:] = vm_vec
        self._I_k_vec_prev[:] = ik_vec
        self._I_m_vec_prev[:] = im_vec

        # 写入预分配历史缓冲区
        self._ensure_hist_capacity()
        idx_h = self._hist_count
        if self._is_multiphase:
            self._V_k_hist_buf[idx_h] = vk_vec
            self._V_m_hist_buf[idx_h] = vm_vec
            self._I_k_hist_buf[idx_h] = ik_vec
            self._I_m_hist_buf[idx_h] = im_vec
        else:
            idx = self._sp_idx
            self._V_k_hist_buf[idx_h] = vk_vec[idx]
            self._V_m_hist_buf[idx_h] = vm_vec[idx]
            self._I_k_hist_buf[idx_h] = ik_vec[idx]
            self._I_m_hist_buf[idx_h] = im_vec[idx]
        self._hist_count += 1

        self._step += 1

    def full_step(
        self, V_k: Union[float, np.ndarray],
        V_m: Union[float, np.ndarray],
        record_history: bool = True,
    ) -> None:
        """单次 JIT 调用完成 G@v + update_states + calculate_history。

        消除:
        - 2 次 numpy matmul (G @ v)
        - 2 次 Python→JIT 过渡
        - 8×nmod 次临时数组分配 (内联插值)
        """
        nc = self.nc
        if self._is_multiphase:
            vk = self._vk_work
            vm = self._vm_work
            vk[:] = V_k
            vm[:] = V_m
        else:
            vk = self._vk_work_sp
            vm = self._vm_work_sp
            vk[:] = 0.0
            vm[:] = 0.0
            idx = self._sp_idx
            vk[idx] = float(V_k)
            vm[idx] = float(V_m)

        # 单次 JIT 调用: G@v + update + history
        ik_hist, im_hist, ik_real, im_real = self._ulm.full_step(
            vk, vm,
            self._I_hist_k_vec, self._I_hist_m_vec,
            self._G_real,
            self._step,
        )

        # 更新状态
        self._I_hist_k_vec[:self.nc] = ik_hist
        self._I_hist_m_vec[:self.nc] = im_hist
        self._V_k_vec_prev[:] = vk
        self._V_m_vec_prev[:] = vm
        self._I_k_vec_prev[:] = ik_real
        self._I_m_vec_prev[:] = im_real

        if not self._is_multiphase:
            idx = self._sp_idx
            self._I_hist_k_scalar = ik_hist[idx]
            self._I_hist_m_scalar = im_hist[idx]

        # 记录线路对象历史是可选的。EMTPSolver 已保存节点电压结果；
        # 在长仿真中每步记录所有线路端口历史会造成明显内存写入和 Python 开销。
        if record_history:
            self._ensure_hist_capacity()
            idx_h = self._hist_count
            if self._is_multiphase:
                self._V_k_hist_buf[idx_h] = vk
                self._V_m_hist_buf[idx_h] = vm
                self._I_k_hist_buf[idx_h] = ik_real
                self._I_m_hist_buf[idx_h] = im_real
            else:
                idx = self._sp_idx
                self._V_k_hist_buf[idx_h] = vk[idx]
                self._V_m_hist_buf[idx_h] = vm[idx]
                self._I_k_hist_buf[idx_h] = ik_real[idx]
                self._I_m_hist_buf[idx_h] = im_real[idx]
            self._hist_count += 1
        self._step += 1

    @property
    def V_k_history(self):
        return self._V_k_hist_buf[:self._hist_count]

    @property
    def V_m_history(self):
        return self._V_m_hist_buf[:self._hist_count]

    @property
    def I_k_history(self):
        return self._I_k_hist_buf[:self._hist_count]

    @property
    def I_m_history(self):
        return self._I_m_hist_buf[:self._hist_count]

    def get_currents(self) -> Tuple[Any, Any]:
        if self._is_multiphase:
            return self._I_k_vec_prev.copy(), self._I_m_vec_prev.copy()
        return self._I_k_vec_prev[0], self._I_m_vec_prev[0]

    def get_info(self) -> Dict[str, Any]:
        info = self._ulm.get_model_info()
        info['name'] = self._name
        info['is_multiphase'] = self._is_multiphase
        info.setdefault('Zc', self.Zc)
        info.setdefault('tau', self.tau)
        info['nc'] = self.nc
        return info

    @classmethod
    def create_from_fitulm(
        cls, name: str, node_k: int, node_m: int,
        fitulm_path: str, line_length: float, dt: float,
        verbose: bool = True,
    ) -> 'ULMLine':
        fit_data = FitULMReader(fitulm_path).read(verbose=verbose)
        ulm_model = ULMModel(fit_data, line_length, dt, verbose)
        if ulm_model.nc != 1:
            raise ValueError(
                f"create_from_fitulm() 只用于单相 ULM 数据,当前数据为 {ulm_model.nc} 相；"
                "多相线路请直接使用 ULMLine(name, ulm_model) 并设置 nodes_k/nodes_m"
            )
        return cls(name, ulm_model, node_k, node_m)

    @classmethod
    def create_from_data(
        cls, name: str, node_k: int, node_m: int,
        fit_data: FitULMData, line_length: float, dt: float,
        verbose: bool = True,
    ) -> 'ULMLine':
        ulm_model = ULMModel(fit_data, line_length, dt, verbose)
        if ulm_model.nc != 1:
            raise ValueError(
                f"create_from_data() 只用于单相 ULM 数据,当前数据为 {ulm_model.nc} 相；"
                "多相线路请直接使用 ULMLine(name, ulm_model) 并设置 nodes_k/nodes_m"
            )
        return cls(name, ulm_model, node_k, node_m)

    @classmethod
    def create_test(
        cls, name: str, node_k: int, node_m: int, dt: float = 1e-6,
        nf: int = 1, n_poles: int = 4, line_length: float = 10000.0,
        verbose: bool = False,
    ) -> 'ULMLine':
        fit_data = create_test_fitulm_data(nf=nf, n_poles=n_poles)
        return cls.create_from_data(
            name, node_k, node_m, fit_data, line_length, dt, verbose,
        )


# ===========================================================================
# 第六部分:便捷函数
# ===========================================================================

def load_ulm_model(
    fitulm_path: str, line_length: float, dt: float,
    verbose: bool = True, precision: int = 15,
) -> ULMModel:
    """从 fitULM 文件加载 ULM 模型。"""
    fit_data = FitULMReader(fitulm_path).read(verbose=verbose, precision=precision)
    return ULMModel(fit_data, line_length, dt, verbose)


def simulate_step_response(
    model: ULMModel, t_end: float,
    excited_phase: int = 0,
    source_amplitude: float = 1.0,
    z_source: float = 1e-6,
    open_end: bool = True,
) -> Dict[str, np.ndarray]:
    """单端阶跃响应计算。"""
    nc = model.nc

    def source_k(t: float) -> np.ndarray:
        v = np.zeros(nc)
        if t >= 0:
            v[excited_phase] = source_amplitude
        return v

    def source_m(t: float) -> np.ndarray:
        return np.zeros(nc)

    z_k = np.zeros((nc, nc))
    z_k[excited_phase, excited_phase] = z_source
    z_m = np.eye(nc) * 1e10 if open_end else np.zeros((nc, nc))

    return TransientSimulator(model).simulate(t_end, source_k, source_m, z_k, z_m)


def plot_results(
    results: Dict[str, np.ndarray],
    phases: Optional[List[int]] = None,
    title: str = "ULM 暂态仿真结果",
    save_path: Optional[str] = None,
):
    """绘制仿真结果(matplotlib 可选依赖)。"""
    import matplotlib.pyplot as plt

    time_us = results['time'] * 1e6
    nc = results['vk'].shape[0]
    if phases is None:
        phases = list(range(min(nc, 6)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    datasets = [
        (axes[0, 0], results['vk'],         1.0,  '电压 (V)',  '发送端电压'),
        (axes[0, 1], results['vm'],         1.0,  '电压 (V)',  '接收端电压'),
        (axes[1, 0], results['ik'] * 1e3,   1.0,  '电流 (mA)', '发送端电流'),
        (axes[1, 1], results['im'] * 1e3,   1.0,  '电流 (mA)', '接收端电流'),
    ]

    for ax, data, _scale, ylabel, subtitle in datasets:
        for i in phases:
            ax.plot(time_us, data[i, :], label=f'相 {i + 1}')
        ax.set_xlabel('时间 (μs)')
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("图像已保存到: %s", save_path)
    return fig


def warmup_jit() -> None:
    """预热 JIT 编译,避免首次仿真时的编译延迟。"""
    fit_data = create_test_fitulm_data(nf=1, n_poles=2)
    model = ULMModel(fit_data, 1000.0, 1e-6, verbose=False)
    TransientSimulator(model).simulate(
        t_end=5e-6,
        source_k=lambda t: np.array([1.0]) if t >= 0 else np.array([0.0]),
        source_m=lambda t: np.array([0.0]),
        z_source_k=np.array([[1.0]]),
        z_source_m=np.array([[1e10]]),
        verbose=False,
    )


# ===========================================================================
# 第七部分:批量并行 kernel（多条 ULM 线路同步并行）
# ===========================================================================
#
# 设计说明
# --------
# _jit_full_step 只处理一条线路，若在 Python 层用 for line in lines 逐条调用，
# Python↔JIT 往返 = n_lines 次，且完全串行。
#
# _jit_batch_full_step 把 n_lines 条线路打包进同一次 kernel 调用，
# 用 numba.prange 在线路维度做并行：
#   - Python 调用开销：1 次（而非 n_lines 次）
#   - 线路内循环（mode, pole, nc）由 Numba 串行展开（与原来相同）
#   - 线路间并行：n_lines 条线路由线程池分配，真正并发执行
#
# 数据布局（各 padded 到 max_* 维度）
# -------
#   所有"per-line"数组的第 0 维 = line_idx，长度 = n_lines
#   例: pn_yc_batch[L, n_yc]   xk_batch[L, n_yc, nc]   etc.
#
# 调用者（EMTPSolver）负责：
#   1. 在 run() 时调用 build_ulm_batch_pack() 一次性打包所有 ULM 线路
#   2. 每步用 step_ulm_batch() 代替 _update_lines_combined 内的 ULM 部分
#
# 注意：nc 必须对所有线路相同（或 pad 到同一最大 nc）；
#        若线路 nc 不一致，调用者按最大 nc pad 并传入各线路实际 nc_arr。
# ---------------------------------------------------------------------------

from numba import prange, set_num_threads, get_num_threads


@njit(parallel=True, cache=True)
def _jit_batch_full_step(
    n_lines,
    nc_arr,          # (n_lines,) int64 — 各线路实际导体数
    max_nc,          # int — pad 后的最大 nc
    n_yc_arr,        # (n_lines,) int64
    max_n_yc,
    nmod_arr,        # (n_lines,) int64
    max_nmod,
    max_n_h_arr,     # (n_lines,) int64
    global_max_n_h,
    history_length_arr,  # (n_lines,) int64
    max_hist_len,
    n_h_poles_batch,     # (n_lines, max_nmod) int64
    dt_arr,              # (n_lines,) float64
    pn_yc_batch,         # (n_lines, max_n_yc) complex128
    qn_yc_batch,         # (n_lines, max_n_yc, max_nc, max_nc) complex128
    yc_d_batch,          # (n_lines, max_nc, max_nc) complex128
    pi_h_batch,          # (n_lines, max_nmod, global_max_n_h) complex128
    qij_h_batch,         # (n_lines, max_nmod, global_max_n_h, max_nc, max_nc) complex128
    tau_batch,           # (n_lines, max_nmod) float64
    # 状态数组（就地更新）
    xk_batch,            # (n_lines, max_n_yc, max_nc) complex128
    xm_batch,            # (n_lines, max_n_yc, max_nc) complex128
    bk_batch,            # (n_lines, max_nmod, global_max_n_h, max_nc) complex128
    bm_batch,            # (n_lines, max_nmod, global_max_n_h, max_nc) complex128
    fk_hist_batch,       # (n_lines, max_nc, max_hist_len) complex128
    fm_hist_batch,       # (n_lines, max_nc, max_hist_len) complex128
    vk_prev_batch,       # (n_lines, max_nc) complex128  (就地更新)
    vm_prev_batch,       # (n_lines, max_nc) complex128  (就地更新)
    # 输入（当前步）
    G_batch,             # (n_lines, max_nc, max_nc) float64
    vk_in_batch,         # (n_lines, max_nc) float64
    vm_in_batch,         # (n_lines, max_nc) float64
    I_hist_k_batch,      # (n_lines, max_nc) float64  (当前 I_hist，就地更新为下步值)
    I_hist_m_batch,      # (n_lines, max_nc) float64
    step_arr,            # (n_lines,) int64
    # 输出
    ik_out_batch,        # (n_lines, max_nc) float64
    im_out_batch,        # (n_lines, max_nc) float64
):
    """批量并行执行 n_lines 条 ULM 线路的 full_step。

    prange 在线路维度并行，每条线路的内部计算与 _jit_full_step 完全等价。
    各线路间无数据依赖，天然适合并行。
    """
    for L in prange(n_lines):  # <-- 真正的并行循环
        nc      = nc_arr[L]
        n_yc    = n_yc_arr[L]
        nmod    = nmod_arr[L]
        max_n_h = max_n_h_arr[L]
        hist_len = history_length_arr[L]
        dt      = dt_arr[L]
        step    = step_arr[L]
        next_step = step + 1

        # ---- 0. i = G·v + I_hist（内联，避免 BLAS 调用）----
        vk_c = np.empty(nc, dtype=np.complex128)
        vm_c = np.empty(nc, dtype=np.complex128)
        ik_c = np.empty(nc, dtype=np.complex128)
        im_c = np.empty(nc, dtype=np.complex128)
        for i in range(nc):
            vk_c[i] = vk_in_batch[L, i]
            vm_c[i] = vm_in_batch[L, i]
            gvk = I_hist_k_batch[L, i]
            gvm = I_hist_m_batch[L, i]
            for j in range(nc):
                gvk += G_batch[L, i, j] * vk_in_batch[L, j]
                gvm += G_batch[L, i, j] * vm_in_batch[L, j]
            ik_c[i] = gvk
            im_c[i] = gvm

        # ---- 1. 更新 x 状态 ----
        v_sum = np.empty(nc, dtype=np.complex128)
        # xk: v_curr=vk_c, v_prev=vk_prev_batch[L]
        for i in range(nc):
            v_sum[i] = vk_c[i] + vk_prev_batch[L, i]
        for n in range(n_yc):
            pn = pn_yc_batch[L, n]
            for i in range(nc):
                qv = 0.0 + 0.0j
                for j in range(nc):
                    qv += qn_yc_batch[L, n, i, j] * v_sum[j]
                xk_batch[L, n, i] = pn * xk_batch[L, n, i] + qv
        # xm: v_curr=vm_c, v_prev=vm_prev_batch[L]
        for i in range(nc):
            v_sum[i] = vm_c[i] + vm_prev_batch[L, i]
        for n in range(n_yc):
            pn = pn_yc_batch[L, n]
            for i in range(nc):
                qv = 0.0 + 0.0j
                for j in range(nc):
                    qv += qn_yc_batch[L, n, i, j] * v_sum[j]
                xm_batch[L, n, i] = pn * xm_batch[L, n, i] + qv

        # ---- 2. 存储 f(next_step) ----
        buf_idx = next_step % hist_len
        # fk = im_c + yc_d @ vm_c + Σ xm
        # fm = ik_c + yc_d @ vk_c + Σ xk
        for i in range(nc):
            k0_fk = 0.0 + 0.0j
            k0_fm = 0.0 + 0.0j
            for j in range(nc):
                k0_fk += yc_d_batch[L, i, j] * vm_c[j]
                k0_fm += yc_d_batch[L, i, j] * vk_c[j]
            xm_sum = 0.0 + 0.0j
            xk_sum = 0.0 + 0.0j
            for n in range(n_yc):
                xm_sum += xm_batch[L, n, i]
                xk_sum += xk_batch[L, n, i]
            fk_hist_batch[L, i, buf_idx] = im_c[i] + k0_fk + xm_sum
            fm_hist_batch[L, i, buf_idx] = ik_c[i] + k0_fm + xk_sum

        # ---- 3. Yc 历史项 ----
        ik_yc_hist = np.zeros(nc, dtype=np.complex128)
        im_yc_hist = np.zeros(nc, dtype=np.complex128)
        for n in range(n_yc):
            pn = pn_yc_batch[L, n]
            for i in range(nc):
                # ik: pn*xk + qn@vk_c
                term_k = pn * xk_batch[L, n, i]
                term_m = pn * xm_batch[L, n, i]
                qvk = 0.0 + 0.0j
                qvm = 0.0 + 0.0j
                for j in range(nc):
                    q = qn_yc_batch[L, n, i, j]
                    qvk += q * vk_c[j]
                    qvm += q * vm_c[j]
                ik_yc_hist[i] += term_k + qvk
                im_yc_hist[i] += term_m + qvm

        # ---- 4. 合并 update_b(step) + compute_b(next_step) ----
        bk_result = np.zeros(nc, dtype=np.complex128)
        bm_result = np.zeros(nc, dtype=np.complex128)

        for mode in range(nmod):
            tau_m = tau_batch[L, mode]
            delay    = tau_m / dt
            delay_dt = (tau_m + dt) / dt

            idx_lo    = int(np.floor(delay))
            frac      = delay - idx_lo
            one_m_frac = 1.0 - frac

            idx_lo_dt  = int(np.floor(delay_dt))
            frac_dt    = delay_dt - idx_lo_dt
            one_m_frac_dt = 1.0 - frac_dt

            s_lo   = (step - idx_lo + 1)    % hist_len
            s_hi   = (step - idx_lo)         % hist_len
            s_lo_d = (step - idx_lo_dt + 1)  % hist_len
            s_hi_d = (step - idx_lo_dt)      % hist_len
            n_lo   = (next_step - idx_lo + 1)    % hist_len
            n_lo_d = (next_step - idx_lo_dt + 1) % hist_len

            fk_sum_curr = np.empty(nc, dtype=np.complex128)
            fm_sum_curr = np.empty(nc, dtype=np.complex128)
            fk_sum_next = np.empty(nc, dtype=np.complex128)
            fm_sum_next = np.empty(nc, dtype=np.complex128)

            for k in range(nc):
                fkh_s_lo  = fk_hist_batch[L, k, s_lo]
                fkh_s_hi  = fk_hist_batch[L, k, s_hi]
                fkh_n_lo  = fk_hist_batch[L, k, n_lo]
                fkh_s_lo_d = fk_hist_batch[L, k, s_lo_d]
                fkh_s_hi_d = fk_hist_batch[L, k, s_hi_d]
                fkh_n_lo_d = fk_hist_batch[L, k, n_lo_d]

                fk_sum_curr[k] = (one_m_frac    * fkh_s_lo   + frac    * fkh_s_hi +
                                  one_m_frac_dt * fkh_s_lo_d + frac_dt * fkh_s_hi_d)
                fk_sum_next[k] = (one_m_frac    * fkh_n_lo   + frac    * fkh_s_lo +
                                  one_m_frac_dt * fkh_n_lo_d + frac_dt * fkh_s_lo_d)

                fmh_s_lo  = fm_hist_batch[L, k, s_lo]
                fmh_s_hi  = fm_hist_batch[L, k, s_hi]
                fmh_n_lo  = fm_hist_batch[L, k, n_lo]
                fmh_s_lo_d = fm_hist_batch[L, k, s_lo_d]
                fmh_s_hi_d = fm_hist_batch[L, k, s_hi_d]
                fmh_n_lo_d = fm_hist_batch[L, k, n_lo_d]

                fm_sum_curr[k] = (one_m_frac    * fmh_s_lo   + frac    * fmh_s_hi +
                                  one_m_frac_dt * fmh_s_lo_d + frac_dt * fmh_s_hi_d)
                fm_sum_next[k] = (one_m_frac    * fmh_n_lo   + frac    * fmh_s_lo +
                                  one_m_frac_dt * fmh_n_lo_d + frac_dt * fmh_s_lo_d)

            n_h = n_h_poles_batch[L, mode]
            for i in range(n_h):
                pi_val = pi_h_batch[L, mode, i]
                for k in range(nc):
                    b_old_k = bk_batch[L, mode, i, k]
                    b_old_m = bm_batch[L, mode, i, k]
                    qf_upd_k = 0.0 + 0.0j
                    qf_cmp_k = 0.0 + 0.0j
                    qf_upd_m = 0.0 + 0.0j
                    qf_cmp_m = 0.0 + 0.0j
                    for j in range(nc):
                        q_val = qij_h_batch[L, mode, i, k, j]
                        qf_upd_k += q_val * fk_sum_curr[j]
                        qf_cmp_k += q_val * fk_sum_next[j]
                        qf_upd_m += q_val * fm_sum_curr[j]
                        qf_cmp_m += q_val * fm_sum_next[j]

                    b_new_k = pi_val * b_old_k + qf_upd_k
                    b_new_m = pi_val * b_old_m + qf_upd_m
                    bk_batch[L, mode, i, k] = b_new_k
                    bm_batch[L, mode, i, k] = b_new_m

                    bk_result[k] += pi_val * b_new_k + qf_cmp_k
                    bm_result[k] += pi_val * b_new_m + qf_cmp_m

        # ---- 5. 输出 ----
        for i in range(nc):
            I_hist_k_batch[L, i] = (ik_yc_hist[i] - bk_result[i]).real
            I_hist_m_batch[L, i] = (im_yc_hist[i] - bm_result[i]).real
            ik_out_batch[L, i]   = ik_c[i].real
            im_out_batch[L, i]   = im_c[i].real
            vk_prev_batch[L, i]  = vk_c[i]
            vm_prev_batch[L, i]  = vm_c[i]


# Serial/parallel dispatchers for batch ULM stepping.
#
# _jit_batch_full_step is kept as the original parallel dispatcher for backward
# compatibility and for run_parallel_diagnostics().
# The serial dispatcher reuses the exact same Python function body, but compiles
# it without parallel=True. This makes the benchmark Config C a first-class mode
# instead of a monkey patch.
_jit_batch_full_step_parallel = _jit_batch_full_step
_jit_batch_full_step_serial = njit(cache=True)(_jit_batch_full_step.py_func)


class ULMBatchPack:
    """把多条 ULMLine 打包为可送入 _jit_batch_full_step 的连续数组。

    使用方法
    --------
    在 EMTPSolver.run() 开始时（传输线初始化后）调用一次::

        self._ulm_batch = ULMBatchPack(ulm_lines)

    每步调用::

        self._ulm_batch.step(vk_in, vm_in)   # 返回 (ik_out, im_out) per line

    状态数组（xk/xm/bk/bm/fk_hist/fm_hist/vk_prev/vm_prev/step_arr/I_hist）
    全部以"批量视图"形式存活在此对象中，由 JIT kernel 就地更新，无需每步从
    各 ULMLine 对象 copy。
    """

    def __init__(self, ulm_lines: List['ULMLine'], parallel: bool = True):
        self.lines = ulm_lines
        self.parallel = bool(parallel)
        self.kernel_name = 'parallel' if self.parallel else 'serial'
        self.n_lines = len(ulm_lines)
        if self.n_lines == 0:
            return

        # --- 维度聚合 ---
        nc_list   = [l.nc for l in ulm_lines]
        n_yc_list = [l._ulm.n_yc for l in ulm_lines]
        nmod_list = [l._ulm.nmod for l in ulm_lines]
        max_nh_list  = [l._ulm.max_n_h for l in ulm_lines]
        hist_list = [l._ulm.history_length for l in ulm_lines]

        max_nc   = max(nc_list)
        max_n_yc = max(n_yc_list)
        max_nmod = max(nmod_list)
        g_max_nh = max(max_nh_list)
        max_hist = max(hist_list)
        NL = self.n_lines

        self.max_nc   = max_nc
        self.max_n_yc = max_n_yc
        self.max_nmod = max_nmod
        self.g_max_nh = g_max_nh
        self.max_hist = max_hist

        # --- meta 数组 ---
        self.nc_arr    = np.array(nc_list,   dtype=np.int64)
        self.n_yc_arr  = np.array(n_yc_list, dtype=np.int64)
        self.nmod_arr  = np.array(nmod_list, dtype=np.int64)
        self.max_n_h_arr = np.array(max_nh_list, dtype=np.int64)
        self.hist_arr  = np.array(hist_list, dtype=np.int64)
        self.dt_arr    = np.array([l._ulm.dt for l in ulm_lines], dtype=np.float64)
        self.step_arr  = np.zeros(NL, dtype=np.int64)

        # n_h_poles_batch: (NL, max_nmod)
        self.n_h_poles_batch = np.zeros((NL, max_nmod), dtype=np.int64)
        for L, line in enumerate(ulm_lines):
            ulm = line._ulm
            self.n_h_poles_batch[L, :ulm.nmod] = ulm.n_h_poles_arr

        # --- 参数数组（只读，从各 ULM 模型 copy）---
        self.pn_yc_batch = np.zeros((NL, max_n_yc), dtype=np.complex128)
        self.qn_yc_batch = np.zeros((NL, max_n_yc, max_nc, max_nc), dtype=np.complex128)
        self.yc_d_batch  = np.zeros((NL, max_nc, max_nc), dtype=np.complex128)
        self.pi_h_batch  = np.zeros((NL, max_nmod, g_max_nh), dtype=np.complex128)
        self.qij_h_batch = np.zeros((NL, max_nmod, g_max_nh, max_nc, max_nc), dtype=np.complex128)
        self.tau_batch   = np.zeros((NL, max_nmod), dtype=np.float64)
        self.G_batch     = np.zeros((NL, max_nc, max_nc), dtype=np.float64)

        for L, line in enumerate(ulm_lines):
            ulm = line._ulm
            nc  = ulm.nc
            n_yc = ulm.n_yc
            nmod = ulm.nmod
            n_h  = ulm.max_n_h

            self.pn_yc_batch[L, :n_yc]                          = ulm.pn_yc
            self.qn_yc_batch[L, :n_yc, :nc, :nc]               = ulm.qn_yc_3d
            self.yc_d_batch [L, :nc, :nc]                       = ulm._yc_d_c128
            self.pi_h_batch [L, :nmod, :n_h]                    = ulm.pi_h_2d
            self.qij_h_batch[L, :nmod, :n_h, :nc, :nc]         = ulm.qij_h_4d
            self.tau_batch  [L, :nmod]                           = ulm.tau
            self.G_batch    [L, :nc, :nc]                        = ulm.G

        # --- 状态数组（可变，就地更新）---
        # 注意: 这些是从各 ULMModel._initialize_state 重新分配的批量版本
        self.xk_batch   = np.zeros((NL, max_n_yc, max_nc), dtype=np.complex128)
        self.xm_batch   = np.zeros((NL, max_n_yc, max_nc), dtype=np.complex128)
        self.bk_batch   = np.zeros((NL, max_nmod, g_max_nh, max_nc), dtype=np.complex128)
        self.bm_batch   = np.zeros((NL, max_nmod, g_max_nh, max_nc), dtype=np.complex128)
        self.fk_hist_batch = np.zeros((NL, max_nc, max_hist), dtype=np.complex128)
        self.fm_hist_batch = np.zeros((NL, max_nc, max_hist), dtype=np.complex128)
        self.vk_prev_batch = np.zeros((NL, max_nc), dtype=np.complex128)
        self.vm_prev_batch = np.zeros((NL, max_nc), dtype=np.complex128)

        # I_hist（当前步，就地更新为下步值）
        self.I_hist_k_batch = np.zeros((NL, max_nc), dtype=np.float64)
        self.I_hist_m_batch = np.zeros((NL, max_nc), dtype=np.float64)

        # 输出缓冲区（复用，避免每步分配）
        self.ik_out_batch = np.zeros((NL, max_nc), dtype=np.float64)
        self.im_out_batch = np.zeros((NL, max_nc), dtype=np.float64)

        self.import_state_from_lines()
        self.bind_line_fast_views()
        """重置所有状态（对应 ULMLine.initialize()）。"""
        self.xk_batch  [:] = 0.0
        self.xm_batch  [:] = 0.0
        self.bk_batch  [:] = 0.0
        self.bm_batch  [:] = 0.0
        self.fk_hist_batch[:] = 0.0
        self.fm_hist_batch[:] = 0.0
        self.vk_prev_batch[:] = 0.0
        self.vm_prev_batch[:] = 0.0
        self.I_hist_k_batch[:] = 0.0
        self.I_hist_m_batch[:] = 0.0
        self.step_arr  [:] = 0

    def import_state_from_lines(self) -> None:
        """从各 ULMLine/ULMModel 导入当前状态。

        注意：这里不调用 reset()，避免在 line 已经绑定 batch 视图后把源状态清零。
        """
        if self.n_lines == 0:
            return

        for L, line in enumerate(self.lines):
            ulm = line._ulm
            nc = ulm.nc
            n_yc = ulm.n_yc
            nmod = ulm.nmod
            n_h = ulm.max_n_h
            hist_len = ulm.history_length

            self.xk_batch[L, :n_yc, :nc] = ulm.xk_2d
            self.xm_batch[L, :n_yc, :nc] = ulm.xm_2d
            self.bk_batch[L, :nmod, :n_h, :nc] = ulm.bk_3d
            self.bm_batch[L, :nmod, :n_h, :nc] = ulm.bm_3d
            self.fk_hist_batch[L, :nc, :hist_len] = ulm.fk_history
            self.fm_hist_batch[L, :nc, :hist_len] = ulm.fm_history

            self.vk_prev_batch[L, :nc] = ulm.vk_prev
            self.vm_prev_batch[L, :nc] = ulm.vm_prev

            self.I_hist_k_batch[L, :nc] = line._I_hist_k_vec[:nc]
            self.I_hist_m_batch[L, :nc] = line._I_hist_m_vec[:nc]
            self.ik_out_batch[L, :nc] = line._I_k_vec_prev[:nc]
            self.im_out_batch[L, :nc] = line._I_m_vec_prev[:nc]

            self.I_hist_k_batch[L, nc:] = 0.0
            self.I_hist_m_batch[L, nc:] = 0.0
            self.ik_out_batch[L, nc:] = 0.0
            self.im_out_batch[L, nc:] = 0.0

            self.step_arr[L] = line._step

    def bind_line_fast_views(self) -> None:
        """把 ULMLine 的历史源/电流数组绑定到 batch 行视图。"""
        if self.n_lines == 0:
            return

        for L, line in enumerate(self.lines):
            nc = int(self.nc_arr[L])

            line._I_hist_k_vec = self.I_hist_k_batch[L, :nc]
            line._I_hist_m_vec = self.I_hist_m_batch[L, :nc]
            line._I_k_vec_prev = self.ik_out_batch[L, :nc]
            line._I_m_vec_prev = self.im_out_batch[L, :nc]

            if not line._is_multiphase:
                idx = line._sp_idx
                line._I_hist_k_scalar = float(line._I_hist_k_vec[idx])
                line._I_hist_m_scalar = float(line._I_hist_m_vec[idx])

    def sync_public_state_to_lines(self, record_history: bool = False) -> None:
        """同步 batch 结果到 ULMLine 的对外可见状态。"""
        if self.n_lines == 0:
            return

        for L, line in enumerate(self.lines):
            nc = int(self.nc_arr[L])

            line._V_k_vec_prev[:nc] = self.vk_prev_batch[L, :nc].real
            line._V_m_vec_prev[:nc] = self.vm_prev_batch[L, :nc].real
            line._I_k_vec_prev[:nc] = self.ik_out_batch[L, :nc]
            line._I_m_vec_prev[:nc] = self.im_out_batch[L, :nc]
            line._I_hist_k_vec[:nc] = self.I_hist_k_batch[L, :nc]
            line._I_hist_m_vec[:nc] = self.I_hist_m_batch[L, :nc]

            line._step = int(self.step_arr[L])
            line._ulm.current_step = line._step
            line._ulm.vk_prev[:nc] = self.vk_prev_batch[L, :nc]
            line._ulm.vm_prev[:nc] = self.vm_prev_batch[L, :nc]

            if not line._is_multiphase:
                idx = line._sp_idx
                line._I_hist_k_scalar = float(self.I_hist_k_batch[L, idx])
                line._I_hist_m_scalar = float(self.I_hist_m_batch[L, idx])

            if record_history:
                line._ensure_hist_capacity()
                h = line._hist_count

                if line._is_multiphase:
                    line._V_k_hist_buf[h] = self.vk_prev_batch[L, :nc].real
                    line._V_m_hist_buf[h] = self.vm_prev_batch[L, :nc].real
                    line._I_k_hist_buf[h] = self.ik_out_batch[L, :nc]
                    line._I_m_hist_buf[h] = self.im_out_batch[L, :nc]
                else:
                    idx = line._sp_idx
                    line._V_k_hist_buf[h] = self.vk_prev_batch[L, idx].real
                    line._V_m_hist_buf[h] = self.vm_prev_batch[L, idx].real
                    line._I_k_hist_buf[h] = self.ik_out_batch[L, idx]
                    line._I_m_hist_buf[h] = self.im_out_batch[L, idx]

                line._hist_count += 1

    def export_model_state_to_lines(self) -> None:
        """完整导出 batch 内部状态回各 ULMModel。"""
        if self.n_lines == 0:
            return

        for L, line in enumerate(self.lines):
            ulm = line._ulm
            nc = ulm.nc
            n_yc = ulm.n_yc
            nmod = ulm.nmod
            n_h = ulm.max_n_h
            hist_len = ulm.history_length

            ulm.xk_2d[:, :] = self.xk_batch[L, :n_yc, :nc]
            ulm.xm_2d[:, :] = self.xm_batch[L, :n_yc, :nc]
            ulm.bk_3d[:, :, :] = self.bk_batch[L, :nmod, :n_h, :nc]
            ulm.bm_3d[:, :, :] = self.bm_batch[L, :nmod, :n_h, :nc]
            ulm.fk_history[:, :] = self.fk_hist_batch[L, :nc, :hist_len]
            ulm.fm_history[:, :] = self.fm_hist_batch[L, :nc, :hist_len]
            ulm.vk_prev[:] = self.vk_prev_batch[L, :nc]
            ulm.vm_prev[:] = self.vm_prev_batch[L, :nc]

        self.sync_public_state_to_lines(record_history=False)

    def step(
        self,
        vk_in_batch: np.ndarray,
        vm_in_batch: np.ndarray,
        sync_lines: bool = False,
        record_history: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """执行一步全部 ULM 线路计算。

        ``self.parallel=True``  使用 Numba parallel/prange batch kernel。
        ``self.parallel=False`` 使用无 parallel 调度的 serial batch kernel。
        两者共享同一函数体，便于做 A/B/C 性能比较和小规模自适应。
        """
        kernel = (
            _jit_batch_full_step_parallel
            if self.parallel else
            _jit_batch_full_step_serial
        )

        kernel(
            self.n_lines,
            self.nc_arr, self.max_nc,
            self.n_yc_arr, self.max_n_yc,
            self.nmod_arr, self.max_nmod,
            self.max_n_h_arr, self.g_max_nh,
            self.hist_arr, self.max_hist,
            self.n_h_poles_batch,
            self.dt_arr,
            self.pn_yc_batch, self.qn_yc_batch, self.yc_d_batch,
            self.pi_h_batch, self.qij_h_batch, self.tau_batch,
            self.xk_batch, self.xm_batch,
            self.bk_batch, self.bm_batch,
            self.fk_hist_batch, self.fm_hist_batch,
            self.vk_prev_batch, self.vm_prev_batch,
            self.G_batch,
            vk_in_batch, vm_in_batch,
            self.I_hist_k_batch, self.I_hist_m_batch,
            self.step_arr,
            self.ik_out_batch, self.im_out_batch,
        )

        self.step_arr += 1

        if sync_lines or record_history:
            self.sync_public_state_to_lines(record_history=record_history)

        return self.ik_out_batch, self.im_out_batch


def run_parallel_diagnostics(n_threads: int = 8, diag_level: int = 4) -> None:
    """验证 _jit_batch_full_step 的 Numba 并行化诊断。

    调用方式::

        from ulm_transmission_line_PARA import run_parallel_diagnostics
        run_parallel_diagnostics(n_threads=8, diag_level=4)

    Parameters
    ----------
    n_threads : int
        设置 Numba 使用的线程数（默认 8）。
    diag_level : int
        parallel_diagnostics 的详细级别（1~4，4 最详细；默认 4）。

    说明
    ----
    诊断输出中，若出现以下关键字则表示并行化成功：

    - ``Parallel loop listing for  ...``  — 列出所有被并行化的循环
    - ``loop #0`` 或 ``loop #N``          — 对应 prange 循环编号
    - ``#parallel_region``                — 确认存在并行区域

    若输出中只有 ``No parallel for-loops`` 则说明 prange
    未被真正展开（常见原因：数组步长不连续、类型推断失败）。

    注意
    ----
    部分 Numba 版本（已知 0.57–0.60）存在 Bug：
    ``parallel_diagnostics`` 在遍历签名时会遇到 ``metadata`` 为 ``None``
    的 overload，导致 ``AttributeError: 'NoneType' object has no attribute 'get'``。
    本函数已针对该 Bug 做了防护：先对每个签名单独尝试，失败时回退到
    手动检查 parfor IR 的替代路径，最终输出等效的判断结论。
    """
    # 1. 线程数设置与确认
    set_num_threads(n_threads)
    actual = get_num_threads()
    print("=" * 60)
    print(f"  Numba 并行诊断")
    print("=" * 60)
    print(f"  请求线程数 : {n_threads}")
    print(f"  实际线程数 : {actual}  (受 CPU 核心数上限)")
    print("-" * 60)

    # 2. 确保函数已经过 JIT 编译
    if not hasattr(_jit_batch_full_step, 'overloads') or \
            len(_jit_batch_full_step.overloads) == 0:
        print("  [INFO] _jit_batch_full_step 尚未编译，跳过诊断。")
        print("  请先运行一次含 >=2 条 ULMLine 的仿真以触发 JIT 编译，")
        print("  然后再调用 run_parallel_diagnostics()。")
        print("=" * 60)
        return

    # 3. 尝试调用 parallel_diagnostics；捕获已知的 metadata=None Bug
    print(f"  parallel_diagnostics(level={diag_level}) 输出：")
    print("-" * 60)

    diag_ok = False
    try:
        _jit_batch_full_step.parallel_diagnostics(level=diag_level)
        diag_ok = True
    except AttributeError as exc:
        if "'NoneType' object has no attribute 'get'" in str(exc):
            print(f"  [WARN] parallel_diagnostics 触发已知 Numba Bug "
                  f"(metadata=None)，改用替代检测路径。")
            print(f"         原始错误: {exc}")
        else:
            # 非预期错误，原样抛出
            raise

    # 4. 替代路径：手动检查 parfor 节点，判断 prange 是否真正展开
    if not diag_ok:
        print("-" * 60)
        print("  [替代检测] 逐签名扫描 parfor IR …")
        parallel_found   = False
        parallel_sig     = None
        n_sigs_ok        = 0
        n_sigs_null_meta = 0

        for sig, ol in _jit_batch_full_step.overloads.items():
            if ol is None:
                n_sigs_null_meta += 1
                continue
            meta = getattr(ol, 'metadata', None)
            if meta is None:
                n_sigs_null_meta += 1
                continue
            n_sigs_ok += 1

            # 方法一：检查 parfor_diagnostics
            pf_diag = meta.get('parfor_diagnostics', None)
            if pf_diag is not None:
                # parfor_diagnostics 存在且非空 → 有并行循环
                try:
                    has_parfors = bool(getattr(pf_diag, 'parfors', None))
                except Exception:
                    has_parfors = True  # 存在即认为有
                if has_parfors:
                    parallel_found = True
                    parallel_sig   = sig
                    break

            # 方法二：检查编译后的 IR 文本中是否含 parfor 关键字
            try:
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    ol.inspect_parfor()
                ir_text = buf.getvalue()
                if 'parfor' in ir_text.lower() or 'parallel' in ir_text.lower():
                    parallel_found = True
                    parallel_sig   = sig
                    break
            except Exception:
                pass

        print(f"  扫描结果: "
              f"有效签名 {n_sigs_ok} 个, "
              f"metadata=None 签名 {n_sigs_null_meta} 个")

        if parallel_found:
            print(f"  ✓ 检测到 parfor 节点（签名: {parallel_sig}）")
            print("    → prange 已被 Numba 展开为真正的并行循环。")
        else:
            # 最后一道防线：用 inspect_llvm 检查 LLVM IR 中的线程相关符号
            llvm_parallel = False
            try:
                for sig, ol in _jit_batch_full_step.overloads.items():
                    if ol is None:
                        continue
                    llvm_text = ol.inspect_llvm()
                    if any(kw in llvm_text for kw in
                           ('omp_', 'tbb_', 'numba_parallel', 'get_thread_id',
                            'launch_threads', 'parallel_reduce')):
                        llvm_parallel = True
                        break
            except Exception:
                pass

            if llvm_parallel:
                print("  ✓ LLVM IR 中含并行线程符号 → prange 已并行（LLVM 路径确认）。")
            else:
                print("  ✗ 未检测到 parfor 节点，也未发现并行线程符号。")
                print("    → prange 很可能被静默退化为串行（只剩线程池开销）。")
                print("    常见原因: 数组步长不连续 / 类型推断失败 / Numba 版本限制。")

    print("=" * 60)
    print("  诊断完成。解读要点：")
    print("  ✓  出现 'Parallel loop listing' + 'loop #N' → prange 已并行")
    print("  ✗  出现 'No parallel for-loops'            → prange 未并行展开")
    print("=" * 60)


__all__ = [
    # 数据结构与 I/O
    'FitULMData', 'FitULMReader', 'FitULMWriter',
    'create_test_fitulm_data', 'convert_to_ulm_format',
    'validate_fitting', 'check_passivity',
    'print_poles_info', 'print_residues_info',
    # 核心模型
    'ULMModel', 'TransientSimulator',
    'load_ulm_model', 'simulate_step_response', 'plot_results',
    # 传输线接口
    'ULMLine',
    # 批量并行
    'ULMBatchPack',
    # JIT 工具
    'warmup_jit',
    # 并行诊断
    'run_parallel_diagnostics',
]