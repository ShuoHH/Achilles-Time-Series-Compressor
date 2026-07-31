"""
数据病理学诊断工具 (Data Pathology Diagnostic Tool)

针对 NeurTS "插值+残差" 架构的数据适配性分析。
检查数据特征是否对该架构不友好。

诊断项目：
1. 频率-分辨率失配检测 (Frequency-Resolution Mismatch)
2. 狄拉克跳变检测 (Dirac Step Detection)
3. 局部线性度分析 (Local Linearity Check)
4. 噪声基底与信噪比 (Noise Floor & SNR)

Usage:
    python data_pathology.py --data_path ./datasets/your_data.csv --data_col 0 --min_resolution 64
"""

import numpy as np
import pandas as pd
import argparse
from typing import Tuple, List, Dict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DiagnosticResult:
    """单项诊断结果"""
    name: str
    status: str  # "PASS", "WARNING", "CRITICAL"
    score: float  # 0-100, 越高越健康
    details: Dict
    recommendation: str


class DataPathologist:
    """数据病理学家：诊断数据是否适合 NeurTS 架构"""
    
    def __init__(self, data: np.ndarray, min_resolution: int = 64, base_block_size: int = 512):
        """
        Args:
            data: 1D 时序数据
            min_resolution: NeurTS 最小分辨率
            base_block_size: 基础块大小
        """
        self.data = data
        self.min_resolution = min_resolution
        self.base_block_size = base_block_size
        self.n = len(data)
        
        # 预计算基础统计量
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.data_range = np.max(data) - np.min(data)
        
        # 诊断结果
        self.results: List[DiagnosticResult] = []
    
    def run_all_diagnostics(self) -> List[DiagnosticResult]:
        """运行所有诊断项目"""
        self.results = []
        
        print("=" * 70)
        print("NeurTS 数据病理学诊断报告")
        print("=" * 70)
        print(f"\n数据概览:")
        print(f"  - 数据长度: {self.n}")
        print(f"  - 均值: {self.mean:.4f}")
        print(f"  - 标准差: {self.std:.4f}")
        print(f"  - 数据范围: {self.data_range:.4f}")
        print(f"  - 最小分辨率 (R_min): {self.min_resolution}")
        print(f"  - 基础块大小: {self.base_block_size}")
        
        # 运行各项诊断
        self.results.append(self.diagnose_frequency_mismatch())
        self.results.append(self.diagnose_dirac_steps())
        self.results.append(self.diagnose_local_linearity())
        self.results.append(self.diagnose_noise_floor())
        
        # 生成总结报告
        self.generate_summary_report()
        
        return self.results
    
    def diagnose_frequency_mismatch(self) -> DiagnosticResult:
        """
        诊断1: 频率-分辨率失配检测
        
        检查数据中是否存在波长小于 min_resolution 的高频震荡。
        如果一个 min_resolution 区间内包含了 >1 个完整的波峰/波谷，
        说明网格密度物理上不足以支撑该信号。
        """
        print("\n" + "-" * 70)
        print("诊断1: 频率-分辨率失配检测 (Frequency-Resolution Mismatch)")
        print("-" * 70)
        
        # 方法1: 过零率分析
        # 去均值后计算过零次数
        centered = self.data - self.mean
        zero_crossings = np.sum(np.abs(np.diff(np.sign(centered))) > 0)
        zero_crossing_rate = zero_crossings / self.n
        
        # 估算平均波长
        if zero_crossings > 0:
            avg_wavelength = 2 * self.n / zero_crossings  # 两次过零 = 一个完整周期
        else:
            avg_wavelength = float('inf')
        
        # 方法2: FFT 主频分析
        fft_result = np.fft.fft(self.data - self.mean)
        freqs = np.fft.fftfreq(self.n)
        power = np.abs(fft_result) ** 2
        
        # 找到主频（排除直流分量）
        positive_freqs = freqs[1:self.n//2]
        positive_power = power[1:self.n//2]
        
        if len(positive_power) > 0:
            dominant_freq_idx = np.argmax(positive_power)
            dominant_freq = positive_freqs[dominant_freq_idx]
            dominant_wavelength = 1 / dominant_freq if dominant_freq > 0 else float('inf')
        else:
            dominant_freq = 0
            dominant_wavelength = float('inf')
        
        # 计算高频能量占比（波长 < min_resolution 的成分）
        high_freq_threshold = 1 / self.min_resolution
        high_freq_mask = np.abs(freqs) > high_freq_threshold
        high_freq_energy = np.sum(power[high_freq_mask])
        total_energy = np.sum(power)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # 方法3: 局部过零率（在每个 min_resolution 窗口内）
        num_windows = self.n // self.min_resolution
        local_zero_crossings = []
        for i in range(num_windows):
            start = i * self.min_resolution
            end = start + self.min_resolution
            window = self.data[start:end] - np.mean(self.data[start:end])
            zc = np.sum(np.abs(np.diff(np.sign(window))) > 0)
            local_zero_crossings.append(zc)
        
        avg_local_zc = np.mean(local_zero_crossings) if local_zero_crossings else 0
        max_local_zc = np.max(local_zero_crossings) if local_zero_crossings else 0
        
        # 判定
        # 如果平均每个窗口有 >2 次过零（即 >1 个完整周期），则存在问题
        if avg_local_zc > 4:
            status = "CRITICAL"
            score = max(0, 100 - avg_local_zc * 10)
        elif avg_local_zc > 2:
            status = "WARNING"
            score = max(0, 100 - avg_local_zc * 5)
        else:
            status = "PASS"
            score = 100 - avg_local_zc * 2
        
        details = {
            "全局过零率": f"{zero_crossing_rate:.4f}",
            "估算平均波长": f"{avg_wavelength:.1f}",
            "FFT主频波长": f"{dominant_wavelength:.1f}",
            "高频能量占比": f"{high_freq_ratio*100:.2f}%",
            "窗口内平均过零数": f"{avg_local_zc:.2f}",
            "窗口内最大过零数": f"{max_local_zc}",
            "min_resolution": self.min_resolution
        }
        
        if status == "CRITICAL":
            recommendation = f"数据包含大量高频成分（窗口内平均{avg_local_zc:.1f}次过零），min_resolution={self.min_resolution}无法捕捉。建议：(1)降低min_resolution (2)对数据进行低通滤波预处理"
        elif status == "WARNING":
            recommendation = f"存在中等程度的高频成分。建议关注高频能量占比{high_freq_ratio*100:.1f}%的来源"
        else:
            recommendation = "频率特性与分辨率匹配良好"
        
        print(f"\n  [结果] {status}")
        for k, v in details.items():
            print(f"    - {k}: {v}")
        print(f"  [建议] {recommendation}")
        
        return DiagnosticResult(
            name="频率-分辨率失配检测",
            status=status,
            score=score,
            details=details,
            recommendation=recommendation
        )
    
    def diagnose_dirac_steps(self) -> DiagnosticResult:
        """
        诊断2: 狄拉克跳变检测 (Dirac Step Detection)
        
        检查是否存在瞬间的、垂直的数值跳变。
        这会破坏插值引导信号，且 Lipschitz 受限的 TCN 无法拟合。
        """
        print("\n" + "-" * 70)
        print("诊断2: 狄拉克跳变检测 (Dirac Step Detection)")
        print("-" * 70)
        
        # 计算一阶差分
        diff = np.abs(np.diff(self.data))
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)
        
        # 检测超过 3σ 的跳变
        threshold_3sigma = diff_mean + 3 * diff_std
        threshold_5sigma = diff_mean + 5 * diff_std
        
        jumps_3sigma = np.where(diff > threshold_3sigma)[0]
        jumps_5sigma = np.where(diff > threshold_5sigma)[0]
        
        num_jumps_3sigma = len(jumps_3sigma)
        num_jumps_5sigma = len(jumps_5sigma)
        
        # 区分"阶跃跳变"和"尖峰噪声"
        # 阶跃跳变：跳变后保持在新位置
        # 尖峰噪声：跳变后立即回落
        step_jumps = []
        spike_jumps = []
        
        for idx in jumps_3sigma:
            if idx < self.n - 2:
                # 检查跳变后是否保持
                post_diff = np.abs(self.data[idx + 2] - self.data[idx + 1])
                jump_size = diff[idx]
                if post_diff < jump_size * 0.3:  # 跳变后保持稳定
                    step_jumps.append(idx)
                else:
                    spike_jumps.append(idx)
            else:
                step_jumps.append(idx)
        
        # 计算跳变幅度统计
        if len(jumps_3sigma) > 0:
            max_jump = np.max(diff[jumps_3sigma])
            avg_jump = np.mean(diff[jumps_3sigma])
            max_jump_relative = max_jump / self.std if self.std > 0 else 0
        else:
            max_jump = 0
            avg_jump = 0
            max_jump_relative = 0
        
        # 判定
        jump_rate = num_jumps_3sigma / self.n
        step_rate = len(step_jumps) / self.n
        
        if len(step_jumps) > 10 or step_rate > 0.001:
            status = "CRITICAL"
            score = max(0, 100 - len(step_jumps) * 5)
        elif num_jumps_3sigma > 50 or jump_rate > 0.01:
            status = "WARNING"
            score = max(0, 100 - num_jumps_3sigma)
        else:
            status = "PASS"
            score = 100 - num_jumps_3sigma * 0.5
        
        details = {
            "一阶差分均值": f"{diff_mean:.4f}",
            "一阶差分标准差": f"{diff_std:.4f}",
            "3σ阈值": f"{threshold_3sigma:.4f}",
            "超3σ跳变数": num_jumps_3sigma,
            "超5σ跳变数": num_jumps_5sigma,
            "阶跃跳变数": len(step_jumps),
            "尖峰噪声数": len(spike_jumps),
            "最大跳变幅度": f"{max_jump:.4f}",
            "最大跳变/标准差": f"{max_jump_relative:.2f}σ"
        }
        
        if status == "CRITICAL":
            recommendation = f"检测到{len(step_jumps)}个阶跃跳变，这会严重破坏Hermite插值。建议：(1)对数据进行平滑预处理 (2)在跳变点强制插入Grid节点 (3)考虑分段处理"
        elif status == "WARNING":
            recommendation = f"存在{num_jumps_3sigma}个异常跳变点，可能影响拟合质量"
        else:
            recommendation = "数据连续性良好，适合插值"
        
        print(f"\n  [结果] {status}")
        for k, v in details.items():
            print(f"    - {k}: {v}")
        print(f"  [建议] {recommendation}")
        
        # 打印前几个跳变位置
        if len(step_jumps) > 0:
            print(f"  [阶跃跳变位置(前10个)]: {step_jumps[:10]}")
        
        return DiagnosticResult(
            name="狄拉克跳变检测",
            status=status,
            score=score,
            details=details,
            recommendation=recommendation
        )
    
    def diagnose_local_linearity(self) -> DiagnosticResult:
        """
        诊断3: 局部线性度分析 (Local Linearity Check)
        
        检查"引导信号"是否有效。
        模拟每隔 L 个点做一次线性插值，计算残差能量。
        """
        print("\n" + "-" * 70)
        print("诊断3: 局部线性度分析 (Local Linearity Check)")
        print("-" * 70)
        
        L = self.base_block_size  # 使用基础块大小作为插值间隔
        
        # 计算线性插值残差
        num_segments = self.n // L
        residuals = []
        residual_ratios = []
        
        for i in range(num_segments):
            start = i * L
            end = min(start + L, self.n)
            segment = self.data[start:end]
            
            # 线性插值
            x = np.arange(len(segment))
            linear_interp = np.linspace(segment[0], segment[-1], len(segment))
            
            # 残差
            residual = segment - linear_interp
            residual_energy = np.sqrt(np.mean(residual ** 2))  # RMS
            segment_energy = np.sqrt(np.mean(segment ** 2))
            
            residuals.append(residual_energy)
            if segment_energy > 0:
                residual_ratios.append(residual_energy / segment_energy)
            else:
                residual_ratios.append(0)
        
        avg_residual = np.mean(residuals)
        max_residual = np.max(residuals)
        avg_ratio = np.mean(residual_ratios)
        max_ratio = np.max(residual_ratios)
        
        # 计算残差与原始数据的比值
        data_rms = np.sqrt(np.mean(self.data ** 2))
        residual_to_data_ratio = avg_residual / data_rms if data_rms > 0 else 0
        
        # 统计残差超过原始幅度的段数
        bad_segments = sum(1 for r in residual_ratios if r > 0.5)
        very_bad_segments = sum(1 for r in residual_ratios if r > 1.0)
        
        # 判定
        if very_bad_segments > num_segments * 0.1 or avg_ratio > 0.8:
            status = "CRITICAL"
            score = max(0, 100 - very_bad_segments * 10)
        elif bad_segments > num_segments * 0.2 or avg_ratio > 0.5:
            status = "WARNING"
            score = max(0, 100 - bad_segments * 5)
        else:
            status = "PASS"
            score = 100 - avg_ratio * 50
        
        details = {
            "插值间隔L": L,
            "段数": num_segments,
            "平均残差RMS": f"{avg_residual:.4f}",
            "最大残差RMS": f"{max_residual:.4f}",
            "平均残差/段能量": f"{avg_ratio*100:.2f}%",
            "最大残差/段能量": f"{max_ratio*100:.2f}%",
            "残差/数据RMS": f"{residual_to_data_ratio*100:.2f}%",
            "残差>50%的段数": f"{bad_segments}/{num_segments}",
            "残差>100%的段数": f"{very_bad_segments}/{num_segments}"
        }
        
        if status == "CRITICAL":
            recommendation = f"线性插值残差过大（{very_bad_segments}段超过原始幅度），说明数据高度非线性。建议：(1)增加分裂次数 (2)使用更小的base_block_size (3)考虑非线性引导信号"
        elif status == "WARNING":
            recommendation = f"部分区域线性度较差（{bad_segments}段残差>50%），可能需要更多分裂"
        else:
            recommendation = "数据局部线性度良好，Hermite引导信号应该有效"
        
        print(f"\n  [结果] {status}")
        for k, v in details.items():
            print(f"    - {k}: {v}")
        print(f"  [建议] {recommendation}")
        
        return DiagnosticResult(
            name="局部线性度分析",
            status=status,
            score=score,
            details=details,
            recommendation=recommendation
        )
    
    def diagnose_noise_floor(self) -> DiagnosticResult:
        """
        诊断4: 噪声基底与信噪比 (Noise Floor & SNR)
        
        检查自适应分裂是否在"追逐噪声"。
        通过移动方差检测平坦区的噪声水平。
        """
        print("\n" + "-" * 70)
        print("诊断4: 噪声基底与信噪比 (Noise Floor & SNR)")
        print("-" * 70)
        
        # 方法1: 小窗口移动方差
        window_size = 5
        local_vars = []
        for i in range(self.n - window_size + 1):
            window = self.data[i:i + window_size]
            local_vars.append(np.var(window))
        
        local_vars = np.array(local_vars)
        
        # 找到"平坦区"（局部方差最小的区域）
        sorted_vars = np.sort(local_vars)
        noise_floor_var = np.median(sorted_vars[:len(sorted_vars)//10])  # 取最小10%的中位数
        noise_floor_std = np.sqrt(noise_floor_var)
        
        # 方法2: 高通滤波估计噪声
        # 使用一阶差分的差分（二阶差分）来估计噪声
        diff2 = np.diff(self.data, n=2)
        noise_estimate = np.std(diff2) / np.sqrt(6)  # 理论校正因子
        
        # 计算信噪比
        signal_power = self.std ** 2
        noise_power = noise_floor_var
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        # 计算噪声占比
        noise_ratio = noise_floor_std / self.std if self.std > 0 else 0
        
        # 方法3: 检测"伪平坦区"的方差
        # 找到一阶差分较小的区域
        diff1 = np.abs(np.diff(self.data))
        flat_threshold = np.percentile(diff1, 20)  # 最平坦的20%
        flat_regions = diff1 < flat_threshold
        
        if np.sum(flat_regions) > 0:
            flat_region_indices = np.where(flat_regions)[0]
            flat_region_vars = []
            for idx in flat_region_indices:
                if idx >= 2 and idx < self.n - 3:
                    local_window = self.data[idx-2:idx+3]
                    flat_region_vars.append(np.var(local_window))
            if flat_region_vars:
                flat_region_noise = np.sqrt(np.median(flat_region_vars))
            else:
                flat_region_noise = 0
        else:
            flat_region_noise = 0
        
        # 判定
        if snr < 10 or noise_ratio > 0.3:
            status = "CRITICAL"
            score = max(0, snr * 5)
        elif snr < 20 or noise_ratio > 0.15:
            status = "WARNING"
            score = max(0, snr * 3)
        else:
            status = "PASS"
            score = min(100, snr * 2)
        
        details = {
            "数据标准差": f"{self.std:.4f}",
            "噪声基底(移动方差法)": f"{noise_floor_std:.4f}",
            "噪声估计(二阶差分法)": f"{noise_estimate:.4f}",
            "平坦区噪声": f"{flat_region_noise:.4f}",
            "信噪比(SNR)": f"{snr:.2f} dB",
            "噪声/信号比": f"{noise_ratio*100:.2f}%"
        }
        
        if status == "CRITICAL":
            recommendation = f"信噪比过低({snr:.1f}dB)，自适应分裂可能在追逐噪声。建议：(1)对数据进行去噪预处理 (2)设置更高的分裂误差阈值 (3)限制最大分裂次数"
        elif status == "WARNING":
            recommendation = f"存在一定噪声({noise_ratio*100:.1f}%)，可能影响分裂效率"
        else:
            recommendation = "信噪比良好，自适应分裂应该有效"
        
        print(f"\n  [结果] {status}")
        for k, v in details.items():
            print(f"    - {k}: {v}")
        print(f"  [建议] {recommendation}")
        
        return DiagnosticResult(
            name="噪声基底与信噪比",
            status=status,
            score=score,
            details=details,
            recommendation=recommendation
        )
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "=" * 70)
        print("诊断总结")
        print("=" * 70)
        
        # 统计各状态数量
        status_counts = {"PASS": 0, "WARNING": 0, "CRITICAL": 0}
        total_score = 0
        
        print("\n诊断项目汇总:")
        print("-" * 70)
        for result in self.results:
            status_counts[result.status] += 1
            total_score += result.score
            status_icon = {"PASS": "[OK]", "WARNING": "[!!]", "CRITICAL": "[XX]"}[result.status]
            print(f"  {status_icon} {result.name}: {result.status} (得分: {result.score:.0f}/100)")
        
        avg_score = total_score / len(self.results) if self.results else 0
        
        print("-" * 70)
        print(f"\n综合健康评分: {avg_score:.0f}/100")
        
        # 总体判定
        if status_counts["CRITICAL"] > 0:
            overall = "不适合"
            print(f"\n[总体判定] 数据 **{overall}** 直接使用 NeurTS 架构")
            print(f"  - 存在 {status_counts['CRITICAL']} 个严重问题需要解决")
        elif status_counts["WARNING"] > 1:
            overall = "需要调整"
            print(f"\n[总体判定] 数据 **{overall}** 后可使用 NeurTS 架构")
            print(f"  - 存在 {status_counts['WARNING']} 个警告需要关注")
        else:
            overall = "适合"
            print(f"\n[总体判定] 数据 **{overall}** 使用 NeurTS 架构")
        
        # 优先处理建议
        print("\n优先处理建议:")
        priority = 1
        for result in self.results:
            if result.status == "CRITICAL":
                print(f"  {priority}. [紧急] {result.name}: {result.recommendation}")
                priority += 1
        for result in self.results:
            if result.status == "WARNING":
                print(f"  {priority}. [建议] {result.name}: {result.recommendation}")
                priority += 1
        
        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='NeurTS 数据病理学诊断工具')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径 (CSV)')
    parser.add_argument('--data_col', type=int, default=0, help='数据列索引')
    parser.add_argument('--min_resolution', type=int, default=64, help='NeurTS 最小分辨率')
    parser.add_argument('--base_block_size', type=int, default=512, help='基础块大小')
    parser.add_argument('--max_samples', type=int, default=None, help='最大采样数（用于大数据集）')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.data_path}")
    df = pd.read_csv(args.data_path, header=None, usecols=[args.data_col])
    # 尝试转换为数值，非数值行会变成 NaN
    data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    
    if args.max_samples and len(data) > args.max_samples:
        print(f"数据过长({len(data)})，截取前{args.max_samples}个样本")
        data = data[:args.max_samples]
    
    # 创建诊断器并运行
    pathologist = DataPathologist(
        data=data,
        min_resolution=args.min_resolution,
        base_block_size=args.base_block_size
    )
    
    pathologist.run_all_diagnostics()


if __name__ == "__main__":
    main()
