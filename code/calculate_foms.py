#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOMS 计算脚本 - 基于 MACRS 方法
严格遵循技术文档规范，使用有效摩擦面积（圆盘面积的一半）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class FOMSCalculator:
    """盘式 TENG 结构优值计算器"""

    # 物理常数（双精度浮点数）
    EPSILON_0 = 8.854e-12  # 真空介电常数 (F/m)
    SIGMA = 1e-5  # 表面电荷密度 (C/m^2)
    R = 0.015  # 圆盘半径 (m)
    PI = np.pi  # 圆周率

    # 有效摩擦面积：圆盘面积的一半
    EFFECTIVE_AREA = 0.5 * PI * R**2

    # 总摩擦电荷量
    Q_TOTAL = SIGMA * EFFECTIVE_AREA

    def __init__(self, start_file: str, end_file: str):
        """
        初始化计算器

        参数:
            start_file: Start 状态 CSV 文件路径
            end_file: End 状态 CSV 文件路径
        """
        self.start_file = Path(start_file)
        self.end_file = Path(end_file)
        self.data = None
        self.results = None

    def load_data(self):
        """加载并合并 start 和 end 数据集"""
        print(f"正在加载数据...")

        # 读取数据
        df_start = pd.read_csv(self.start_file)
        df_end = pd.read_csv(self.end_file)

        # 添加状态标签
        df_start["state"] = "start"
        df_end["state"] = "end"

        # 合并数据
        self.data = pd.concat([df_start, df_end], ignore_index=True)

        print(f"  - Start 状态: {len(df_start)} 行")
        print(f"  - End 状态: {len(df_end)} 行")
        print(f"  - 合并后总计: {len(self.data)} 行")

        # 验证必需列
        required_cols = ["n", "E", "dd", "hh", "d", "h", "A", "V"]
        missing_cols = set(required_cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

    def validate_group(self, group_data):
        """
        验证组数据的有效性

        参数:
            group_data: 单个结构组的数据

        返回:
            bool: 数据是否有效
        """
        # 检查是否有足够的数据点
        start_count = (group_data["state"] == "start").sum()
        end_count = (group_data["state"] == "end").sum()

        if start_count < 2 or end_count < 2:
            return False

        # 检查是否有有效的电压数据
        if group_data["V"].isna().any() or np.isinf(group_data["V"]).any():
            return False

        return True

    def linear_regression(self, A, V):
        """
        执行线性回归: V = k * A + b

        参数:
            A: 电荷转移比例（自变量）
            V: 端电压（因变量）

        返回:
            tuple: (k, b) 斜率和截距
        """
        try:
            # 使用 numpy polyfit 进行线性拟合
            coeffs = np.polyfit(A, V, deg=1)
            k, b = coeffs[0], coeffs[1]
            return k, b
        except Exception:
            return np.nan, np.nan

    def calculate_group_foms(self, group_name, group_data):
        """
        计算单个结构组的 FOMS

        参数:
            group_name: 组名 (n, E, dd, hh)
            group_data: 该组的数据

        返回:
            dict: 计算结果
        """
        n, E, dd, hh = group_name

        # 验证数据有效性
        if not self.validate_group(group_data):
            return self._create_nan_result(n, E, dd, hh)

        # 分离 start 和 end 状态
        start_data = group_data[group_data["state"] == "start"].sort_values("A")
        end_data = group_data[group_data["state"] == "end"].sort_values("A")

        # 线性回归
        k_start, b_start = self.linear_regression(
            start_data["A"].values, start_data["V"].values
        )
        k_end, b_end = self.linear_regression(
            end_data["A"].values, end_data["V"].values
        )

        # 检查回归结果
        if np.isnan(k_start) or np.isnan(k_end):
            return self._create_nan_result(n, E, dd, hh)

        # 计算倒电容 (1/C)
        invC_start = k_start / self.Q_TOTAL
        invC_end = k_end / self.Q_TOTAL

        # 检查倒电容是否接近零
        if abs(invC_start) < 1e-20 or abs(invC_end) < 1e-20:
            return self._create_nan_result(n, E, dd, hh)

        # 计算短路电荷
        Qsc_start = -b_start / invC_start
        Qsc_end = -b_end / invC_end

        # 计算 MACRS 参数
        Qsc_MACRS = abs(Qsc_end - Qsc_start)
        Voc_MACRS = abs(b_end - b_start)

        # 计算 FOMS（使用修正公式）
        # 注意：由于Q_total使用了有效面积(0.5*圆盘面积)，需要乘以2进行补偿
        term_const = (n * self.EPSILON_0) / (self.SIGMA**2 * self.PI**2 * self.R**3)
        term_Q = Qsc_MACRS**2
        term_C = invC_start + invC_end
        FOMS = 2 * term_const * term_Q * term_C  # 乘以2补偿有效面积

        return {
            "n": n,
            "E": E,
            "dd": dd,
            "hh": hh,
            "inv_C_start": invC_start,
            "inv_C_end": invC_end,
            "Qsc_start": Qsc_start,
            "Qsc_end": Qsc_end,
            "Qsc_MACRS": Qsc_MACRS,
            "Voc_MACRS": Voc_MACRS,
            "FOMS": FOMS,
        }

    def _create_nan_result(self, n, E, dd, hh):
        """创建包含 NaN 的结果字典"""
        return {
            "n": n,
            "E": E,
            "dd": dd,
            "hh": hh,
            "inv_C_start": np.nan,
            "inv_C_end": np.nan,
            "Qsc_start": np.nan,
            "Qsc_end": np.nan,
            "Qsc_MACRS": np.nan,
            "Voc_MACRS": np.nan,
            "FOMS": np.nan,
        }

    def calculate_all(self):
        """计算所有结构组的 FOMS"""
        print(f"\n开始计算 FOMS...")

        # 按结构参数分组
        grouped = self.data.groupby(["n", "E", "dd", "hh"])
        total_groups = len(grouped)

        print(f"  - 共有 {total_groups} 个结构组")

        # 计算每个组的 FOMS
        results_list = []
        valid_count = 0

        for i, (group_name, group_data) in enumerate(grouped, 1):
            result = self.calculate_group_foms(group_name, group_data)
            results_list.append(result)

            if not np.isnan(result["FOMS"]):
                valid_count += 1

            # 进度提示
            if i % 100 == 0 or i == total_groups:
                print(f"  - 进度: {i}/{total_groups} ({i*100//total_groups}%)")

        # 转换为 DataFrame
        self.results = pd.DataFrame(results_list)

        print(f"\n计算完成!")
        print(f"  - 有效结果: {valid_count}/{total_groups}")
        print(f"  - 失败/无效: {total_groups - valid_count}")

    def save_results(self, output_file: str = "calculated_foms_macrs.csv"):
        """保存计算结果"""
        if self.results is None:
            raise ValueError("尚未计算结果，请先调用 calculate_all()")

        output_path = Path(output_file)
        self.results.to_csv(output_path, index=False, float_format="%.15e")

        print(f"\n结果已保存至: {output_path}")
        print(f"  - 总行数: {len(self.results)}")

    def print_summary(self):
        """打印计算结果摘要"""
        if self.results is None:
            return

        print("\n" + "=" * 60)
        print("计算结果摘要".center(60))
        print("=" * 60)

        valid_results = self.results.dropna(subset=["FOMS"])

        if len(valid_results) > 0:
            print(f"\nFOMS 统计:")
            print(f"  - 最小值: {valid_results['FOMS'].min():.6e}")
            print(f"  - 最大值: {valid_results['FOMS'].max():.6e}")
            print(f"  - 平均值: {valid_results['FOMS'].mean():.6e}")
            print(f"  - 中位数: {valid_results['FOMS'].median():.6e}")

            print(f"\nQsc_MACRS 统计 (C):")
            print(f"  - 最小值: {valid_results['Qsc_MACRS'].min():.6e}")
            print(f"  - 最大值: {valid_results['Qsc_MACRS'].max():.6e}")

            print(f"\nVoc_MACRS 统计 (V):")
            print(f"  - 最小值: {valid_results['Voc_MACRS'].min():.6e}")
            print(f"  - 最大值: {valid_results['Voc_MACRS'].max():.6e}")
        else:
            print("\n警告: 没有有效的计算结果！")

        print("=" * 60 + "\n")


def main():
    """主函数"""
    # 设置文件路径
    start_file = "../data/CSV/merged_dataset_start.csv"
    end_file = "../data/CSV/merged_dataset_end.csv"
    output_file = "../data/calculated_foms_macrs.csv"

    print("=" * 60)
    print("盘式 TENG 结构优值 (FOMS) 计算程序".center(60))
    print("基于 MACRS 方法 | 使用有效摩擦面积".center(60))
    print("=" * 60 + "\n")

    # 创建计算器实例
    calculator = FOMSCalculator(start_file, end_file)

    # 执行计算流程
    calculator.load_data()
    calculator.calculate_all()
    calculator.print_summary()
    calculator.save_results(output_file)

    print("\n程序执行完毕！")


if __name__ == "__main__":
    main()
