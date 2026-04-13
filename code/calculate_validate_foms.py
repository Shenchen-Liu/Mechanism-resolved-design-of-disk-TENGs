#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 Validate 数据集的 FOMS
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from calculate_foms import FOMSCalculator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Calculate FOMS for Validate Dataset")
    parser.add_argument(
        "--validate_set",
        type=str,
        default="validate2",
        choices=["validate", "validate2", "validate3"],
        help="验证数据集选择: validate, validate2 或 validate3",
    )
    return parser.parse_args()


def main():
    """计算 validate 数据集的 FOMS"""
    args = parse_args()

    # 设置文件路径
    start_file = f"../data/CSV/disk_TENG_{args.validate_set}_start.csv"
    end_file = f"../data/CSV/disk_TENG_{args.validate_set}_end.csv"
    output_file = f"../data/{args.validate_set}_foms_macrs.csv"

    print("=" * 60)
    print(f"Calculate FOMS - {args.validate_set.upper()}")
    print("=" * 60)

    calculator = FOMSCalculator(start_file, end_file)
    calculator.load_data()
    calculator.calculate_all()
    calculator.save_results(output_file)

    print(f"\n✅ 完成！结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
