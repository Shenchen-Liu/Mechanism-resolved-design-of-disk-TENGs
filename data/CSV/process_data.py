import pandas as pd
import glob
import os
import numpy as np
import re


def process_teng_datasets():
    # 1. 定义预期的参数集合和目标列结构
    EXPECTED_PARAMS = {
        "A": {0, 0.25, 0.5, 0.75, 1},
        "E": {1, 2, 3, 5, 7, 10},
        "dd": {0.03125, 0.0625, 0.125, 0.25, 0.5, 1},
        "hh": {0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1},
    }

    # 目标列顺序：n在第一列，d在dd后，h在hh后
    TARGET_COLUMNS = ["n", "A", "E", "dd", "d", "hh", "h", "V"]

    # 定义重命名映射
    RENAME_MAP = {"d (m)": "d", "h1 (m)": "h", "es.V0_1-es.V0_2 (V)": "V"}

    # 存储处理后的DataFrame列表
    start_dfs = []
    end_dfs = []
    validate_start_dfs = []
    validate_end_dfs = []

    # 获取当前目录下所有csv文件
    csv_files = glob.glob("*.csv")

    print(f"找到 {len(csv_files)} 个CSV文件，开始处理...")
    print("-" * 60)

    for file_path in csv_files:
        filename = os.path.basename(file_path)

        # 跳过之前的合并结果文件
        if filename.startswith("merged_dataset"):
            continue

        # 检查是否为validate文件
        is_validate = "validate" in filename.lower()

        # 解析文件名中的 n 值 (例如 disk_TENG_n64_start.csv -> 64)
        # validate文件不需要从文件名解析n值
        if is_validate:
            expected_n = None  # validate文件的n值从数据中读取
        else:
            n_match = re.search(r"n(\d+)_", filename)
            if not n_match:
                print(f"[跳过] 文件 {filename} 无法从文件名解析出 n 值。")
                continue
            expected_n = int(n_match.group(1))

        # 判断是start还是end
        if "_start" in filename:
            file_type = "start"
        elif "_end" in filename:
            file_type = "end"
        else:
            continue

        try:
            # 读取CSV
            df = pd.read_csv(file_path)

            # -------------------------------------------------
            # 步骤1: 列名初步重命名
            # -------------------------------------------------
            df.rename(columns=RENAME_MAP, inplace=True)

            # -------------------------------------------------
            # 步骤1.5: 处理validate文件的固定列
            # -------------------------------------------------
            if is_validate:
                print(
                    f"[Validate文件] 正在处理 {filename}，添加固定列 E=1, dd=0.125, hh=0.0625..."
                )
                # 添加固定值列
                df["E"] = 1
                df["dd"] = 0.125
                df["hh"] = 0.0625

            # -------------------------------------------------
            # 步骤2: 智能纠错 (修复 n 和 d 混淆的问题)
            # -------------------------------------------------
            # validate文件跳过智能纠错逻辑
            if not is_validate:
                # 检查 d 列是否实际上存的是 n 的值 (即 d 列全等于 expected_n)
                if "d" in df.columns:
                    d_mean = df["d"].mean()
                    d_std = df["d"].std()
                    # 如果 d 列的值非常接近 expected_n 且方差极小（几乎是常数），说明这列其实是 n
                    d_is_actually_n = np.isclose(d_mean, expected_n, rtol=0.01) and (
                        pd.isna(d_std) or d_std < 0.001
                    )
                else:
                    d_is_actually_n = False

                # 检查 n 列是否实际上存的是 d 的值 (即 n 列是小数，远小于 expected_n，或者有波动)
                if "n" in df.columns:
                    try:
                        n_mean = df["n"].mean()
                        # 如果 n 列均值远小于 expected_n (比如 n=64, 但均值是 0.00x)，或者 n 列就是小数
                        n_is_actually_d = (n_mean < 1.0 and expected_n >= 2) or (
                            df["n"].std() > 0.0001
                        )
                    except:
                        n_is_actually_d = False
                else:
                    n_is_actually_d = False

                # 执行互换或重命名
                if d_is_actually_n and n_is_actually_d:
                    print(
                        f"[自动修复] 文件 {filename}: 检测到 n 列和 d 列内容混淆，正在互换..."
                    )
                    # 临时重命名以交换
                    df.rename(columns={"n": "temp_d", "d": "n"}, inplace=True)
                    df.rename(columns={"temp_d": "d"}, inplace=True)
                elif d_is_actually_n and "n" not in df.columns:
                    print(
                        f"[自动修复] 文件 {filename}: d 列看起来是 n，正在重命名 d->n..."
                    )
                    df.rename(columns={"d": "n"}, inplace=True)

            # -------------------------------------------------
            # 步骤3: 强制数据修正与补全
            # -------------------------------------------------
            # 对于非validate文件，强制将 n 列赋值为文件名解析出的值
            # 对于validate文件，n值已经在数据中，不需要覆盖
            if not is_validate:
                df["n"] = expected_n
            else:
                # validate文件已有n列，确保它是整数类型
                df["n"] = df["n"].astype(int)

            # -------------------------------------------------
            # 步骤4: 检查必要列是否存在
            # -------------------------------------------------
            missing_cols = [col for col in TARGET_COLUMNS if col not in df.columns]
            if missing_cols:
                print(
                    f"[错误] 文件 {filename} 缺少必要的列: {missing_cols}，跳过该文件。"
                )
                continue

            # -------------------------------------------------
            # 步骤5: 调整列顺序
            # -------------------------------------------------
            df = df[TARGET_COLUMNS]

            # -------------------------------------------------
            # 步骤6: 数据验证
            # -------------------------------------------------

            # A. 检查行数（validate文件行数不同，跳过此检查）
            row_count = len(df)
            if not is_validate and row_count != 1620:
                print(f"[警告] 文件 {filename} 行数为 {row_count} (预期 1620)")
            elif is_validate:
                print(f"[信息] Validate文件 {filename} 包含 {row_count} 行数据")

            # B. 检查重复组合
            param_cols = ["n", "A", "E", "dd", "hh"]
            duplicates = df.duplicated(subset=param_cols).sum()
            if duplicates > 0:
                print(f"[警告] 文件 {filename} 包含 {duplicates} 个重复的参数组合！")

            # C. 检查参数取值范围（validate文件可能有不同的参数值，跳过严格检查）
            valid_check = True
            if not is_validate:
                for param, allowed_values in EXPECTED_PARAMS.items():
                    if param not in df.columns:
                        continue
                    current_values = set(df[param].round(10).unique())
                    allowed_values_rounded = {round(x, 10) for x in allowed_values}
                    invalid_values = current_values - allowed_values_rounded
                    if invalid_values:
                        print(
                            f"[警告] 文件 {filename} 列 '{param}' 包含非法值: {invalid_values}"
                        )
                        valid_check = False

            if valid_check:
                n_display = "多个n值" if is_validate else f"n={expected_n}"
                print(f"[成功] 文件 {filename} ({n_display}) 处理完毕。")

            # -------------------------------------------------
            # 步骤7: 加入列表
            # -------------------------------------------------
            if is_validate:
                # validate文件单独存储
                if file_type == "start":
                    validate_start_dfs.append(df)
                else:
                    validate_end_dfs.append(df)
            else:
                # 普通文件合并到主数据集
                if file_type == "start":
                    start_dfs.append(df)
                else:
                    end_dfs.append(df)

        except Exception as e:
            print(f"[异常] 处理文件 {filename} 时发生错误: {str(e)}")

    print("-" * 60)

    # -------------------------------------------------
    # 步骤8: 合并并保存
    # -------------------------------------------------

    # 合并 Start 文件
    if start_dfs:
        merged_start = pd.concat(start_dfs, ignore_index=True)
        # 强制类型转换，确保 n 是整数
        merged_start["n"] = merged_start["n"].astype(int)
        merged_start.sort_values(by=["n", "A", "E", "dd", "hh"], inplace=True)
        output_start = "merged_dataset_start.csv"
        # 移除 float_format 参数，保持原始精度
        merged_start.to_csv(output_start, index=False)
        print(f"合并完成: {output_start} (共 {len(merged_start)} 行)")
    else:
        print("未找到符合条件的 Start 文件。")

    # 合并 End 文件
    if end_dfs:
        merged_end = pd.concat(end_dfs, ignore_index=True)
        merged_end["n"] = merged_end["n"].astype(int)
        merged_end.sort_values(by=["n", "A", "E", "dd", "hh"], inplace=True)
        output_end = "merged_dataset_end.csv"
        # 移除 float_format 参数，保持原始精度
        merged_end.to_csv(output_end, index=False)
        print(f"合并完成: {output_end} (共 {len(merged_end)} 行)")
    else:
        print("未找到符合条件的 End 文件。")

    # 合并 Validate Start 文件
    if validate_start_dfs:
        validate_start = pd.concat(validate_start_dfs, ignore_index=True)
        validate_start["n"] = validate_start["n"].astype(int)
        validate_start.sort_values(by=["n", "A", "E", "dd", "hh"], inplace=True)
        output_validate_start = "validate_dataset_start.csv"
        validate_start.to_csv(output_validate_start, index=False)
        print(f"验证集合并完成: {output_validate_start} (共 {len(validate_start)} 行)")
    else:
        print("未找到符合条件的 Validate Start 文件。")

    # 合并 Validate End 文件
    if validate_end_dfs:
        validate_end = pd.concat(validate_end_dfs, ignore_index=True)
        validate_end["n"] = validate_end["n"].astype(int)
        validate_end.sort_values(by=["n", "A", "E", "dd", "hh"], inplace=True)
        output_validate_end = "validate_dataset_end.csv"
        validate_end.to_csv(output_validate_end, index=False)
        print(f"验证集合并完成: {output_validate_end} (共 {len(validate_end)} 行)")
    else:
        print("未找到符合条件的 Validate End 文件。")


if __name__ == "__main__":
    process_teng_datasets()
