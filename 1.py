import pandas as pd
import re


def extract_log_params(file_path):
    # 1. 定义需要提取的参数名
    # 你提到的参数列表（注意：你描述中说了5个，但只列出了4个，如果还有第5个，请添加到这个列表中）
    target_keys = ['zd_kl', 'zc_kl', 'hmm', 'rec']

    # 用于存储提取到的数据
    data_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # 这是一个临时字典，用于存放当前行找到的参数
            current_params = {}
            found_any = False

            for key in target_keys:
                # 2. 正则表达式逻辑：
                # 匹配 key 后跟 (空格/等号/冒号) 再跟 数字(包含小数)
                # 例如支持: zd_kl=0.01, zd_kl: 0.01, zd_kl 0.01
                pattern = f"{key}[\s=:_]+([\d\.]+)"

                match = re.search(pattern, line)
                if match:
                    # 提取数值并转为 float
                    current_params[key] = float(match.group(1))
                    found_any = True

            # 如果这一行包含了我们需要的参数（至少包含一个），就把它加入列表
            # 通常实验Log里，这些参数会出现在同一行 Args Namespace 中
            if found_any:
                # 补全没有找到的key为None，或者根据需求填0
                for key in target_keys:
                    if key not in current_params:
                        current_params[key] = None
                data_list.append(current_params)

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None

    # 3. 转换为 DataFrame 并输出
    if data_list:
        df = pd.DataFrame(data_list)
        # 调整列顺序以匹配你的要求
        df = df[target_keys]
        return df
    else:
        print("未在日志中匹配到相关参数，请检查参数名拼写或日志格式。")
        return pd.DataFrame()


# --- 执行部分 ---

file_name = 'result_long_term_forecast_ETTh1.csv_.txt'

# 提取数据
df_params = extract_log_params(file_name)

# 打印表格
print("-" * 30)
print(f"从 {file_name} 提取的参数表：")
print("-" * 30)
print(df_params)

# 如果需要保存为新的 CSV
# df_params.to_csv('extracted_weights.csv', index=False)