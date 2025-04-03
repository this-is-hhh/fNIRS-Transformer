def calculate_f1score(tn, tp, fn, fp):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score


def calculate_f1_from_file(file_path):
    max_f1 = 0
    max_f1_line = 0
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 9):
                if i + 5 < len(lines) and i + 6 < len(lines):
                    line_6 = lines[i + 5].strip()
                    # print(line_6)
                    line_7 = lines[i + 6].strip()
                    # print(line_7)
                    try:
                        parts_6 = line_6.split('(')[1].split('/')
                        a = int(parts_6[0])
                        b = int(parts_6[1].rstrip(')'))
                        parts_7 = line_7.split('(')[1].split('/')
                        c = int(parts_7[0])
                        d = int(parts_7[1].rstrip(')'))
                    except (IndexError, ValueError):
                        print(f"解析第 {i + 1} 行数据时出错，请检查数据格式。")
                        continue
                    tn = a
                    tp = c
                    fn = b - a
                    fp = d - c
                    f1_score = calculate_f1score(tn, tp, fn, fp)
                    if f1_score >= max_f1:
                        max_f1 = f1_score
                        max_f1_line = i + 1
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"发生未知错误: {e}")

    print(f"最大的 F1 分数是 {max_f1}，位于第 {max_f1_line} 行。")


if __name__ == "__main__":
    for i in range(1,5):
        file_path = f'/data1/zxj_log/save/D/KFold/1086_7regions_ST_GELU_multiscale_50/fNIRS-T/{i}/training_log_run_{i}.txt'
        calculate_f1_from_file(file_path)
    