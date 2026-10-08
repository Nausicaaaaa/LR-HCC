#!/usr/bin/env python3
"""
生成基线特征表（表1）

读取训练集、验证集、测试集的txt文件，
从data1.xlsx和data2.xlsx中提取临床数据，
统计基线特征并生成表格。

输出：
    LIFT/data/baseline_table.csv
    LIFT/data/baseline_table.md
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats


def load_txt_cases(txt_path):
    """读取txt文件，返回case列表"""
    cases = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) > 0:
                case_name = parts[0].strip()
                if case_name and case_name != 'casename':
                    cases.append(case_name)
    return cases


def load_mapping(mapping_path):
    """读取case_name_mapping文件"""
    df = pd.read_csv(mapping_path, sep='\t', dtype=str)
    df['original_case_name'] = df['original_case_name'].str.strip()
    df['converted_case_name'] = df['converted_case_name'].str.strip()
    return df


def build_data_dict(excel_path):
    """读取Excel文件，建立ID -> 行的字典"""
    df = pd.read_excel(excel_path, header=None)
    # 数据从第3行（索引3）开始
    data_dict = {}
    for idx in range(3, len(df)):
        val = df.iloc[idx, 0]
        if pd.notna(val):
            key = str(val).strip()
            data_dict[key.lower()] = df.iloc[idx]
    return data_dict, df


def find_clinical_data(case_name, conv_to_orig, data_dict, data_dict_lower):
    """查找case对应的临床数据"""
    # 1. 直接匹配
    if case_name in conv_to_orig:
        orig = conv_to_orig[case_name]
        if orig.lower() in data_dict_lower:
            return data_dict_lower[orig.lower()]
    
    # 2. 尝试去掉 _1, _2 后缀
    if '_' in case_name:
        parts = case_name.split('_')
        if parts[-1] in ['1', '2']:
            base = '_'.join(parts[:-1])
            if base in conv_to_orig:
                orig = conv_to_orig[base]
                if orig.lower() in data_dict_lower:
                    return data_dict_lower[orig.lower()]
    
    return None


def safe_float(val):
    """安全转换为浮点数"""
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def safe_str(val):
    """安全转换为字符串"""
    if pd.isna(val):
        return np.nan
    return str(val).strip()


def format_mean_sd(values):
    """格式化 mean ± SD"""
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return "N/A"
    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    return f"{mean:.1f} ± {sd:.1f}"


def format_median_iqr(values):
    """格式化 median (Q1, Q3)"""
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return "N/A"
    median = np.median(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    return f"{median:.1f} ({q1:.1f}, {q3:.1f})"


def format_n_percent(n, total):
    """格式化 n (%)"""
    if total == 0:
        return "0 (0.0)"
    return f"{n} ({n/total*100:.1f})"


def compute_p_value(groups, is_continuous=True):
    """计算多组比较的P值"""
    if is_continuous:
        # 过滤掉空组
        groups = [[v for v in g if not np.isnan(v)] for g in groups]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return np.nan
        
        # 使用Kruskal-Wallis检验（非参数）
        try:
            h, p = stats.kruskal(*groups)
            return p
        except:
            return np.nan
    else:
        # 分类变量使用卡方检验
        groups = [[v for v in g if pd.notna(v) and v is not None and str(v).lower() != 'nan'] for g in groups]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return np.nan
        
        # 获取所有类别
        all_categories = set()
        for g in groups:
            all_categories.update(g)
        
        all_categories = sorted(all_categories)
        if len(all_categories) == 0:
            return np.nan
        
        # 构建列联表
        contingency = []
        for g in groups:
            row = []
            for cat in all_categories:
                row.append(g.count(cat))
            contingency.append(row)
        
        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            return p
        except:
            return np.nan


def main():
    # 路径设置
    # 项目根目录（脚本位于 scripts/analysis/ 下，向上回溯到根）
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'LIFT', 'data')
    labels_dir = os.path.join(data_dir, 'labels')
    
    train_txt = os.path.join(labels_dir, 'train_fold1.txt')
    val_txt = os.path.join(labels_dir, 'val_fold1.txt')
    test_txt = os.path.join(labels_dir, 'test.txt')
    
    data1_path = os.path.join(data_dir, 'data1.xlsx')
    data2_path = os.path.join(data_dir, 'data2.xlsx')
    mapping_path = os.path.join(base_dir, 'case_name_mapping.txt')
    mapping_test_path = os.path.join(base_dir, 'case_name_mapping_test.txt')
    
    # 读取case列表
    train_cases = load_txt_cases(train_txt)
    val_cases = load_txt_cases(val_txt)
    test_cases = load_txt_cases(test_txt)
    
    print(f"训练集: {len(train_cases)} cases")
    print(f"验证集: {len(val_cases)} cases")
    print(f"测试集: {len(test_cases)} cases")
    
    # 读取映射文件
    mapping_df = load_mapping(mapping_path)
    mapping_test_df = load_mapping(mapping_test_path)
    
    # 建立映射字典
    conv_to_orig = dict(zip(mapping_df['converted_case_name'], mapping_df['original_case_name']))
    conv_to_orig_test = dict(zip(mapping_test_df['converted_case_name'], mapping_test_df['original_case_name']))
    
    # 读取临床数据
    data1_dict, data1_df = build_data_dict(data1_path)
    data2_dict, data2_df = build_data_dict(data2_path)
    
    # 列索引（基于data1.xlsx和data2.xlsx的结构）
    COL_AGE = 6
    COL_SEX = 5
    COL_AFP = 7
    COL_PLT = 8
    COL_ALB = 9
    COL_ALT = 10
    COL_AST = 11
    COL_ALP = 12
    COL_TBIL = 13
    COL_PT = 14
    COL_LIRADS = 15
    COL_SIZE = 16
    
    # 提取数据
    def extract_data(cases, conv_to_orig_map, data_dict, is_test=False):
        """提取一组case的临床数据"""
        data = {
            'age': [],
            'sex_m': [],
            'afp': [],
            'plt': [],
            'alb': [],
            'alt': [],
            'ast': [],
            'alp': [],
            'tbil': [],
            'pt': [],
            'size': [],
            'diagnosis': [],  # HCC, noHCC, liangxing
            'lirads': [],
        }
        
        for case in cases:
            if is_test:
                row = find_clinical_data(case, conv_to_orig_map, data_dict, data_dict)
            else:
                row = find_clinical_data(case, conv_to_orig_map, data_dict, data_dict)
            
            if row is None:
                continue
            
            # Age
            data['age'].append(safe_float(row.iloc[COL_AGE]))
            
            # Sex
            sex = safe_str(row.iloc[COL_SEX])
            if sex == 'M' or sex == '男':
                data['sex_m'].append(1)
            elif sex == 'F' or sex == '女':
                data['sex_m'].append(0)
            else:
                data['sex_m'].append(np.nan)
            
            # AFP
            data['afp'].append(safe_float(row.iloc[COL_AFP]))
            
            # PLT (0/1)
            data['plt'].append(safe_float(row.iloc[COL_PLT]))
            
            # Alb
            data['alb'].append(safe_float(row.iloc[COL_ALB]))
            
            # ALT
            data['alt'].append(safe_float(row.iloc[COL_ALT]))
            
            # AST
            data['ast'].append(safe_float(row.iloc[COL_AST]))
            
            # ALP
            data['alp'].append(safe_float(row.iloc[COL_ALP]))
            
            # TBIL
            data['tbil'].append(safe_float(row.iloc[COL_TBIL]))
            
            # PT
            data['pt'].append(safe_float(row.iloc[COL_PT]))
            
            # Size (mm)
            data['size'].append(safe_float(row.iloc[COL_SIZE]))
            
            # Diagnosis (from txt file label1)
            # 需要从txt文件中获取
            data['diagnosis'].append(None)
            
            # LI-RADS (后续从txt文件覆盖，保证与模型训练标签一致)
            data['lirads'].append(None)
        
        return data
    
    # 由于diagnosis需要从txt文件中获取，我们需要先读取txt文件的标签信息
    def get_label_from_txt(txt_path):
        """从txt文件中获取label信息"""
        labels = {}
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    case_name = parts[0].strip()
                    label1 = parts[1].strip()
                    label2 = parts[2].strip()
                    labels[case_name] = {'label1': label1, 'label2': label2}
        return labels
    
    train_labels = get_label_from_txt(train_txt)
    val_labels = get_label_from_txt(val_txt)
    test_labels = get_label_from_txt(test_txt)
    
    # 提取训练集和验证集数据（来自data1.xlsx）
    train_data = extract_data(train_cases, conv_to_orig, data1_dict, is_test=False)
    val_data = extract_data(val_cases, conv_to_orig, data1_dict, is_test=False)
    test_data = extract_data(test_cases, conv_to_orig_test, data2_dict, is_test=True)
    
    # 添加diagnosis和LI-RADS信息（从txt文件读取，保证与模型训练标签一致）
    for data, labels, cases in [(train_data, train_labels, train_cases),
                                (val_data, val_labels, val_cases),
                                (test_data, test_labels, test_cases)]:
        data['diagnosis'] = []
        data['lirads'] = []
        for case in cases:
            if case in labels:
                data['diagnosis'].append(labels[case]['label1'])
                data['lirads'].append(labels[case]['label2'])
            else:
                data['diagnosis'].append(None)
                data['lirads'].append(None)
    
    # 统计函数
    def stat_continuous(data_list):
        """统计连续变量"""
        data_list = [v for v in data_list if not np.isnan(v)]
        if not data_list:
            return "N/A", "N/A", []
        mean = np.mean(data_list)
        sd = np.std(data_list, ddof=1)
        median = np.median(data_list)
        q1 = np.percentile(data_list, 25)
        q3 = np.percentile(data_list, 75)
        return f"{mean:.1f} ± {sd:.1f}", f"{median:.1f} ({q1:.1f}, {q3:.1f})", data_list
    
    def stat_categorical(data_list):
        """统计分类变量"""
        counts = {}
        for v in data_list:
            if pd.notna(v) and v is not None:
                counts[v] = counts.get(v, 0) + 1
        return counts
    
    # 生成统计结果
    results = []
    
    def add_row(characteristic, overall_val, train_val, val_val, test_val, p_val=""):
        if isinstance(p_val, (int, float)) and not np.isnan(p_val):
            p_val = f"{p_val:.5f}"
        results.append({
            'Characteristics': characteristic,
            'Overall': overall_val,
            'Training set': train_val,
            'Val set': val_val,
            'Test set': test_val,
            'P value': p_val
        })
    
    # 1. No. of lesions (病例数)
    total_cases = len(train_cases) + len(val_cases) + len(test_cases)
    add_row("No. of lesions", str(total_cases), str(len(train_cases)), str(len(val_cases)), str(len(test_cases)))
    
    # 2. Age
    all_age = train_data['age'] + val_data['age'] + test_data['age']
    age_overall, age_overall_iqr, _ = stat_continuous(all_age)
    age_train, age_train_iqr, _ = stat_continuous(train_data['age'])
    age_val, age_val_iqr, _ = stat_continuous(val_data['age'])
    age_test, age_test_iqr, _ = stat_continuous(test_data['age'])
    age_p = compute_p_value([train_data['age'], val_data['age'], test_data['age']], is_continuous=True)
    add_row("Age, years", age_overall, age_train, age_val, age_test, age_p)
    
    # 3. Sex
    all_sex = [v for v in train_data['sex_m'] + val_data['sex_m'] + test_data['sex_m'] if not np.isnan(v)]
    train_sex = [v for v in train_data['sex_m'] if not np.isnan(v)]
    val_sex = [v for v in val_data['sex_m'] if not np.isnan(v)]
    test_sex = [v for v in test_data['sex_m'] if not np.isnan(v)]
    
    def fmt_sex(sex_list):
        total = len(sex_list)
        if total == 0:
            return "N/A"
        male = sum(sex_list)
        female = total - male
        return f"{male}/{female} ({male/total*100:.1f}/{female/total*100:.1f})"
    
    sex_p = compute_p_value([train_data['sex_m'], val_data['sex_m'], test_data['sex_m']], is_continuous=False)
    add_row("Sex (M/F)", fmt_sex(all_sex), fmt_sex(train_sex), fmt_sex(val_sex), fmt_sex(test_sex), sex_p)
    
    # 4. AFP
    all_afp = train_data['afp'] + val_data['afp'] + test_data['afp']
    afp_overall, _, _ = stat_continuous(all_afp)
    afp_train, _, _ = stat_continuous(train_data['afp'])
    afp_val, _, _ = stat_continuous(val_data['afp'])
    afp_test, _, _ = stat_continuous(test_data['afp'])
    afp_p = compute_p_value([train_data['afp'], val_data['afp'], test_data['afp']], is_continuous=True)
    add_row("AFP (ng/mL)", afp_overall, afp_train, afp_val, afp_test, afp_p)
    
    # 5. PLT (二值化: 0=≤100, 1=>100)
    all_plt = [v for v in train_data['plt'] + val_data['plt'] + test_data['plt'] if not np.isnan(v)]
    train_plt = [v for v in train_data['plt'] if not np.isnan(v)]
    val_plt = [v for v in val_data['plt'] if not np.isnan(v)]
    test_plt = [v for v in test_data['plt'] if not np.isnan(v)]
    
    def fmt_plt(plt_list):
        total = len(plt_list)
        if total == 0:
            return "N/A"
        leq100 = sum(1 for v in plt_list if v == 0)
        gt100 = sum(1 for v in plt_list if v == 1)
        return f"{leq100}/{gt100} ({leq100/total*100:.1f}/{gt100/total*100:.1f})"
    
    plt_p = compute_p_value([train_data['plt'], val_data['plt'], test_data['plt']], is_continuous=False)
    add_row("PLT (≤100/>100)", fmt_plt(all_plt), fmt_plt(train_plt), fmt_plt(val_plt), fmt_plt(test_plt), plt_p)
    
    # 6. Alb (二值化: 0=≤35, 1=>35)
    all_alb = [v for v in train_data['alb'] + val_data['alb'] + test_data['alb'] if not np.isnan(v)]
    train_alb = [v for v in train_data['alb'] if not np.isnan(v)]
    val_alb = [v for v in val_data['alb'] if not np.isnan(v)]
    test_alb = [v for v in test_data['alb'] if not np.isnan(v)]
    
    def fmt_alb(alb_list):
        total = len(alb_list)
        if total == 0:
            return "N/A"
        leq35 = sum(1 for v in alb_list if v == 0)
        gt35 = sum(1 for v in alb_list if v == 1)
        return f"{leq35}/{gt35} ({leq35/total*100:.1f}/{gt35/total*100:.1f})"
    
    alb_p = compute_p_value([train_data['alb'], val_data['alb'], test_data['alb']], is_continuous=False)
    add_row("Alb (≤35/>35 g/L)", fmt_alb(all_alb), fmt_alb(train_alb), fmt_alb(val_alb), fmt_alb(test_alb), alb_p)
    
    # 7. ALT (二值化: 0=≤50, 1=>50)
    all_alt = [v for v in train_data['alt'] + val_data['alt'] + test_data['alt'] if not np.isnan(v)]
    train_alt = [v for v in train_data['alt'] if not np.isnan(v)]
    val_alt = [v for v in val_data['alt'] if not np.isnan(v)]
    test_alt = [v for v in test_data['alt'] if not np.isnan(v)]
    
    def fmt_alt(alt_list):
        total = len(alt_list)
        if total == 0:
            return "N/A"
        leq50 = sum(1 for v in alt_list if v == 0)
        gt50 = sum(1 for v in alt_list if v == 1)
        return f"{leq50}/{gt50} ({leq50/total*100:.1f}/{gt50/total*100:.1f})"
    
    alt_p = compute_p_value([train_data['alt'], val_data['alt'], test_data['alt']], is_continuous=False)
    add_row("ALT (≤50/>50 U/L)", fmt_alt(all_alt), fmt_alt(train_alt), fmt_alt(val_alt), fmt_alt(test_alt), alt_p)
    
    # 8. AST (二值化: 0=≤40, 1=>40)
    all_ast = [v for v in train_data['ast'] + val_data['ast'] + test_data['ast'] if not np.isnan(v)]
    train_ast = [v for v in train_data['ast'] if not np.isnan(v)]
    val_ast = [v for v in val_data['ast'] if not np.isnan(v)]
    test_ast = [v for v in test_data['ast'] if not np.isnan(v)]
    
    def fmt_ast(ast_list):
        total = len(ast_list)
        if total == 0:
            return "N/A"
        leq40 = sum(1 for v in ast_list if v == 0)
        gt40 = sum(1 for v in ast_list if v == 1)
        return f"{leq40}/{gt40} ({leq40/total*100:.1f}/{gt40/total*100:.1f})"
    
    ast_p = compute_p_value([train_data['ast'], val_data['ast'], test_data['ast']], is_continuous=False)
    add_row("AST (≤40/>40 U/L)", fmt_ast(all_ast), fmt_ast(train_ast), fmt_ast(val_ast), fmt_ast(test_ast), ast_p)
    
    # 9. ALP (二值化: 0=≤40, 1=>40)
    all_alp = [v for v in train_data['alp'] + val_data['alp'] + test_data['alp'] if not np.isnan(v)]
    train_alp = [v for v in train_data['alp'] if not np.isnan(v)]
    val_alp = [v for v in val_data['alp'] if not np.isnan(v)]
    test_alp = [v for v in test_data['alp'] if not np.isnan(v)]
    
    def fmt_alp(alp_list):
        total = len(alp_list)
        if total == 0:
            return "N/A"
        leq40 = sum(1 for v in alp_list if v == 0)
        gt40 = sum(1 for v in alp_list if v == 1)
        return f"{leq40}/{gt40} ({leq40/total*100:.1f}/{gt40/total*100:.1f})"
    
    alp_p = compute_p_value([train_data['alp'], val_data['alp'], test_data['alp']], is_continuous=False)
    add_row("ALP (≤40/>40 U/L)", fmt_alp(all_alp), fmt_alp(train_alp), fmt_alp(val_alp), fmt_alp(test_alp), alp_p)
    
    # 10. TBIL (二值化: 0=≤17, 1=>17)
    all_tbil = [v for v in train_data['tbil'] + val_data['tbil'] + test_data['tbil'] if not np.isnan(v)]
    train_tbil = [v for v in train_data['tbil'] if not np.isnan(v)]
    val_tbil = [v for v in val_data['tbil'] if not np.isnan(v)]
    test_tbil = [v for v in test_data['tbil'] if not np.isnan(v)]
    
    def fmt_tbil(tbil_list):
        total = len(tbil_list)
        if total == 0:
            return "N/A"
        leq17 = sum(1 for v in tbil_list if v == 0)
        gt17 = sum(1 for v in tbil_list if v == 1)
        return f"{leq17}/{gt17} ({leq17/total*100:.1f}/{gt17/total*100:.1f})"
    
    tbil_p = compute_p_value([train_data['tbil'], val_data['tbil'], test_data['tbil']], is_continuous=False)
    add_row("TBIL (≤17/>17 μmol/L)", fmt_tbil(all_tbil), fmt_tbil(train_tbil), fmt_tbil(val_tbil), fmt_tbil(test_tbil), tbil_p)
    
    # 11. PT (二值化: 0=≤13, 1=>13)
    all_pt = [v for v in train_data['pt'] + val_data['pt'] + test_data['pt'] if not np.isnan(v)]
    train_pt = [v for v in train_data['pt'] if not np.isnan(v)]
    val_pt = [v for v in val_data['pt'] if not np.isnan(v)]
    test_pt = [v for v in test_data['pt'] if not np.isnan(v)]
    
    def fmt_pt(pt_list):
        total = len(pt_list)
        if total == 0:
            return "N/A"
        leq13 = sum(1 for v in pt_list if v == 0)
        gt13 = sum(1 for v in pt_list if v == 1)
        return f"{leq13}/{gt13} ({leq13/total*100:.1f}/{gt13/total*100:.1f})"
    
    pt_p = compute_p_value([train_data['pt'], val_data['pt'], test_data['pt']], is_continuous=False)
    add_row("PT (≤13/>13 s)", fmt_pt(all_pt), fmt_pt(train_pt), fmt_pt(val_pt), fmt_pt(test_pt), pt_p)
    
    # 12. Lesion size
    all_size = train_data['size'] + val_data['size'] + test_data['size']
    size_overall, _, _ = stat_continuous(all_size)
    size_train, _, _ = stat_continuous(train_data['size'])
    size_val, _, _ = stat_continuous(val_data['size'])
    size_test, _, _ = stat_continuous(test_data['size'])
    add_row("Lesion size (mm)", "", "", "", "")
    
    # 12a. Size categories (cm)
    def get_size_category_binary(size_data, min_cm, max_cm):
        """判断每个size是否属于指定范围"""
        result = []
        for s in size_data:
            if np.isnan(s):
                result.append(np.nan)
            else:
                s_cm = s / 10
                if min_cm <= s_cm < max_cm:
                    result.append(1)
                else:
                    result.append(0)
        return result
    
    def size_in_range(size_list, min_cm, max_cm):
        """统计在指定范围内的size数量"""
        sizes_cm = [s/10 for s in size_list if not np.isnan(s)]
        if not sizes_cm:
            return "N/A"
        count = sum(1 for s in sizes_cm if min_cm <= s < max_cm)
        total = len(sizes_cm)
        return f"{count} ({count/total*100:.1f})"
    
    # 计算各尺寸范围的P值
    train_size_1_2 = get_size_category_binary(train_data['size'], 1, 2)
    val_size_1_2 = get_size_category_binary(val_data['size'], 1, 2)
    test_size_1_2 = get_size_category_binary(test_data['size'], 1, 2)
    size_1_2_p = compute_p_value([train_size_1_2, val_size_1_2, test_size_1_2], is_continuous=False)
    
    train_size_2_5 = get_size_category_binary(train_data['size'], 2, 5)
    val_size_2_5 = get_size_category_binary(val_data['size'], 2, 5)
    test_size_2_5 = get_size_category_binary(test_data['size'], 2, 5)
    size_2_5_p = compute_p_value([train_size_2_5, val_size_2_5, test_size_2_5], is_continuous=False)
    
    train_size_5_10 = get_size_category_binary(train_data['size'], 5, 10)
    val_size_5_10 = get_size_category_binary(val_data['size'], 5, 10)
    test_size_5_10 = get_size_category_binary(test_data['size'], 5, 10)
    size_5_10_p = compute_p_value([train_size_5_10, val_size_5_10, test_size_5_10], is_continuous=False)
    
    add_row("  1 cm ≤ size < 2 cm", 
            size_in_range(all_size, 1, 2),
            size_in_range(train_data['size'], 1, 2),
            size_in_range(val_data['size'], 1, 2),
            size_in_range(test_data['size'], 1, 2),
            size_1_2_p)
    
    add_row("  2 cm ≤ size < 5 cm", 
            size_in_range(all_size, 2, 5),
            size_in_range(train_data['size'], 2, 5),
            size_in_range(val_data['size'], 2, 5),
            size_in_range(test_data['size'], 2, 5),
            size_2_5_p)
    
    add_row("  5 cm ≤ size < 10 cm", 
            size_in_range(all_size, 5, 10),
            size_in_range(train_data['size'], 5, 10),
            size_in_range(val_data['size'], 5, 10),
            size_in_range(test_data['size'], 5, 10),
            size_5_10_p)
    
    # 13. Final diagnosis
    all_diag = train_data['diagnosis'] + val_data['diagnosis'] + test_data['diagnosis']
    
    def fmt_diag(diag_list):
        counts = {}
        for d in diag_list:
            if d:
                counts[d] = counts.get(d, 0) + 1
        total = len([d for d in diag_list if d])
        if total == 0:
            return "N/A"
        hcc = counts.get('HCC', 0)
        nohcc = counts.get('noHCC', 0)
        liangxing = counts.get('liangxing', 0)
        return f"{hcc}/{nohcc}/{liangxing} ({hcc/total*100:.1f}/{nohcc/total*100:.1f}/{liangxing/total*100:.1f})"
    
    diag_p = compute_p_value([train_data['diagnosis'], val_data['diagnosis'], test_data['diagnosis']], is_continuous=False)
    add_row("Final diagnosis (HCC/noHCC/liangxing)", fmt_diag(all_diag), fmt_diag(train_data['diagnosis']), fmt_diag(val_data['diagnosis']), fmt_diag(test_data['diagnosis']), diag_p)
    
    # 14. LI-RADS category
    all_lirads = train_data['lirads'] + val_data['lirads'] + test_data['lirads']
    
    def fmt_lirads(lirads_list):
        counts = {}
        for l in lirads_list:
            if l and l != 'nan':
                # Normalize
                l_norm = str(l).strip().upper().replace('  ', ' ')
                counts[l_norm] = counts.get(l_norm, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return "N/A"
        
        lr12 = counts.get('LR 1/2', 0)
        lr3 = counts.get('LR 3', 0)
        lr4 = counts.get('LR 4', 0)
        lr5 = counts.get('LR 5', 0)
        lrm = counts.get('LR M', 0)
        return f"{lr12}/{lr3}/{lr4}/{lr5}/{lrm} ({lr12/total*100:.1f}/{lr3/total*100:.1f}/{lr4/total*100:.1f}/{lr5/total*100:.1f}/{lrm/total*100:.1f})"
    
    lirads_p = compute_p_value([train_data['lirads'], val_data['lirads'], test_data['lirads']], is_continuous=False)
    add_row("LI-RADS (LR 1/2/3/4/5/M)", fmt_lirads(all_lirads), fmt_lirads(train_data['lirads']), fmt_lirads(val_data['lirads']), fmt_lirads(test_data['lirads']), lirads_p)
    
    # 保存结果
    results_df = pd.DataFrame(results)
    
    # CSV输出
    csv_path = os.path.join(data_dir, 'baseline_table.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV已保存: {csv_path}")
    
    # Markdown输出
    md_path = os.path.join(data_dir, 'baseline_table.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 表1：基线特征\n\n")
        f.write("| Characteristics | Overall | Training set | Val set | Test set | P value |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for _, row in results_df.iterrows():
            f.write(f"| {row['Characteristics']} | {row['Overall']} | {row['Training set']} | {row['Val set']} | {row['Test set']} | {row['P value']} |\n")
    
    print(f"Markdown已保存: {md_path}")
    
    # 打印结果
    print("\n=== 基线特征表 ===\n")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
