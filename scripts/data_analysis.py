# 曼德拉效应数据分析脚本
# 统计实体频率 + 分类随时间变化趋势

import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import sys

# 可选：matplotlib绑定可视化
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    HAS_MATPLOTLIB = True
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_MATPLOTLIB = False
    print("提示: 安装 matplotlib 可生成可视化图表")


def load_analyzed_data(data_dir: Path):
    """加载所有LLM分析结果"""
    all_comments = []
    
    json_files = sorted(data_dir.glob('*.json'))
    json_files = [f for f in json_files if not f.name.startswith('_')]
    
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            comments = json.load(f)
            all_comments.extend(comments)
    
    return all_comments


def analyze_entities(comments):
    """统计实体出现频率"""
    entity_counter = Counter()
    
    for comment in comments:
        analysis = comment.get('analysis', {})
        entities = analysis.get('entities', [])
        
        for entity in entities:
            # 标准化处理
            entity = entity.strip().lower()
            if entity:
                entity_counter[entity] += 1
    
    return entity_counter


def analyze_categories_by_time(comments):
    """分析分类随时间的变化"""
    # 按小时统计
    hourly_stats = defaultdict(lambda: {
        'MANDELA_EFFECT': 0,
        'REBUTTAL': 0,
        'CONTENT': 0,
        'NOISE': 0
    })
    
    # 按天统计
    daily_stats = defaultdict(lambda: {
        'MANDELA_EFFECT': 0,
        'REBUTTAL': 0,
        'CONTENT': 0,
        'NOISE': 0
    })
    
    for comment in comments:
        timestamp_str = comment.get('timestamp', '')
        analysis = comment.get('analysis', {})
        category = analysis.get('category', 'NOISE')
        
        if not timestamp_str:
            continue
        
        try:
            # 解析时间戳
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # 按小时
            hour_key = timestamp.strftime('%Y-%m-%d %H:00')
            hourly_stats[hour_key][category] += 1
            
            # 按天
            day_key = timestamp.strftime('%Y-%m-%d')
            daily_stats[day_key][category] += 1
            
        except (ValueError, AttributeError):
            continue
    
    return dict(hourly_stats), dict(daily_stats)


def analyze_stance_distribution(comments):
    """分析立场分布"""
    stance_counter = Counter()
    
    for comment in comments:
        analysis = comment.get('analysis', {})
        stance = analysis.get('stance', 'NEUTRAL')
        stance_counter[stance] += 1
    
    return stance_counter


def analyze_depth_distribution(comments):
    """分析评论深度分布"""
    depth_category = defaultdict(lambda: Counter())
    
    for comment in comments:
        depth = comment.get('depth', 0)
        analysis = comment.get('analysis', {})
        category = analysis.get('category', 'NOISE')
        
        depth_category[depth][category] += 1
    
    return dict(depth_category)


def print_report(comments, entity_counter, hourly_stats, daily_stats, 
                 stance_counter, depth_category):
    """打印分析报告"""
    print("=" * 70)
    print("曼德拉效应评论数据分析报告")
    print("=" * 70)
    
    # 基本统计
    total = len(comments)
    category_counter = Counter()
    for comment in comments:
        analysis = comment.get('analysis', {})
        category = analysis.get('category', 'NOISE')
        category_counter[category] += 1
    
    print(f"\n📊 基本统计")
    print(f"  总评论数: {total}")
    print()
    
    # 分类统计
    print("📈 分类分布:")
    for cat in ['MANDELA_EFFECT', 'REBUTTAL', 'CONTENT', 'NOISE']:
        count = category_counter.get(cat, 0)
        pct = count / max(total, 1) * 100
        bar = '█' * int(pct / 2)
        print(f"  {cat:18} {count:5} ({pct:5.1f}%) {bar}")
    print()
    
    # 立场统计
    print("🎯 立场分布:")
    for stance, count in stance_counter.most_common():
        pct = count / max(total, 1) * 100
        print(f"  {stance:10} {count:5} ({pct:5.1f}%)")
    print()
    
    # 实体TOP 30
    print("🏷️  高频实体 (Top 30):")
    for i, (entity, count) in enumerate(entity_counter.most_common(30), 1):
        print(f"  {i:2}. {entity:30} {count:4}")
    print()
    
    # 评论深度与分类关系
    print("📏 评论深度与分类关系:")
    print(f"  {'深度':6} {'MANDELA':12} {'REBUTTAL':12} {'CONTENT':12} {'NOISE':12}")
    for depth in sorted(depth_category.keys()):
        cats = depth_category[depth]
        m = cats.get('MANDELA_EFFECT', 0)
        r = cats.get('REBUTTAL', 0)
        c = cats.get('CONTENT', 0)
        n = cats.get('NOISE', 0)
        print(f"  {depth:6} {m:12} {r:12} {c:12} {n:12}")
    print()
    
    # 时间趋势 (按天)
    print("📅 每日分类变化:")
    sorted_days = sorted(daily_stats.keys())
    print(f"  {'日期':12} {'MANDELA':10} {'REBUTTAL':10} {'CONTENT':10} {'NOISE':10} {'总计':8}")
    for day in sorted_days:
        cats = daily_stats[day]
        m = cats.get('MANDELA_EFFECT', 0)
        r = cats.get('REBUTTAL', 0)
        c = cats.get('CONTENT', 0)
        n = cats.get('NOISE', 0)
        total_day = m + r + c + n
        print(f"  {day:12} {m:10} {r:10} {c:10} {n:10} {total_day:8}")
    print()
    
    # 曼德拉效应相关实体分析
    print("🔍 曼德拉效应相关关键词:")
    mandela_keywords = [
        '记得', '记忆', '以前', '看过', '印象', '感觉', '好像',
        '毕导', '视频', '博主', '三年前', '几年前'
    ]
    for kw in mandela_keywords:
        # 查找包含此关键词的实体
        related = [(e, c) for e, c in entity_counter.items() if kw in e]
        if related:
            total_count = sum(c for _, c in related)
            print(f"  '{kw}': {total_count} 次 - {[e for e, _ in related[:5]]}")
    
    print("\n" + "=" * 70)


def generate_visualizations(daily_stats, entity_counter, output_dir: Path):
    """生成可视化图表"""
    if not HAS_MATPLOTLIB:
        print("跳过可视化: matplotlib 未安装")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 图1: 每日分类变化趋势
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sorted_days = sorted(daily_stats.keys())
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in sorted_days]
    
    categories = ['MANDELA_EFFECT', 'REBUTTAL', 'CONTENT', 'NOISE']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']
    
    for cat, color in zip(categories, colors):
        values = [daily_stats[d].get(cat, 0) for d in sorted_days]
        ax.plot(dates, values, marker='o', label=cat, color=color, linewidth=2)
    
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('评论数量', fontsize=12)
    ax.set_title('曼德拉效应评论分类随时间变化趋势', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart1_path = output_dir / 'category_trend.png'
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f"  已保存: {chart1_path}")
    
    # 图2: 实体词云/条形图 (Top 20)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_entities = entity_counter.most_common(20)
    entities = [e for e, _ in top_entities]
    counts = [c for _, c in top_entities]
    
    bars = ax.barh(range(len(entities)), counts, color='#3498db')
    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels(entities)
    ax.invert_yaxis()
    ax.set_xlabel('出现次数', fontsize=12)
    ax.set_title('高频实体 Top 20', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)
    
    plt.tight_layout()
    
    chart2_path = output_dir / 'entity_frequency.png'
    plt.savefig(chart2_path, dpi=150)
    plt.close()
    print(f"  已保存: {chart2_path}")
    
    # 图3: 分类占比饼图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    category_totals = defaultdict(int)
    for day_stats in daily_stats.values():
        for cat, count in day_stats.items():
            category_totals[cat] += count
    
    labels = list(category_totals.keys())
    sizes = list(category_totals.values())
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90,
        explode=[0.05 if l == 'MANDELA_EFFECT' else 0 for l in labels]
    )
    ax.set_title('评论分类占比', fontsize=14, fontweight='bold')
    
    chart3_path = output_dir / 'category_pie.png'
    plt.savefig(chart3_path, dpi=150)
    plt.close()
    print(f"  已保存: {chart3_path}")


def save_report_json(comments, entity_counter, hourly_stats, daily_stats,
                     stance_counter, depth_category, output_path: Path):
    """保存分析报告为JSON"""
    # 分类统计
    category_counter = Counter()
    for comment in comments:
        analysis = comment.get('analysis', {})
        category = analysis.get('category', 'NOISE')
        category_counter[category] += 1
    
    report = {
        'summary': {
            'total_comments': len(comments),
            'category_distribution': dict(category_counter),
            'stance_distribution': dict(stance_counter),
        },
        'entities': {
            'top_50': entity_counter.most_common(50),
            'total_unique': len(entity_counter),
        },
        'time_series': {
            'daily': daily_stats,
            'hourly': hourly_stats,
        },
        'depth_analysis': {
            str(k): dict(v) for k, v in depth_category.items()
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"  已保存: {output_path}")


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'processed' / 'llm_analyzed'
    output_dir = script_dir.parent / 'data' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("加载数据...")
    comments = load_analyzed_data(data_dir)
    print(f"已加载 {len(comments)} 条评论")
    
    if not comments:
        print("错误: 没有找到分析数据，请先运行 batch_analyze.py")
        return
    
    print("分析中...")
    
    # 各项分析
    entity_counter = analyze_entities(comments)
    hourly_stats, daily_stats = analyze_categories_by_time(comments)
    stance_counter = analyze_stance_distribution(comments)
    depth_category = analyze_depth_distribution(comments)
    
    # 打印报告
    print_report(comments, entity_counter, hourly_stats, daily_stats,
                 stance_counter, depth_category)
    
    # 保存JSON报告
    print("\n保存报告...")
    save_report_json(comments, entity_counter, hourly_stats, daily_stats,
                     stance_counter, depth_category,
                     output_dir / 'mandela_effect_report.json')
    
    # 生成可视化
    print("\n生成可视化图表...")
    generate_visualizations(daily_stats, entity_counter, output_dir)
    
    print("\n✅ 分析完成!")


if __name__ == '__main__':
    main()
