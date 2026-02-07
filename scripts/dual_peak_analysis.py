# 双高峰时间精细分析
# 高峰1: 2025-12-18 至 2025-12-26
# 高峰2: 2026-01-22 至 2026-02-04

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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


def analyze_hourly_in_range(comments, start_date: str, end_date: str):
    """分析指定日期范围内的小时级别统计"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    
    hourly_stats = defaultdict(lambda: {
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
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            timestamp = timestamp.replace(tzinfo=None)  # 移除时区信息
            
            if start <= timestamp < end:
                hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                hourly_stats[hour_key][category] += 1
        except (ValueError, AttributeError):
            continue
    
    return dict(hourly_stats)


def generate_dual_peak_chart(comments, output_dir: Path):
    """生成双高峰精细时间分析图"""
    
    # 高峰1: 2025-12-18 至 2025-12-26
    peak1_start = '2025-12-18'
    peak1_end = '2025-12-21'
    peak1_hourly = analyze_hourly_in_range(comments, peak1_start, peak1_end)
    
    # 高峰2: 2026-01-22 至 2026-02-04
    peak2_start = '2026-01-22'
    peak2_end = '2026-02-04'
    peak2_hourly = analyze_hourly_in_range(comments, peak2_start, peak2_end)
    
    print(f"高峰1 ({peak1_start} ~ {peak1_end}): {sum(sum(v.values()) for v in peak1_hourly.values())} 条评论")
    print(f"高峰2 ({peak2_start} ~ {peak2_end}): {sum(sum(v.values()) for v in peak2_hourly.values())} 条评论")
    
    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    categories = ['MANDELA_EFFECT', 'REBUTTAL', 'CONTENT', 'NOISE']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']
    
    # ===== 高峰1 =====
    sorted_hours1 = sorted(peak1_hourly.keys())
    if sorted_hours1:
        dates1 = [datetime.strptime(h, '%Y-%m-%d %H:00') for h in sorted_hours1]
        
        for cat, color in zip(categories, colors):
            values = [peak1_hourly[h].get(cat, 0) for h in sorted_hours1]
            ax1.plot(dates1, values, label=cat, color=color, linewidth=1.5, alpha=0.8)
            ax1.fill_between(dates1, values, alpha=0.2, color=color)
        
        ax1.set_title(f'高峰1: {peak1_start} ~ {peak1_end} (每小时统计)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间', fontsize=11)
        ax1.set_ylabel('评论数量', fontsize=11)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H时'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 标注最高点
        total_per_hour = {h: sum(peak1_hourly[h].values()) for h in sorted_hours1}
        max_hour = max(total_per_hour, key=total_per_hour.get)
        max_val = total_per_hour[max_hour]
        max_date = datetime.strptime(max_hour, '%Y-%m-%d %H:00')
        ax1.annotate(f'峰值: {max_val}条\n{max_hour}', 
                     xy=(max_date, max_val), 
                     xytext=(max_date, max_val + 10),
                     fontsize=10, ha='center',
                     arrowprops=dict(arrowstyle='->', color='red'))
    
    # ===== 高峰2 =====
    sorted_hours2 = sorted(peak2_hourly.keys())
    if sorted_hours2:
        dates2 = [datetime.strptime(h, '%Y-%m-%d %H:00') for h in sorted_hours2]
        
        for cat, color in zip(categories, colors):
            values = [peak2_hourly[h].get(cat, 0) for h in sorted_hours2]
            ax2.plot(dates2, values, label=cat, color=color, linewidth=1.5, alpha=0.8)
            ax2.fill_between(dates2, values, alpha=0.2, color=color)
        
        ax2.set_title(f'高峰2: {peak2_start} ~ {peak2_end} (每小时统计)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('时间', fontsize=11)
        ax2.set_ylabel('评论数量', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H时'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 标注最高点
        total_per_hour = {h: sum(peak2_hourly[h].values()) for h in sorted_hours2}
        max_hour = max(total_per_hour, key=total_per_hour.get)
        max_val = total_per_hour[max_hour]
        max_date = datetime.strptime(max_hour, '%Y-%m-%d %H:00')
        ax2.annotate(f'峰值: {max_val}条\n{max_hour}', 
                     xy=(max_date, max_val), 
                     xytext=(max_date, max_val + 5),
                     fontsize=10, ha='center',
                     arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    output_path = output_dir / 'dual_peak_hourly.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n已保存: {output_path}")
    
    # 生成统计报告
    print("\n" + "=" * 60)
    print("高峰时段详细统计")
    print("=" * 60)
    
    for peak_name, hourly_data, start, end in [
        ("高峰1", peak1_hourly, peak1_start, peak1_end),
        ("高峰2", peak2_hourly, peak2_start, peak2_end)
    ]:
        print(f"\n【{peak_name}】{start} ~ {end}")
        
        # 按天汇总
        daily_summary = defaultdict(lambda: {'total': 0, 'MANDELA_EFFECT': 0, 'REBUTTAL': 0})
        for hour, cats in hourly_data.items():
            day = hour.split(' ')[0]
            daily_summary[day]['total'] += sum(cats.values())
            daily_summary[day]['MANDELA_EFFECT'] += cats.get('MANDELA_EFFECT', 0)
            daily_summary[day]['REBUTTAL'] += cats.get('REBUTTAL', 0)
        
        print(f"{'日期':12} {'总计':8} {'曼德拉':10} {'反驳':8}")
        for day in sorted(daily_summary.keys()):
            d = daily_summary[day]
            print(f"{day:12} {d['total']:8} {d['MANDELA_EFFECT']:10} {d['REBUTTAL']:8}")
        
        # 最活跃时段
        if hourly_data:
            sorted_by_total = sorted(hourly_data.items(), 
                                     key=lambda x: sum(x[1].values()), 
                                     reverse=True)[:5]
            print(f"\n  最活跃5个小时:")
            for hour, cats in sorted_by_total:
                total = sum(cats.values())
                m = cats.get('MANDELA_EFFECT', 0)
                print(f"    {hour}: {total}条 (曼德拉: {m})")
    
    return peak1_hourly, peak2_hourly


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'processed' / 'llm_analyzed'
    output_dir = script_dir.parent / 'data' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("加载数据...")
    comments = load_analyzed_data(data_dir)
    print(f"已加载 {len(comments)} 条评论\n")
    
    print("生成双高峰精细分析图...")
    generate_dual_peak_chart(comments, output_dir)
    
    print("\n✅ 完成!")


if __name__ == '__main__':
    main()
