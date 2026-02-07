# 批量处理所有评论 - 曼德拉效应分析
# 使用DeepSeek API分析所有2616条评论

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.llm_analyzer import MandelaEffectAnalyzer, LLM_BACKEND


def batch_analyze_all():
    script_dir = Path(__file__).parent
    input_dir = script_dir.parent / 'data' / 'processed' / 'with_context'
    output_dir = script_dir.parent / 'data' / 'processed' / 'llm_analyzed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查已处理的文件
    processed_files = set(f.name for f in output_dir.glob('*.json') if not f.name.startswith('_'))
    
    # 获取所有待处理文件
    all_files = sorted(input_dir.glob('*.json'))
    all_files = [f for f in all_files if f.name != '_meta.json']
    
    # 过滤已处理的文件
    pending_files = [f for f in all_files if f.name not in processed_files]
    
    print("=" * 60)
    print("曼德拉效应评论批量分析")
    print(f"LLM后端: {LLM_BACKEND}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"总文件数: {len(all_files)}")
    print(f"已处理: {len(processed_files)}")
    print(f"待处理: {len(pending_files)}")
    print("=" * 60)
    
    if not pending_files:
        print("所有文件已处理完毕!")
        return
    
    # 分析器
    analyzer = MandelaEffectAnalyzer()
    
    # 统计
    total_comments = 0
    category_counts = {
        'MANDELA_EFFECT': 0,
        'REBUTTAL': 0,
        'CONTENT': 0,
        'NOISE': 0
    }
    
    start_time = time.time()
    
    try:
        for i, filepath in enumerate(pending_files):
            file_start = time.time()
            
            with open(filepath, 'r', encoding='utf-8') as f:
                comments = json.load(f)
            
            print(f"\n[{i+1}/{len(pending_files)}] {filepath.name} ({len(comments)} 条评论)")
            
            results = []
            for j, comment in enumerate(comments):
                processed = analyzer.analyze_comment(comment)
                results.append(asdict(processed))
                
                # 更新统计
                category = processed.analysis.get('category', 'NOISE') if processed.analysis else 'NOISE'
                if category in category_counts:
                    category_counts[category] += 1
                
                # 进度条
                progress = (j + 1) / len(comments) * 100
                print(f"\r  进度: {progress:.0f}% [{j+1}/{len(comments)}]", end='', flush=True)
                
                # API限流
                time.sleep(0.2)
            
            print()  # 换行
            
            # 保存结果
            output_file = output_dir / filepath.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            total_comments += len(results)
            elapsed = time.time() - file_start
            print(f"  ✓ 完成 (耗时 {elapsed:.1f}s)")
            
            # 每10个文件显示整体进度
            if (i + 1) % 10 == 0:
                overall_elapsed = time.time() - start_time
                rate = total_comments / overall_elapsed
                print(f"\n--- 整体进度: {i+1}/{len(pending_files)} 文件, {total_comments} 条评论, {rate:.1f} 条/秒 ---")
                print(f"    分类统计: {category_counts}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断处理")
    
    finally:
        # 最终统计
        elapsed = time.time() - start_time
        stats = analyzer.get_stats()
        
        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"  总耗时: {elapsed / 60:.1f} 分钟")
        print(f"  处理评论数: {stats['processed']}")
        print(f"  成功解析: {stats['success']} ({stats['success_rate']})")
        print(f"  解析失败: {stats['failed']}")
        print(f"  LLM请求数: {stats['llm_requests']}")
        print(f"  LLM错误数: {stats['llm_errors']}")
        print("\n分类统计:")
        for cat, count in category_counts.items():
            pct = count / max(total_comments, 1) * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")
        print("=" * 60)
        
        # 保存统计信息
        stats_file = output_dir / '_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_files': len(pending_files),
                'total_comments': total_comments,
                'category_counts': category_counts,
                'llm_stats': stats,
                'elapsed_seconds': elapsed
            }, f, ensure_ascii=False, indent=2)
        print(f"\n统计信息已保存到: {stats_file}")


if __name__ == '__main__':
    batch_analyze_all()
