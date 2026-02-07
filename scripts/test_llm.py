# 快速测试 - 分析置顶评论（曼德拉效应核心讨论）

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.llm_analyzer import MandelaEffectAnalyzer, LLM_BACKEND

def test_top_comment():
    script_dir = Path(__file__).parent
    input_dir = script_dir.parent / 'data' / 'processed' / 'with_context'
    output_dir = script_dir.parent / 'data' / 'processed' / 'llm_analyzed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载置顶评论
    top_file = input_dir / 'top_283848159169.json'
    
    print("=" * 60)
    print("测试分析置顶评论 (曼德拉效应核心)")
    print(f"LLM后端: {LLM_BACKEND}")
    print("=" * 60)
    
    with open(top_file, 'r', encoding='utf-8') as f:
        comments = json.load(f)
    
    print(f"评论数: {len(comments)}")
    print(f"只测试前10条...\n")
    
    analyzer = MandelaEffectAnalyzer()
    
    results = []
    for i, comment in enumerate(comments[:10]):
        print(f"[{i+1}/10] 分析中...")
        print(f"  作者: {comment['author']}")
        print(f"  内容: {comment['text'][:60]}...")
        
        processed = analyzer.analyze_comment(comment)
        results.append({
            'id': processed.node_id,
            'author': processed.author,
            'raw_text': processed.raw_text,
            'analysis': processed.analysis
        })
        
        analysis = processed.analysis
        print(f"  → 分类: {analysis['category']}")
        print(f"  → 立场: {analysis['stance']}")
        print(f"  → 重写: {analysis['rewrite_text'][:80]}...")
        print(f"  → 实体: {analysis['entities']}")
        print()
    
    # 保存测试结果
    test_output = output_dir / '_test_top_10.json'
    with open(test_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存到: {test_output}")
    
    # 统计
    stats = analyzer.get_stats()
    print(f"\n统计: {stats}")


if __name__ == '__main__':
    test_top_comment()
