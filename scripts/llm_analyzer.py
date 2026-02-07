# LLM 交互层 - 曼德拉效应评论分析
# 支持 DeepSeek API 和本地 Ollama (Llama/Qwen)

import json
import re
import requests
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEEPSEEK_API_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    OLLAMA_API_URL, OLLAMA_MODEL,
    LLM_BACKEND, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
    MAX_CONTEXT_CHARS
)


# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT = """你是一个语义分析专家，专门研究网络评论中的"曼德拉效应"（集体虚假记忆）。
你的任务是读取一段评论对话，并输出严格的 JSON 格式分析结果。

请执行以下步骤：
1. **分类 (Category)**:
   - `MANDELA_EFFECT`: 评论者声称自己有特定记忆（如"我记得以前看过"、"是蓝色的"），或者在讨论记忆的不可靠性。
   - `REBUTTAL`: 评论者反驳虚假记忆，提供事实依据（如"那是另一个博主"、"你记错了"）。
   - `CONTENT`: 仅讨论视频内容（物理、面条、搞笑），与记忆偏差无关。
   - `NOISE`: 无意义的表情、纯赞同(+1)、攻击性语言或无关内容。

2. **重写 (Rewrite) [至关重要]**:
   - 很多评论是简短的回复（如"我也是"、"不对"）。
   - 你必须结合【上下文】，将其重写为一个**完整的、独立的陈述句**。
   - 示例：
     - 输入："我也是" (上下文：楼主说记得是蓝衣服) -> 重写："我也记得博主以前穿的是蓝衣服。"
     - 输入："不对" (上下文：楼主说记得是蓝衣服) -> 重写："我不认为博主穿过蓝衣服，是你记错了。"

3. **立场 (Stance)**:
   - `SUPPORT`: 支持曼德拉效应/虚假记忆的存在。
   - `REFUTE`: 反对虚假记忆，坚持客观事实。
   - `NEUTRAL`: 中立或无关。

4. **细节提取 (Entities)**:
   - 提取评论中提到的具体记忆细节（如：时间、颜色、衣着、其他博主名字）。

请仅输出 JSON，不要包含 Markdown 代码块标记。格式如下：
{
  "category": "MANDELA_EFFECT|REBUTTAL|CONTENT|NOISE",
  "rewrite_text": "重写后的完整陈述句",
  "stance": "SUPPORT|REFUTE|NEUTRAL",
  "entities": ["提取的实体1", "提取的实体2"]
}"""


# ============================================================
# User Prompt 模板
# ============================================================
USER_PROMPT_TEMPLATE = """[上下文对话]
{context}

[当前待分析评论]
作者: {author}
内容: "{text}"

请分析[当前待分析评论]。"""


# ============================================================
# 数据结构
# ============================================================
@dataclass
class AnalysisResult:
    """LLM分析结果"""
    category: str  # MANDELA_EFFECT | REBUTTAL | CONTENT | NOISE
    rewrite_text: str
    stance: str  # SUPPORT | REFUTE | NEUTRAL
    entities: List[str]


@dataclass
class ProcessedComment:
    """处理后的评论数据"""
    node_id: str
    raw_text: str
    author: str
    depth: int
    like_count: int
    timestamp: str
    
    # LLM分析结果
    analysis: Optional[Dict[str, Any]]
    
    # 图谱元数据
    graph_meta: Dict[str, Any]


# ============================================================
# JSON 解析容错
# ============================================================
def repair_json(text: str) -> str:
    """尝试修复常见的JSON格式错误"""
    # 移除可能的markdown代码块标记
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    
    # 移除开头的非JSON字符
    match = re.search(r'\{', text)
    if match:
        text = text[match.start():]
    
    # 移除结尾的非JSON字符
    match = re.search(r'\}[^}]*$', text)
    if match:
        text = text[:match.end() - len(match.group()) + 1]
    
    # 修复常见问题
    # 1. 单引号改双引号
    # 2. 末尾多余逗号
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    
    return text


def parse_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """解析LLM响应，带容错"""
    try:
        # 直接解析
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # 尝试修复后解析
    try:
        repaired = repair_json(response_text)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 尝试使用 json_repair 库 (如果安装了)
    try:
        from json_repair import repair_json as jr_repair
        repaired = jr_repair(response_text)
        return json.loads(repaired)
    except (ImportError, json.JSONDecodeError):
        pass
    
    # 最后尝试：提取JSON部分
    try:
        match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    
    return None


# ============================================================
# 上下文截断
# ============================================================
def truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    截断上下文，保留Root和Direct Parent
    """
    if len(context) <= max_chars:
        return context
    
    lines = context.split('\n')
    
    if len(lines) <= 2:
        # 只有Root和Parent，直接截断
        return context[:max_chars] + "..."
    
    # 保留第一行(Root)和最后一行(Direct Parent)
    root_line = lines[0]
    parent_line = lines[-1]
    
    # 计算剩余空间
    reserved = len(root_line) + len(parent_line) + 50  # 50 for [...] marker
    
    if reserved >= max_chars:
        # 空间不够，截断Root和Parent
        half = max_chars // 2
        root_line = root_line[:half] + "..."
        parent_line = parent_line[:half] + "..."
        return f"{root_line}\n[...]\n{parent_line}"
    
    # 中间内容
    middle_budget = max_chars - reserved
    middle_lines = lines[1:-1]
    
    if middle_lines:
        middle_text = "\n".join(middle_lines)
        if len(middle_text) > middle_budget:
            middle_text = f"[...省略 {len(middle_lines)} 条中间回复...]"
    else:
        middle_text = ""
    
    result_lines = [root_line]
    if middle_text:
        result_lines.append(middle_text)
    result_lines.append(parent_line)
    
    return "\n".join(result_lines)


# ============================================================
# LLM 客户端
# ============================================================
class LLMClient:
    """LLM客户端，支持多种后端"""
    
    def __init__(self, backend: str = LLM_BACKEND):
        self.backend = backend
        self.request_count = 0
        self.error_count = 0
    
    def call(self, user_prompt: str) -> Optional[str]:
        """调用LLM获取响应"""
        if self.backend == "deepseek":
            return self._call_deepseek(user_prompt)
        elif self.backend == "ollama":
            return self._call_ollama(user_prompt)
        else:
            raise ValueError(f"未知的LLM后端: {self.backend}")
    
    def _call_deepseek(self, user_prompt: str) -> Optional[str]:
        """调用DeepSeek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
        
        try:
            self.request_count += 1
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=LLM_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            print(f"DeepSeek API 错误: {e}")
            return None
    
    def _call_ollama(self, user_prompt: str) -> Optional[str]:
        """调用本地Ollama"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
            }
        }
        
        try:
            self.request_count += 1
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=LLM_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
        
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            print(f"Ollama 错误: {e}")
            return None


# ============================================================
# 评论分析器
# ============================================================
class MandelaEffectAnalyzer:
    """曼德拉效应评论分析器"""
    
    def __init__(self, backend: str = LLM_BACKEND):
        self.llm = LLMClient(backend=backend)
        self.processed_count = 0
        self.success_count = 0
        self.failed_ids = []
    
    def analyze_comment(self, comment: Dict[str, Any]) -> ProcessedComment:
        """分析单条评论"""
        node_id = comment['id']
        raw_text = comment['text']
        author = comment['author']
        depth = comment['depth']
        context = comment.get('context', '')
        
        # 截断上下文
        truncated_context = truncate_context(context) if context else "(无上下文，这是楼主评论)"
        
        # 构建用户提示
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=truncated_context,
            author=author,
            text=raw_text
        )
        
        # 调用LLM
        llm_response = self.llm.call(user_prompt)
        
        # 解析响应
        analysis = None
        if llm_response:
            parsed = parse_llm_response(llm_response)
            if parsed:
                analysis = {
                    "category": parsed.get("category", "NOISE"),
                    "rewrite_text": parsed.get("rewrite_text", raw_text),
                    "stance": parsed.get("stance", "NEUTRAL"),
                    "entities": parsed.get("entities", [])
                }
                self.success_count += 1
            else:
                self.failed_ids.append(node_id)
                # 默认值
                analysis = {
                    "category": "NOISE",
                    "rewrite_text": raw_text,
                    "stance": "NEUTRAL",
                    "entities": [],
                    "_parse_error": True,
                    "_raw_response": llm_response[:500]  # 保留部分原始响应用于调试
                }
        else:
            self.failed_ids.append(node_id)
            analysis = {
                "category": "NOISE",
                "rewrite_text": raw_text,
                "stance": "NEUTRAL",
                "entities": [],
                "_api_error": True
            }
        
        self.processed_count += 1
        
        return ProcessedComment(
            node_id=node_id,
            raw_text=raw_text,
            author=author,
            depth=depth,
            like_count=comment.get('like_count', 0),
            timestamp=comment.get('timestamp', ''),
            analysis=analysis,
            graph_meta={
                "root_id": comment.get('root_id', node_id),
                "parent_id": comment.get('parent_id'),
                "is_leaf": True  # 将在后续处理中更新
            }
        )
    
    def analyze_batch(self, comments: List[Dict[str, Any]], 
                      delay: float = 0.5,
                      progress_callback=None) -> List[ProcessedComment]:
        """批量分析评论"""
        results = []
        
        for i, comment in enumerate(comments):
            result = self.analyze_comment(comment)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(comments), result)
            
            # 请求间隔，避免限流
            if i < len(comments) - 1:
                time.sleep(delay)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "processed": self.processed_count,
            "success": self.success_count,
            "failed": len(self.failed_ids),
            "success_rate": f"{self.success_count / max(self.processed_count, 1) * 100:.1f}%",
            "llm_requests": self.llm.request_count,
            "llm_errors": self.llm.error_count,
        }


# ============================================================
# 主程序
# ============================================================
def process_comment_file(filepath: Path, analyzer: MandelaEffectAnalyzer) -> List[Dict]:
    """处理单个评论文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        comments = json.load(f)
    
    results = []
    for comment in comments:
        processed = analyzer.analyze_comment(comment)
        results.append(asdict(processed))
    
    return results


def main():
    script_dir = Path(__file__).parent
    input_dir = script_dir.parent / 'data' / 'processed' / 'with_context'
    output_dir = script_dir.parent / 'data' / 'processed' / 'llm_analyzed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("曼德拉效应评论分析器")
    print(f"LLM后端: {LLM_BACKEND}")
    print(f"温度: {LLM_TEMPERATURE}")
    print("=" * 60)
    
    # 获取所有待处理文件
    json_files = sorted(input_dir.glob('*.json'))
    json_files = [f for f in json_files if f.name != '_meta.json']
    
    print(f"找到 {len(json_files)} 个评论文件")
    
    # 分析器
    analyzer = MandelaEffectAnalyzer(backend=LLM_BACKEND)
    
    # 处理文件 (示例：只处理前3个文件用于测试)
    test_mode = True
    files_to_process = json_files[:3] if test_mode else json_files
    
    print(f"\n{'[测试模式] ' if test_mode else ''}处理 {len(files_to_process)} 个文件...")
    
    total_comments = 0
    
    for i, filepath in enumerate(files_to_process):
        print(f"\n处理 [{i+1}/{len(files_to_process)}]: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            comments = json.load(f)
        
        print(f"  评论数: {len(comments)}")
        
        results = []
        for j, comment in enumerate(comments):
            processed = analyzer.analyze_comment(comment)
            results.append(asdict(processed))
            
            # 进度显示
            category = processed.analysis.get('category', 'UNKNOWN') if processed.analysis else 'ERROR'
            print(f"    [{j+1}/{len(comments)}] {comment['author'][:10]}: {category}")
            
            # API请求间隔
            time.sleep(0.3)
        
        # 保存结果
        output_file = output_dir / filepath.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        total_comments += len(results)
        print(f"  ✓ 已保存到 {output_file.name}")
    
    # 统计
    stats = analyzer.get_stats()
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"  处理评论数: {stats['processed']}")
    print(f"  成功解析: {stats['success']} ({stats['success_rate']})")
    print(f"  解析失败: {stats['failed']}")
    print(f"  LLM请求数: {stats['llm_requests']}")
    print(f"  LLM错误数: {stats['llm_errors']}")
    print("=" * 60)
    
    if test_mode:
        print("\n⚠️  测试模式：只处理了前3个文件")
        print("要处理全部文件，请将 test_mode 设为 False")


if __name__ == '__main__':
    main()
