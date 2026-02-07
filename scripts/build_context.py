# 评论树构建器 - 森林构建 + 上下文路径生成
# 用于为LLM生成带有祖先链路的上下文

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CommentNode:
    """评论节点"""
    id: str
    text: str
    author: str
    author_mid: str
    timestamp: str
    like_count: int
    parent_id: Optional[str] = None
    root_id: Optional[str] = None
    children: List['CommentNode'] = field(default_factory=list)
    depth: int = 0


class CommentForestBuilder:
    """
    评论森林构建器
    将扁平的评论列表转换为树结构，复杂度 O(N)
    """
    
    def __init__(self):
        self.node_map: Dict[str, CommentNode] = {}
        self.roots: List[CommentNode] = []
    
    def build_from_file(self, filepath: Path) -> Optional[CommentNode]:
        """从单个JSON文件构建评论树"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self.build_tree(data)
    
    def build_tree(self, data: dict) -> CommentNode:
        """
        构建评论树 O(N)
        
        步骤:
        1. 创建 Map<ID, Node>
        2. 遍历列表，根据parent_id建立父子关系
        """
        # 创建根节点
        root = CommentNode(
            id=data['id'],
            text=data['text'],
            author=data['author'],
            author_mid=data['author_mid'],
            timestamp=data['timestamp'],
            like_count=data['like_count'],
            parent_id=None,
            root_id=data['id'],
            depth=0
        )
        
        self.node_map = {root.id: root}
        
        # 如果没有回复，直接返回
        replies = data.get('replies', [])
        if not replies:
            return root
        
        # Step 1: 创建所有节点的Map
        for reply in replies:
            node = CommentNode(
                id=reply['id'],
                text=reply['text'],
                author=reply['author'],
                author_mid=reply['author_mid'],
                timestamp=reply['timestamp'],
                like_count=reply['like_count'],
                parent_id=reply.get('parent_id'),
                root_id=reply.get('root_id', root.id),
            )
            self.node_map[node.id] = node
        
        # Step 2: 建立父子关系
        for reply in replies:
            node = self.node_map[reply['id']]
            parent_id = reply.get('parent_id')
            
            if parent_id and parent_id in self.node_map:
                parent = self.node_map[parent_id]
                parent.children.append(node)
            else:
                # 如果找不到父节点，挂到根节点
                root.children.append(node)
        
        # Step 3: 计算深度 (BFS)
        self._calculate_depths(root)
        
        return root
    
    def _calculate_depths(self, root: CommentNode):
        """计算每个节点的深度"""
        queue = [(root, 0)]
        while queue:
            node, depth = queue.pop(0)
            node.depth = depth
            for child in node.children:
                queue.append((child, depth + 1))


class ContextPathGenerator:
    """
    上下文路径生成器
    为每个节点生成祖先链路，供LLM分析
    """
    
    def __init__(self, max_middle_ancestors: int = 2):
        """
        Args:
            max_middle_ancestors: 中间祖先的最大数量（不含Root和Direct Parent）
        """
        self.max_middle_ancestors = max_middle_ancestors
    
    def get_ancestor_chain(self, node: CommentNode, node_map: Dict[str, CommentNode]) -> List[CommentNode]:
        """获取从Root到Direct Parent的祖先链（不含自己）"""
        chain = []
        current_id = node.parent_id
        
        while current_id and current_id in node_map:
            ancestor = node_map[current_id]
            chain.append(ancestor)
            current_id = ancestor.parent_id
        
        # 反转，使Root在最前面
        chain.reverse()
        return chain
    
    def generate_context(self, node: CommentNode, node_map: Dict[str, CommentNode]) -> str:
        """
        为节点生成上下文文本
        
        Case A: 楼主 (Root, depth=0)
            Context = [空]
        
        Case B: 直接回复 (depth=1)
            Context = 楼主说: "{Root.text}"
        
        Case C: 深层回复 (depth > 1)
            Context = 楼主说: "{Root.text}" \n ... \n 上级回复({Parent.author})说: "{Parent.text}"
        """
        if node.depth == 0:
            # Case A: 楼主
            return ""
        
        ancestors = self.get_ancestor_chain(node, node_map)
        
        if not ancestors:
            return ""
        
        if node.depth == 1:
            # Case B: 直接回复楼主
            root = ancestors[0]
            return f'楼主({root.author})说: "{self._truncate_text(root.text)}"'
        
        # Case C: 深层回复
        root = ancestors[0]
        direct_parent = ancestors[-1]
        
        context_parts = []
        
        # 始终保留Root
        context_parts.append(f'楼主({root.author})说: "{self._truncate_text(root.text)}"')
        
        # 中间层级（如果有）
        middle_ancestors = ancestors[1:-1]  # 不含Root和Direct Parent
        
        if len(middle_ancestors) > self.max_middle_ancestors:
            # 截断中间层级
            context_parts.append(f"...(省略 {len(middle_ancestors) - self.max_middle_ancestors} 条中间回复)...")
            # 只保留最后几条中间祖先
            middle_ancestors = middle_ancestors[-self.max_middle_ancestors:]
        
        for ancestor in middle_ancestors:
            context_parts.append(f'{ancestor.author} 回复说: "{self._truncate_text(ancestor.text)}"')
        
        # 始终保留Direct Parent
        if direct_parent.id != root.id:
            context_parts.append(f'上级回复({direct_parent.author})说: "{self._truncate_text(direct_parent.text)}"')
        
        return "\n".join(context_parts)
    
    def _truncate_text(self, text: str, max_length: int = 200) -> str:
        """截断过长的文本"""
        text = text.replace('\n', ' ').strip()
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def generate_llm_input(self, node: CommentNode, node_map: Dict[str, CommentNode]) -> dict:
        """
        生成供LLM处理的完整输入
        """
        context = self.generate_context(node, node_map)
        
        return {
            'id': node.id,
            'author': node.author,
            'text': node.text,
            'like_count': node.like_count,
            'depth': node.depth,
            'context': context,
            'timestamp': node.timestamp,
        }


def process_comment_tree(filepath: Path, context_generator: ContextPathGenerator) -> List[dict]:
    """
    处理单个评论树文件，为每个节点生成LLM输入
    """
    builder = CommentForestBuilder()
    root = builder.build_from_file(filepath)
    
    results = []
    
    # BFS遍历所有节点
    queue = [root]
    while queue:
        node = queue.pop(0)
        llm_input = context_generator.generate_llm_input(node, builder.node_map)
        results.append(llm_input)
        queue.extend(node.children)
    
    return results


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'raw' / 'comments_v2'
    output_dir = script_dir.parent / 'data' / 'processed' / 'with_context'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    context_generator = ContextPathGenerator(max_middle_ancestors=2)
    
    # 处理所有评论树文件
    json_files = list(data_dir.glob('*.json'))
    json_files = [f for f in json_files if f.name != '_meta.json']
    
    print(f"找到 {len(json_files)} 个评论树文件")
    
    total_comments = 0
    
    for i, filepath in enumerate(json_files):
        results = process_comment_tree(filepath, context_generator)
        total_comments += len(results)
        
        # 保存处理后的结果
        output_file = output_dir / filepath.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        if (i + 1) % 200 == 0:
            print(f"已处理 {i + 1}/{len(json_files)} 个文件...")
    
    print(f"\n处理完成!")
    print(f"  评论树文件: {len(json_files)}")
    print(f"  总评论数: {total_comments}")
    print(f"  输出目录: {output_dir}")
    
    # 示例输出
    print("\n" + "=" * 60)
    print("示例输出 (置顶评论树的前5条):")
    print("=" * 60)
    
    top_file = output_dir / 'top_283848159169.json'
    if top_file.exists():
        with open(top_file, 'r', encoding='utf-8') as f:
            sample = json.load(f)[:5]
        
        for item in sample:
            print(f"\n【{item['author']}】(深度: {item['depth']}, 赞: {item['like_count']})")
            print(f"评论: {item['text'][:100]}...")
            if item['context']:
                print(f"上下文:\n{item['context']}")
            print("-" * 40)


if __name__ == '__main__':
    main()
