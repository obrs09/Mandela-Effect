#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
曼德拉效应评论向量化与可视化流水线
Post-Processing Pipeline: Vectorization + UMAP Visualization

功能:
1. 使用 BAAI/bge-m3 对 rewrite_text 进行向量化
2. 使用 UMAP Train-Project 模式降维
3. 生成 3D 静态图 (X=语义, Y=语义, Z=时间)
4. 生成动态 GIF 动画展示曼德拉病毒传播

作者: GitHub Copilot
日期: 2026-02-07
"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 配置
# ============================================================

# 数据路径
DATA_DIR = Path(__file__).parent.parent / "data"
LLM_ANALYZED_DIR = DATA_DIR / "processed" / "llm_analyzed"
OUTPUT_DIR = DATA_DIR / "vectorized"

# Embedding 配置
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# UMAP 配置
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
UMAP_N_COMPONENTS = 2  # 2D for visualization (Z will be time)

# 颜色映射
CATEGORY_COLORS = {
    "MANDELA_EFFECT": "#FF4136",  # 红色 - 幻觉簇
    "REBUTTAL": "#0074D9",        # 蓝色 - 事实簇
    "CONTENT": "#AAAAAA",         # 灰色 - 内容
    "NOISE": "#DDDDDD",           # 浅灰 - 噪音
}

# 动画配置
ANIMATION_DURATION = 20  # 秒
ANIMATION_FPS = 30


class CommentLoader:
    """加载和预处理 LLM 分析后的评论数据"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def load_all_comments(self) -> List[Dict]:
        """加载所有评论数据"""
        comments = []
        json_files = glob.glob(str(self.data_dir / "*.json"))
        
        # 排除统计文件
        exclude_files = {"_stats.json", "_test_top_10.json"}
        
        for filepath in json_files:
            filename = os.path.basename(filepath)
            if filename in exclude_files:
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 每个文件是一个列表（树结构展平）
                if isinstance(data, list):
                    for comment in data:
                        if self._is_valid_comment(comment):
                            comments.append(comment)
                elif isinstance(data, dict):
                    if self._is_valid_comment(data):
                        comments.append(data)
                        
            except Exception as e:
                print(f"加载 {filename} 失败: {e}")
                continue
        
        print(f"✓ 加载 {len(comments)} 条评论")
        return comments
    
    def _is_valid_comment(self, comment: Dict) -> bool:
        """检查评论是否有效（必须有 analysis.rewrite_text）"""
        if "analysis" not in comment:
            return False
        analysis = comment["analysis"]
        if "rewrite_text" not in analysis or not analysis["rewrite_text"]:
            return False
        return True
    
    def prepare_for_embedding(self, comments: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        准备用于 embedding 的文本列表
        返回: (texts, metadata_list)
        """
        texts = []
        metadata = []
        
        for c in comments:
            text = c["analysis"]["rewrite_text"]
            texts.append(text)
            
            # 解析时间戳
            timestamp_str = c.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except:
                timestamp = datetime.now()
            
            meta = {
                "node_id": c.get("node_id", ""),
                "category": c["analysis"].get("category", "NOISE"),
                "stance": c["analysis"].get("stance", "NEUTRAL"),
                "timestamp": timestamp,
                "raw_text": c.get("raw_text", ""),
                "author": c.get("author", ""),
                "depth": c.get("depth", 0),
                "like_count": c.get("like_count", 0),
            }
            metadata.append(meta)
        
        return texts, metadata


class EmbeddingEngine:
    """使用 BGE-M3 模型生成文本向量"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.use_flag_embedding = False
        
    def load_model(self):
        """加载 BGE-M3 模型 - 优先使用 sentence-transformers (更稳定)"""
        # 优先使用 sentence-transformers (更稳定，兼容性更好)
        try:
            from sentence_transformers import SentenceTransformer
            print(f"正在加载 {self.model_name} (sentence-transformers) ...")
            self.model = SentenceTransformer(self.model_name)
            self.use_flag_embedding = False
            print(f"✓ 模型加载完成")
            return
        except ImportError:
            pass
        
        # 备选方案：FlagEmbedding
        try:
            from FlagEmbedding import BGEM3FlagModel
            print(f"正在加载 {self.model_name} (FlagEmbedding) ...")
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True,
                device="cuda"
            )
            self.use_flag_embedding = True
            print(f"✓ 模型加载完成")
        except Exception as e:
            raise RuntimeError(f"无法加载 embedding 模型: {e}")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码文本为向量
        返回: (N, dim) 的 numpy 数组
        """
        if self.model is None:
            self.load_model()
        
        print(f"正在向量化 {len(texts)} 条文本 ...")
        
        if self.use_flag_embedding:
            # FlagEmbedding BGEM3
            result = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=512
            )
            if isinstance(result, dict):
                embeddings = result.get('dense_vecs', result.get('dense'))
            else:
                embeddings = result
        else:
            # SentenceTransformer
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        embeddings = np.array(embeddings)
        print(f"✓ 向量化完成: shape = {embeddings.shape}")
        return embeddings


class UMAPReducer:
    """
    UMAP 降维器 - 使用 Train-Project 模式
    
    策略:
    1. 使用早期数据 fit 基准地图
    2. 使用 transform 投射新数据，保持结构稳定
    """
    
    def __init__(
        self,
        n_neighbors: int = UMAP_N_NEIGHBORS,
        min_dist: float = UMAP_MIN_DIST,
        metric: str = UMAP_METRIC,
        n_components: int = UMAP_N_COMPONENTS,
        random_state: int = UMAP_RANDOM_STATE
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray) -> 'UMAPReducer':
        """
        训练 UMAP 模型（建立基准地图）
        """
        import umap
        
        print(f"正在训练 UMAP (n={len(embeddings)}) ...")
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            n_components=self.n_components,
            random_state=self.random_state,
            verbose=True
        )
        self.reducer.fit(embeddings)
        self.is_fitted = True
        print("✓ UMAP 训练完成")
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        投射数据到已训练的 UMAP 空间
        """
        if not self.is_fitted:
            raise ValueError("UMAP 模型尚未训练，请先调用 fit()")
        
        coords = self.reducer.transform(embeddings)
        return coords
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        一步完成训练和投射
        """
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def save(self, filepath: str):
        """保存训练好的 UMAP 模型"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.reducer, f)
        print(f"✓ UMAP 模型已保存到 {filepath}")
    
    def load(self, filepath: str) -> 'UMAPReducer':
        """加载已训练的 UMAP 模型"""
        import pickle
        with open(filepath, 'rb') as f:
            self.reducer = pickle.load(f)
        self.is_fitted = True
        print(f"✓ UMAP 模型已从 {filepath} 加载")
        return self


class Visualizer3D:
    """3D 静态可视化器 (X=语义, Y=语义, Z=时间)"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_3d_plot(
        self,
        coords_2d: np.ndarray,
        metadata: List[Dict],
        filename: str = "mandela_3d_manifold.html"
    ):
        """
        创建 3D 交互式可视化
        X, Y = UMAP 语义坐标
        Z = 时间戳
        """
        import plotly.graph_objects as go
        import plotly.express as px
        
        # 准备数据
        x = coords_2d[:, 0]
        y = coords_2d[:, 1]
        
        # 将时间转换为数值（天数偏移）
        timestamps = [m["timestamp"] for m in metadata]
        min_time = min(timestamps)
        z = [(t - min_time).total_seconds() / 86400 for t in timestamps]  # 天数
        
        categories = [m["category"] for m in metadata]
        texts = [m["raw_text"][:50] + "..." if len(m["raw_text"]) > 50 else m["raw_text"] 
                 for m in metadata]
        authors = [m["author"] for m in metadata]
        
        # 创建颜色映射
        colors = [CATEGORY_COLORS.get(c, "#DDDDDD") for c in categories]
        
        # 创建 3D 散点图
        fig = go.Figure()
        
        # 分类别添加数据，以便图例分开
        for cat, color in CATEGORY_COLORS.items():
            mask = [c == cat for c in categories]
            if not any(mask):
                continue
                
            cat_x = [x[i] for i, m in enumerate(mask) if m]
            cat_y = [y[i] for i, m in enumerate(mask) if m]
            cat_z = [z[i] for i, m in enumerate(mask) if m]
            cat_texts = [texts[i] for i, m in enumerate(mask) if m]
            cat_authors = [authors[i] for i, m in enumerate(mask) if m]
            
            cat_names = {
                "MANDELA_EFFECT": "曼德拉效应 (幻觉簇)",
                "REBUTTAL": "事实反驳 (事实簇)",
                "CONTENT": "内容讨论",
                "NOISE": "噪音",
            }
            
            fig.add_trace(go.Scatter3d(
                x=cat_x,
                y=cat_y,
                z=cat_z,
                mode='markers',
                name=cat_names.get(cat, cat),
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.7,
                ),
                text=[f"{a}: {t}" for a, t in zip(cat_authors, cat_texts)],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title="曼德拉效应评论语义流形 (3D)",
            scene=dict(
                xaxis_title="语义维度 1",
                yaxis_title="语义维度 2",
                zaxis_title="时间 (天)",
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            width=1200,
            height=800,
        )
        
        # 保存为 HTML
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        print(f"✓ 3D 可视化已保存到 {output_path}")
        
        return fig
    
    def create_static_3d_matplotlib(
        self,
        coords_2d: np.ndarray,
        metadata: List[Dict],
        filename: str = "mandela_3d_static.png"
    ):
        """
        创建 matplotlib 静态 3D 图
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = coords_2d[:, 0]
        y = coords_2d[:, 1]
        
        timestamps = [m["timestamp"] for m in metadata]
        min_time = min(timestamps)
        z = [(t - min_time).total_seconds() / 86400 for t in timestamps]
        
        categories = [m["category"] for m in metadata]
        
        # 分类绘制
        for cat, color in CATEGORY_COLORS.items():
            mask = np.array([c == cat for c in categories])
            if not mask.any():
                continue
            
            cat_names = {
                "MANDELA_EFFECT": "曼德拉效应",
                "REBUTTAL": "事实反驳",
                "CONTENT": "内容讨论",
                "NOISE": "噪音",
            }
            
            ax.scatter(
                x[mask], y[mask], np.array(z)[mask],
                c=color, label=cat_names.get(cat, cat),
                alpha=0.6, s=10
            )
        
        ax.set_xlabel('语义维度 1')
        ax.set_ylabel('语义维度 2')
        ax.set_zlabel('时间 (天)')
        ax.set_title('曼德拉效应评论语义流形')
        ax.legend(loc='upper left')
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 静态 3D 图已保存到 {output_path}")


class AnimationGenerator:
    """动态 GIF/视频生成器 - 展示曼德拉病毒传播"""
    
    def __init__(self, output_dir: Path, duration: int = ANIMATION_DURATION, fps: int = ANIMATION_FPS):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration
        self.fps = fps
    
    def create_animation(
        self,
        coords_2d: np.ndarray,
        metadata: List[Dict],
        filename: str = "mandela_spread.gif"
    ):
        """
        创建累积动画 - 展示评论随时间增长
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 按时间排序
        timestamps = [m["timestamp"] for m in metadata]
        sorted_indices = np.argsort(timestamps)
        
        x = coords_2d[sorted_indices, 0]
        y = coords_2d[sorted_indices, 1]
        categories = [metadata[i]["category"] for i in sorted_indices]
        sorted_timestamps = [timestamps[i] for i in sorted_indices]
        
        # 计算帧数和每帧显示的数据点
        total_frames = self.duration * self.fps
        n_points = len(x)
        points_per_frame = max(1, n_points // total_frames)
        
        print(f"生成动画: {total_frames} 帧, 每帧约 {points_per_frame} 个点")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 初始化散点图
        scatter_data = {cat: {"x": [], "y": []} for cat in CATEGORY_COLORS.keys()}
        scatters = {}
        
        for cat, color in CATEGORY_COLORS.items():
            scatters[cat] = ax.scatter([], [], c=color, s=15, alpha=0.6, label={
                "MANDELA_EFFECT": "曼德拉效应",
                "REBUTTAL": "事实反驳", 
                "CONTENT": "内容讨论",
                "NOISE": "噪音"
            }.get(cat, cat))
        
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
        ax.set_xlabel('语义维度 1')
        ax.set_ylabel('语义维度 2')
        ax.legend(loc='upper right')
        
        title = ax.set_title('曼德拉效应传播 - 初始化中...')
        counter = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          verticalalignment='top', fontsize=12)
        
        def init():
            for scat in scatters.values():
                scat.set_offsets(np.empty((0, 2)))
            return list(scatters.values()) + [title, counter]
        
        def update(frame):
            # 计算当前帧应该显示多少点
            current_n = min(n_points, (frame + 1) * points_per_frame)
            
            # 更新各类别数据
            for cat in CATEGORY_COLORS.keys():
                mask = [categories[i] == cat for i in range(current_n)]
                cat_x = [x[i] for i in range(current_n) if mask[i]]
                cat_y = [y[i] for i in range(current_n) if mask[i]]
                
                if cat_x:
                    scatters[cat].set_offsets(np.column_stack([cat_x, cat_y]))
                else:
                    scatters[cat].set_offsets(np.empty((0, 2)))
            
            # 更新标题和计数器
            if current_n > 0:
                current_time = sorted_timestamps[current_n - 1]
                title.set_text(f'曼德拉效应传播 - {current_time.strftime("%Y-%m-%d %H:%M")}')
                
                # 统计各类别数量
                counts = {cat: sum(1 for c in categories[:current_n] if c == cat) 
                         for cat in CATEGORY_COLORS.keys()}
                counter.set_text(
                    f'总计: {current_n}\n'
                    f'曼德拉: {counts["MANDELA_EFFECT"]}\n'
                    f'反驳: {counts["REBUTTAL"]}'
                )
            
            return list(scatters.values()) + [title, counter]
        
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=total_frames, interval=1000/self.fps,
            blit=True
        )
        
        # 保存为 GIF
        output_path = self.output_dir / filename
        print(f"正在保存动画到 {output_path} ...")
        
        writer = PillowWriter(fps=self.fps)
        anim.save(str(output_path), writer=writer)
        plt.close()
        
        print(f"✓ 动画已保存到 {output_path}")
        return output_path


def main():
    """主流水线"""
    print("=" * 60)
    print("曼德拉效应评论向量化与可视化流水线")
    print("=" * 60)
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    print("\n[1/5] 加载评论数据...")
    loader = CommentLoader(LLM_ANALYZED_DIR)
    comments = loader.load_all_comments()
    texts, metadata = loader.prepare_for_embedding(comments)
    
    print(f"    - 有效评论数: {len(texts)}")
    print(f"    - 类别分布:")
    from collections import Counter
    cat_counts = Counter(m["category"] for m in metadata)
    for cat, count in cat_counts.items():
        print(f"      {cat}: {count}")
    
    # 2. 向量化
    print("\n[2/5] 文本向量化 (BGE-M3)...")
    embedder = EmbeddingEngine(EMBEDDING_MODEL)
    embeddings = embedder.encode(texts)
    
    # 保存 embeddings
    embeddings_path = OUTPUT_DIR / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"    ✓ Embeddings 已保存到 {embeddings_path}")
    
    # 保存 metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    # 转换 datetime 为字符串
    meta_serializable = []
    for m in metadata:
        m_copy = m.copy()
        m_copy["timestamp"] = m_copy["timestamp"].isoformat()
        meta_serializable.append(m_copy)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(meta_serializable, f, ensure_ascii=False, indent=2)
    print(f"    ✓ Metadata 已保存到 {metadata_path}")
    
    # 3. UMAP 降维
    print("\n[3/5] UMAP 降维 (Train-Project 模式)...")
    reducer = UMAPReducer()
    coords_2d = reducer.fit_transform(embeddings)
    
    # 保存 UMAP 模型
    umap_model_path = OUTPUT_DIR / "umap_model.pkl"
    reducer.save(str(umap_model_path))
    
    # 保存坐标
    coords_path = OUTPUT_DIR / "umap_coords.npy"
    np.save(coords_path, coords_2d)
    print(f"    ✓ UMAP 坐标已保存到 {coords_path}")
    
    # 4. 3D 可视化
    print("\n[4/5] 生成 3D 可视化...")
    visualizer = Visualizer3D(OUTPUT_DIR)
    visualizer.create_3d_plot(coords_2d, metadata)
    visualizer.create_static_3d_matplotlib(coords_2d, metadata)
    
    # 5. 动画生成
    print("\n[5/5] 生成传播动画...")
    animator = AnimationGenerator(OUTPUT_DIR, duration=20, fps=15)
    animator.create_animation(coords_2d, metadata)
    
    print("\n" + "=" * 60)
    print("✓ 流水线完成!")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("  生成文件:")
    print("    - embeddings.npy (向量)")
    print("    - metadata.json (元数据)")
    print("    - umap_model.pkl (UMAP模型)")
    print("    - umap_coords.npy (降维坐标)")
    print("    - mandela_3d_manifold.html (3D交互图)")
    print("    - mandela_3d_static.png (3D静态图)")
    print("    - mandela_spread.gif (传播动画)")
    print("=" * 60)


if __name__ == "__main__":
    main()
