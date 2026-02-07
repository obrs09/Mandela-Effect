# B站评论爬虫 V5 - 保存完整父子关系
# 支持构建真正的评论树结构

import requests
import json
import time
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BILIBILI_COOKIE


class BilibiliCommentScraperV5:
    def __init__(self, cookie=None):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.bilibili.com',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Origin': 'https://www.bilibili.com',
        }
        if cookie:
            self.headers['Cookie'] = cookie
        self.session.headers.update(self.headers)
    
    def bvid_to_aid(self, bvid):
        url = f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}'
        try:
            resp = self.session.get(url)
            data = resp.json()
            if data['code'] == 0:
                aid = data['data']['aid']
                title = data['data']['title']
                print(f"视频标题: {title}")
                print(f"AV号: av{aid}")
                return aid, title
            return None, None
        except Exception as e:
            print(f"请求失败: {e}")
            return None, None
    
    def get_comments_cursor(self, oid, cursor=None, mode=3):
        url = 'https://api.bilibili.com/x/v2/reply/main'
        params = {'type': 1, 'oid': oid, 'mode': mode, 'ps': 20}
        if cursor:
            params['next'] = cursor
        try:
            resp = self.session.get(url, params=params)
            return resp.json()
        except Exception as e:
            print(f"获取评论失败: {e}")
            return None
    
    def get_replies_cursor(self, oid, root_id, page=1):
        url = 'https://api.bilibili.com/x/v2/reply/reply'
        params = {'type': 1, 'oid': oid, 'root': root_id, 'ps': 20, 'pn': page}
        try:
            resp = self.session.get(url, params=params)
            return resp.json()
        except Exception as e:
            print(f"获取回复失败: {e}")
            return None
    
    def format_comment(self, comment, is_root=False):
        """格式化评论，保留parent信息"""
        formatted = {
            'id': str(comment['rpid']),
            'text': comment['content']['message'],
            'author': comment['member']['uname'],
            'author_mid': str(comment['member']['mid']),
            'timestamp': datetime.fromtimestamp(comment['ctime']).isoformat() + 'Z',
            'like_count': comment['like'],
        }
        
        if is_root:
            formatted['reply_count'] = comment.get('rcount', 0)
        else:
            # 保存parent_id用于构建树
            parent_id = comment.get('parent', 0)
            root_id = comment.get('root', 0)
            # 如果parent == root，说明直接回复楼主
            # 如果parent != root，说明回复的是其他楼层
            formatted['parent_id'] = str(parent_id) if parent_id else str(root_id)
            formatted['root_id'] = str(root_id)
        
        return formatted
    
    def build_comment_data(self, comment, aid, delay):
        """构建评论数据（包含所有回复和parent关系）"""
        root = self.format_comment(comment, is_root=True)
        root['replies'] = []
        
        reply_count = comment.get('rcount', 0)
        if reply_count == 0:
            return root
        
        fetched_ids = set()
        
        # 预加载的回复
        preload_replies = comment.get('replies') or []
        for reply in preload_replies:
            formatted = self.format_comment(reply, is_root=False)
            root['replies'].append(formatted)
            fetched_ids.add(reply['rpid'])
        
        # 分页获取剩余回复
        if reply_count > len(preload_replies):
            reply_page = 1
            while len(fetched_ids) < reply_count:
                time.sleep(delay)
                reply_data = self.get_replies_cursor(aid, comment['rpid'], reply_page)
                
                if not reply_data or reply_data['code'] != 0:
                    break
                
                page_replies = reply_data.get('data', {}).get('replies') or []
                if not page_replies:
                    break
                
                for reply in page_replies:
                    if reply['rpid'] not in fetched_ids:
                        formatted = self.format_comment(reply, is_root=False)
                        root['replies'].append(formatted)
                        fetched_ids.add(reply['rpid'])
                
                reply_page += 1
                if reply_page > 100:
                    break
        
        return root
    
    def save_comment_data(self, data, output_dir, is_top=False):
        filename = f"{'top_' if is_top else ''}{data['id']}.json"
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def scrape_all_comments(self, bvid, output_dir, delay=0.5):
        aid, title = self.bvid_to_aid(bvid)
        if not aid:
            return None
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        total_comments = 0
        cursor = None
        page = 0
        processed_ids = set()
        
        print("\n开始爬取评论...")
        
        while True:
            page += 1
            print(f"正在获取第 {page} 批主评论 (已保存 {saved_count} 个, 共 {total_comments} 条)...")
            
            data = self.get_comments_cursor(aid, cursor)
            
            if not data:
                break
            
            if data['code'] != 0:
                print(f"API错误 (code={data['code']}): {data.get('message', '未知错误')}")
                break
            
            if page == 1:
                api_total = data['data'].get('cursor', {}).get('all_count', 0)
                print(f"API报告评论总数: {api_total}")
                
                # 置顶评论
                top_replies = data['data'].get('top_replies') or []
                if top_replies:
                    print(f"发现 {len(top_replies)} 条置顶评论")
                    for top_comment in top_replies:
                        if top_comment['rpid'] not in processed_ids:
                            print(f"  ↳ 处理置顶评论 (回复数: {top_comment.get('rcount', 0)})...")
                            comment_data = self.build_comment_data(top_comment, aid, delay)
                            comment_data['is_top'] = True
                            self.save_comment_data(comment_data, output_dir, is_top=True)
                            processed_ids.add(top_comment['rpid'])
                            saved_count += 1
                            total_comments += 1 + len(comment_data['replies'])
            
            cursor_info = data['data'].get('cursor', {})
            is_end = cursor_info.get('is_end', True)
            next_cursor = cursor_info.get('next', 0)
            
            replies = data['data'].get('replies') or []
            if not replies:
                print("没有更多评论了")
                break
            
            for comment in replies:
                if comment['rpid'] in processed_ids:
                    continue
                
                reply_count = comment.get('rcount', 0)
                if reply_count > 3:
                    print(f"  ↳ 处理评论 (回复数: {reply_count})...")
                
                comment_data = self.build_comment_data(comment, aid, delay)
                comment_data['is_top'] = False
                self.save_comment_data(comment_data, output_dir, is_top=False)
                processed_ids.add(comment['rpid'])
                saved_count += 1
                total_comments += 1 + len(comment_data['replies'])
            
            if is_end:
                print("已到达评论末尾")
                break
            
            cursor = next_cursor
            time.sleep(delay)
            
            if page > 500:
                break
        
        # 元信息
        meta = {
            'bvid': bvid,
            'aid': aid,
            'title': title,
            'scrape_time': datetime.now().isoformat() + 'Z',
            'total_trees': saved_count,
            'total_comments': total_comments,
        }
        with open(output_dir / '_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f"\n爬取完成! 共 {saved_count} 个评论树, {total_comments} 条评论")
        return {'saved_count': saved_count, 'total_comments': total_comments}


def main():
    bvid = 'BV1cpqpB4Eda'
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'data' / 'raw' / 'comments_v2'
    
    print("=" * 60)
    print("B站评论爬虫 V5 - 保存完整父子关系")
    print("=" * 60)
    
    scraper = BilibiliCommentScraperV5(cookie=BILIBILI_COOKIE)
    scraper.scrape_all_comments(bvid, output_dir, delay=0.5)


if __name__ == '__main__':
    main()
