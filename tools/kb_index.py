"""知识库检索内核：纯标准库实现的 BM25 索引。

为什么不用向量库 / embedding：
- 项目铁律是零第三方依赖（不能用 numpy / faiss / sentence-transformers）；
- 中文短文本（评价、店名、招牌菜）用「单字 + 相邻二字 bigram + BM25 + 同义词扩展」
  召回已经够用，不需要 GPU、不需要联网调 embedding 接口；
- 语料规模是几百到几千条，内存构建毫秒级，不需要持久化索引文件。

分词策略（没有 jieba 的替代方案）：
- 中文连续片段切「单字 + 二字」：单字保证「粥」「辣」这类短查询能命中，
  二字保证「排队」「性价比」这类词有区分度；
- 英文/数字按字符类切分；
- IDF 自动压掉「的、了、很」这类高频噪声，另配一份停用词表兜底。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

# 标点、空白、全角符号统一视为分隔符
_SEPARATOR = re.compile(
    r"[\s\u3000,，。.；;、!！?？:：\"'“”‘’()（）\[\]【】<>《》…—\-_/\\|+*&#@$%~^`{}=~]+"
)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RUN = re.compile(r"[a-z0-9]+")

# 停用词：频率极高但没有区分度，不建索引
STOPWORDS = frozenset(
    "的 了 是 我 你 他 她 也 就 都 和 在 有 很 不 这 那 个 还 又 被 把 给 让 对 从 到 去 说 吃 家 店 一 二 三 两 上 下".split()
)

# 同义词/上位词扩展：查询命中 key 时连带检索 alts（权重减半）。
# 只收录「学生真实说法 → 知识库里可能出现的词」，避免过度扩张带来噪声。
SYNONYMS: Dict[str, List[str]] = {
    "辣": ["麻辣", "香辣", "重口", "辣味", "川菜", "湘菜"],
    "清淡": ["不辣", "少油", "养生", "粤菜", "汤粉"],
    "便宜": ["实惠", "性价比", "划算", "人均低"],
    "贵": ["偏贵", "不划算", "性价比低"],
    "难吃": ["一般", "踩雷", "不推荐", "失望", "避雷"],
    "好吃": ["香", "推荐", "惊艳", "回味"],
    "排队": ["等位", "等候", "人多", "高峰", "饭点"],
    "卫生": ["干净", "不干净", "环境"],
    "分量": ["份量", "管饱", "量大", "量少"],
    "早餐": ["早饭", "早茶", "包子", "粥", "豆浆"],
    "午餐": ["中饭", "午饭", "快餐"],
    "晚餐": ["晚饭", "夜宵"],
    "夜宵": ["宵夜", "烧烤", "深夜", "炸串"],
    "聚餐": ["聚会", "多人", "包间", "请客", "大桌"],
    "外卖": ["配送", "送餐", "打包"],
    "食堂": ["饭堂", "canteen", "窗口"],
    "面": ["粉面", "拉面", "米粉", "汤面"],
    "饭": ["盖饭", "煲仔饭", "炒饭", "快餐"],
    "甜": ["糖水", "甜品", "绿豆沙", "双皮奶"],
}


def normalize(text: Any) -> str:
    """小写 + 标点归一。"""
    return _SEPARATOR.sub(" ", str(text or "").lower()).strip()


def tokenize(text: Any) -> List[str]:
    """中文单字 + 二字 bigram + 英文数字。"""
    text = normalize(text)
    tokens: List[str] = []
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        if len(run) > 1:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(_LATIN_RUN.findall(text))
    return [t for t in tokens if t not in STOPWORDS]


def query_weights(query: Any) -> Counter:
    """查询词权重：原词 1.0，同义词扩展 0.5。"""
    weights: Counter = Counter()
    for token in tokenize(query):
        weights[token] += 1.0
    raw = str(query or "")
    for key, alternatives in SYNONYMS.items():
        if key not in raw:
            continue
        for alt in alternatives:
            for token in tokenize(alt):
                weights[token] += 0.5
    return weights


class BM25Index:
    """倒排索引 + Okapi BM25 打分。"""

    def __init__(self, docs: Iterable[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs: List[Dict[str, Any]] = list(docs)
        self.k1 = k1
        self.b = b
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_len: List[int] = []
        for doc_id, doc in enumerate(self.docs):
            tokens = tokenize(doc.get("text", ""))
            self.doc_len.append(len(tokens))
            for token in tokens:
                self.postings[token][doc_id] = self.postings[token].get(doc_id, 0) + 1
        self.length = len(self.docs)
        self.avgdl = (sum(self.doc_len) / self.length) if self.length else 0.0

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, {}))
        if not self.length:
            return 0.0
        return math.log(1.0 + (self.length - df + 0.5) / (df + 0.5))

    def search(self, query: Any, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """返回 [(文档, 分数)]，按分数降序；零分文档不返回。"""
        if not self.length:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for term, weight in query_weights(query).items():
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for doc_id, tf in posting.items():
                dl = self.doc_len[doc_id] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_id] += weight * idf * (tf * (self.k1 + 1) / denom)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self.docs[doc_id], score) for doc_id, score in ranked[:top_k]]


def normalize_scores(hits: List[Tuple[Any, float]]) -> List[float]:
    """把 BM25 分数归一化到 0~1（除以本批最高分），便于与评分加权融合。"""
    if not hits:
        return []
    top = max(score for _, score in hits) or 1.0
    return [score / top for _, score in hits]
