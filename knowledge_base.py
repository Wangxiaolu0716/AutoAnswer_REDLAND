"""
本地题库模块：从 CSV / Excel 题库文件中按题干模糊匹配查找答案。

支持：
- CSV（如腾讯文档导出的 .csv，utf-8-sig 编码）
- Excel（.xlsx，需要 openpyxl：pip install openpyxl）
"""

import os
import re
import csv
import difflib
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None


class KnowledgeBase:
    """从 CSV / Excel 题库中按题干查找答案"""

    def __init__(self, path: str, question_col: str = "题目",
                 answer_col: str = "答案", detail_col: str = "解析",
                 min_ratio: float = 0.55):
        self.path = path
        self.question_col = question_col
        self.answer_col = answer_col
        self.detail_col = detail_col
        self.min_ratio = min_ratio
        self.rows: List[Dict[str, str]] = []
        self.loaded = False

    def load(self) -> bool:
        """加载题库文件，成功返回 True；文件缺失/类型不支持时返回 False（不抛异常）。"""
        path = self.path
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

        if not os.path.exists(path):
            logger.warning(f"未找到题库文件: {path}")
            return False

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":
                return self._load_csv(path)
            if ext in (".xlsx", ".xlsm"):
                return self._load_xlsx(path)
            logger.error(f"不支持的题库文件类型: {ext}（仅支持 .csv / .xlsx）")
            return False
        except Exception as e:  # pragma: no cover
            logger.error(f"读取题库文件失败: {e}")
            return False

    def _load_csv(self, path: str) -> bool:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = [c.strip() if c else "" for c in next(reader, [])]

            q_idx = self._find_col(header, self.question_col)
            a_idx = self._find_col(header, self.answer_col)
            d_idx = self._find_col(header, self.detail_col) if self.detail_col else None

            if q_idx is None or a_idx is None:
                logger.error(
                    f"CSV 表头找不到「{self.question_col}」或「{self.answer_col}」列，"
                    f"实际表头: {header}"
                )
                return False

            self.rows = []
            for row in reader:
                q = self._cell(row, q_idx)
                a = self._cell(row, a_idx)
                if not q or not a:
                    continue
                d = self._cell(row, d_idx) if d_idx is not None else ""
                self.rows.append({"question": q, "answer": a, "detail": d})

        self.loaded = True
        logger.info(f"CSV 题库加载完成: {len(self.rows)} 条（文件: {os.path.basename(path)}）")
        return True

    def _load_xlsx(self, path: str) -> bool:
        if openpyxl is None:
            logger.warning("未安装 openpyxl，无法读取 Excel 题库。请运行: pip install openpyxl")
            return False

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        header = [str(c).strip() if c is not None else "" for c in next(rows_iter, [])]

        q_idx = self._find_col(header, self.question_col)
        a_idx = self._find_col(header, self.answer_col)
        d_idx = self._find_col(header, self.detail_col) if self.detail_col else None

        if q_idx is None or a_idx is None:
            logger.error(
                f"Excel 表头找不到「{self.question_col}」或「{self.answer_col}」列，"
                f"实际表头: {header}"
            )
            wb.close()
            return False

        self.rows = []
        for row in rows_iter:
            if row is None:
                continue
            q = self._cell(row, q_idx)
            a = self._cell(row, a_idx)
            if not q or not a:
                continue
            d = self._cell(row, d_idx) if d_idx is not None else ""
            self.rows.append({"question": q, "answer": a, "detail": d})

        wb.close()
        self.loaded = True
        logger.info(f"Excel 题库加载完成: {len(self.rows)} 条（文件: {os.path.basename(path)}）")
        return True

    def search(self, question_text: str) -> Optional[Dict]:
        """按题干模糊匹配，返回最相似一条的 {answer, detail}；未达阈值返回 None。"""
        if not self.loaded or not self.rows or not question_text:
            return None

        q = self._normalize(question_text)
        if not q:
            return None

        best_row = None
        best_score = 0.0
        for row in self.rows:
            stored = self._normalize(row["question"])
            if not stored:
                continue
            # 任一方向包含 → 视为完全匹配；否则计算相似度
            if stored in q or q in stored:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, q, stored).ratio()
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None and best_score >= self.min_ratio:
            logger.debug(f"题库匹配: “{best_row['question'][:30]}…” 相似度 {best_score:.2f}")
            return {"answer": best_row["answer"], "detail": best_row["detail"]}
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        """去除空白与常见标点并转小写，提升对 OCR 噪声的容错。"""
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[，。、；：！？,.!?;:()（）\[\]【】《》〈〉·—\-~～]', '', text)
        return text.lower()

    @staticmethod
    def _find_col(header: List[str], target: str) -> Optional[int]:
        """在表头中定位列：先精确匹配，再模糊（互相包含）匹配。"""
        if not target:
            return None
        for i, h in enumerate(header):
            if h == target:
                return i
        for i, h in enumerate(header):
            if h and (target in h or h in target):
                return i
        return None

    @staticmethod
    def _cell(row, idx: Optional[int]) -> str:
        if idx is None or idx < 0 or idx >= len(row):
            return ""
        v = row[idx]
        return str(v).strip() if v is not None else ""
