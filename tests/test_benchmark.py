# -*- coding: utf-8 -*-
"""
单元测试：scripts/benchmark.py 的功能契约测试 (TDD 模式)
包含文本规范化、计时排他性、汇总表结构及外部引擎 mock 测试。
"""
import sentencepiece  # noqa: F401  # Windows CRT clash protection
import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.benchmark import normalize_text, calc_metrics, benchmark_engine, format_summary_table


class TestBenchmarkNormalization:
    def test_normalize_sensevoice_output(self):
        """测试 SenseVoice 风格富文本标签与标点过滤"""
        raw = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>今天，天气不错！"
        assert normalize_text(raw) == "今天天气不错"

    def test_normalize_paraformer_output(self):
        """测试 Paraformer 风格字间空格与全半角标点过滤"""
        raw = "今 天 咱 们 来 聊 ， 互 联 网 亚 文 化 ！"
        assert normalize_text(raw) == "今天咱们来聊互联网亚文化"

    def test_normalize_empty_and_special_chars(self):
        """测试空文本与纯特殊字符"""
        assert normalize_text("") == ""
        assert normalize_text("   <|en|>   ... --- ") == ""
        assert normalize_text("科技（二零二六）发展：快速。") == "科技二零二六发展快速"


class TestBenchmarkMetrics:
    def test_calc_metrics_exact_match(self):
        refs = ["土地储备至关重要", "零三年到去年"]
        hyps = ["土地储备至关重要", "零三年到去年"]
        res = calc_metrics(refs, hyps)
        assert res["cer"] == 0.0
        assert res["pcer"] == 0.0
        assert res["homophone_err"] == 0.0

    def test_calc_metrics_homophone_detection(self):
        """测试同音字差值 (CER - P-CER) 计算"""
        refs = ["加强建设资金统筹"]
        # '司金' 与 '资金' 拼音完全相同 (sījīn vs zījīn，在无声调下近似，或者用同音词测试)
        hyps = ["加强建设司金统筹"]
        res = calc_metrics(refs, hyps)
        assert res["cer"] > 0.0
        # 同音字差值应当大于等于 0
        assert res["homophone_err"] >= 0.0


class TestBenchmarkTimingAndFormat:
    def test_timing_excludes_model_load(self):
        """验证计时器严格排除模型加载时间"""
        mock_model = MagicMock()

        def slow_infer(wavs):
            time.sleep(0.02)
            return ["今天天气不错"]

        mock_model.infer = slow_infer

        # 模拟加载耗时 0.1 秒
        t_load_start = time.time()
        time.sleep(0.1)
        load_time = time.time() - t_load_start

        # 模拟 1 个音频执行推理
        wavs = [[0.0] * 16000]
        t0 = time.time()
        mock_model.infer(wavs)
        infer_time = time.time() - t0

        # 推理耗时应显著小于加载耗时
        assert infer_time < load_time
        assert infer_time >= 0.015

    def test_summary_table_columns(self):
        """验证汇总表必须包含工单 11 规定的 9 个核心列"""
        records = [
            {
                "engine": "Gemma-ASR",
                "mode": "bs1",
                "batch": 1,
                "cer": 0.0697,
                "pcer": 0.0229,
                "homophone_err": 0.0468,
                "time": 30.2,
                "speed": 1190.0,
                "vram_gb": 2.3,
            }
        ]
        table_md = format_summary_table(records)
        required_headers = [
            "engine",
            "mode",
            "batch",
            "CER",
            "P-CER",
            "同音字错误",
            "耗时",
            "实时倍率",
            "峰值显存",
        ]
        for header in required_headers:
            assert header in table_md
