# -*- coding: utf-8 -*-
"""
Phase 0 生死探针：中文语音与文本跨模态余弦相似度对齐验证
严格基于 Google EmbeddingGemma 2 官方文档推荐的 SentenceTransformer 接口进行评测
"""

import os
import sys
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = str(PROJECT_ROOT / "models" / "google" / "embeddinggemma-2")
AUDIO_PATH = str(PROJECT_ROOT / "example_audio.wav")
SLICE_PATH = str(PROJECT_ROOT / "example_slice.wav")

# 候选文本池 (严格梯度测试用例)
CANDIDATES = [
    # 1. 绝对正样本 (Ground Truth)
    ("【正样本】精准匹配原句", "今天咱们来聊一个互联网亚文化里，绕不开的传奇贴吧，神奈川冲浪里吧。"),
    # 2. 近音/同音/形近干扰项 (测试声学消歧)
    ("【近音干扰】同音与错别字", "今天咱们来聊一个互联网下文化里，找不开的神奇贴吧，神乃川冲浪里吧。"),
    # 3. 语义高度相关干扰项 (来自音频后半段的描述)
    ("【同文相关】同篇后文语境", "神友们最擅长的就是用一种极其荒诞自嘲的黑色幽默去解构生活。"),
    # 4. 关键人名/亚文化概念干扰项
    ("【概念干扰】抽象系与人物", "也就是李老八孙笑川那个圈子衍生出来的一个核心聚集地。"),
    # 5. 完全无关负样本 A
    ("【无关样本】日常天气闲聊", "今天北京的天气非常晴朗，出门记得做好防晒和补水。"),
    # 6. 完全无关负样本 B
    ("【无关样本】技术论文片段", "基于深度学习的多模态大模型在语音识别领域的应用与工程实践。"),
]

def prepare_audio_slice(src_path: str, dst_path: str, target_sr: int = 16000, slice_seconds: float = 8.5):
    """读取音频、重采样至 16kHz 并截取前 slice_seconds 秒生成切片文件"""
    wav_np, sr = sf.read(src_path)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=-1)
    
    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(wav_np) * target_sr / sr)
        wav_np = scipy.signal.resample(wav_np, num_samples)
        sr = target_sr
        
    max_samples = int(sr * slice_seconds)
    slice_data = wav_np[:max_samples].astype(np.float32)
    sf.write(dst_path, slice_data, target_sr)
    return dst_path, len(slice_data) / target_sr

def main():
    print("=" * 75)
    print("🔬 EmbeddingGemma 2 官方标准跨模态相似度探针 (Phase 0 验证)")
    print("=" * 75)
    print(f"[*] 模型路径: {MODEL_PATH}")
    print(f"[*] 音频样本: {AUDIO_PATH}")
    
    if not os.path.exists(AUDIO_PATH):
        print(f"[错误] 未找到音频样本文件: {AUDIO_PATH}")
        sys.exit(1)
        
    print("\n[1/4] 正在切取前 8.5 秒黄金语音切片 (16kHz 单声道)...")
    slice_file, dur = prepare_audio_slice(AUDIO_PATH, SLICE_PATH, target_sr=16000, slice_seconds=8.5)
    print(f"      音频切片已生成: {slice_file} (时长: {dur:.2f}s)")

    print("\n[2/4] 正在按官方规范加载 SentenceTransformer 模型 (bfloat16)...")
    model = SentenceTransformer(MODEL_PATH, model_kwargs={"torch_dtype": torch.bfloat16})
    print("      官方模型加载成功！")

    print("\n[3/4] 正在提取 768 维音频与候选文本向量...")
    # 官方推荐多模态输入格式
    audio_emb = model.encode({"text": "<|audio|>", "audio": [slice_file]}, normalize_embeddings=True)
    
    text_labels = [c[0] for c in CANDIDATES]
    text_contents = [c[1] for c in CANDIDATES]
    text_embs = model.encode(text_contents, prompt_name="SentenceSimilarity", normalize_embeddings=True)

    print("\n[4/4] 正在计算余弦相似度矩阵并生成排行榜...")
    sim_matrix = model.similarity(audio_emb, text_embs)[0].cpu().numpy()

    results = []
    for label, content, sim in zip(text_labels, text_contents, sim_matrix):
        results.append((float(sim), label, content))
    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 75)
    print(f"{'排名':^4} | {'相似度':^8} | {'类别':<18} | {'候选文本内容'}")
    print("-" * 75)
    for rank, (sim, label, content) in enumerate(results, 1):
        truncated = content if len(content) <= 35 else content[:32] + "..."
        print(f" #{rank:<3} | {sim:8.4f} | {label:<18} | {truncated}")
    print("=" * 75)

    # 科学裁决判定
    top1_sim, top1_label, top1_text = results[0]
    is_hit = "正样本" in top1_label
    margin = results[0][0] - results[1][0] if len(results) > 1 else 0.0

    print("\n🎯 【Phase 0 科学判定结论】:")
    if is_hit and margin >= 0.05:
        print(f"🟢 [大绿灯 - 绝对通过] 正确文本以显著分差 ({margin:+.4f}) 稳居 Top-1！")
        print(f"   命中内容: \"{top1_text}\"")
        print("   证明：EmbeddingGemma 2 的音频 Conformer 对中文语音具备强劲的语义区分度与几何对齐，训练 Reader 路线完全成立！")
    elif is_hit:
        print(f"🟡 [黄灯 - 条件通过] 正样本位列 Top-1，但与干扰项分差较小 ({margin:+.4f})。")
        print("   证明：模型能识别中文语音语义，但同音字消歧较为敏感，后续必须挂载轻量 LM 重打分。")
    else:
        print(f"🔴 [红灯 - 预警] 正样本未命中 Top-1 (当前 Top-1 为: {top1_label})。")
        print("   证明：冻结编码器对中文语音缺乏足够先验，需重新评估直接使用 Conformer 做 CTC 的可行性。")
    print("=" * 75)

if __name__ == "__main__":
    main()
