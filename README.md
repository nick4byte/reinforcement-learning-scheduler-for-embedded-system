# 基於異質運算平台的智慧任務排程系統設計

## 📌 專案簡介

本專案旨在設計並實作一套**基於異質運算平台（Heterogeneous Computing Platform）的智慧任務排程系統**，透過分析任務特性與系統資源狀態，將任務動態分配至最適合的運算單元（如 CPU、GPU 或其他加速器），以提升整體效能、降低延遲並改善資源使用率。

本專案以**嵌入式系統與作業系統設計概念**為核心，結合排程演算法與系統層實作，適合作為作業系統、嵌入式系統或異質運算相關研究與實務專題。

---

## 🎯 專案目標

* 建立可運行於嵌入式／異質運算平台的任務排程架構
* 分析不同運算單元（CPU / GPU / 加速器）的特性
* 設計智慧化任務分派策略（Rule-based 或 Heuristic）
* 比較不同排程策略對效能與資源利用率的影響

---

## 🧠 系統架構概述

本系統以**強化學習為核心的智慧排程器**為主體，將異質運算平台的任務分派問題建模為 Markov Decision Process（MDP），並透過 Gymnasium 環境與 PPO 演算法進行學習。

系統主要由以下模組組成：

1. **任務排程環境（`taskscheduler`）**

   * 繼承自 `gymnasium.Env`
   * 定義任務數量、CPU/GPU/NPU 資源數
   * Observation：系統資源與任務狀態
   * Action：任務分派至指定運算單元
   * Reward：根據完成時間與資源利用率設計

2. **運算裝置模型（Device Model）**

   * 以速度係數模擬不同裝置效能（CPU < GPU < NPU）

3. **強化學習模型（PPO Agent）**

   * 使用 PyTorch 實作 Policy Network 與 Value Network
   * 透過互動式訓練學習最佳排程策略

4. **訓練與視覺化模組**

   * 支援 TensorBoard 紀錄訓練過程
   * 使用 Matplotlib 繪製 reward 與效能趨勢

---

## 🛠️ 使用技術與環境

* **程式語言**：Python 3
* **核心框架**：

  * Gymnasium（自訂環境）
  * PyTorch（強化學習模型訓練）
* **強化學習方法**：

  * Policy Gradient / PPO（Proximal Policy Optimization）
* **運算平台抽象**：CPU / GPU / NPU（以速度係數模擬）
* **專案性質**：

  * 課程專題／個人 Side Project
  * 聚焦概念驗證（Concept Proof）與系統設計

---

## 🚀 專案特色

* 將作業系統排程概念實際應用於異質運算平台
* 聚焦嵌入式系統可行性，而非純理論模擬
* 架構模組化，方便後續擴充 AI 或 Machine Learning 排程策略

---

## 📂 專案目錄結構（範例）

```text
.
├── src/                # 核心程式碼
│   ├── scheduler/      # 排程器實作
│   ├── task/           # 任務定義
│   └── monitor/        # 資源監控
├── docs/               # 專案文件
├── experiments/        # 測試與效能比較
├── README.md
└── LICENSE
```

---

## 🧪 訓練環境說明

* **作業系統**：macOS / Linux（開發與測試）
* **Python 套件**：

  * gymnasium
  * torch
  * numpy
  * matplotlib
* **訓練方式**：

  * 單機訓練（CPU 為主）
  * 透過模擬環境驗證排程策略有效性

---

## 📊 未來擴充方向

* 將模擬式裝置模型替換為真實異質平台（如 Jetson / FPGA / NPU）
* 導入即時系統（RTOS）與 deadline-aware 排程
* 比較 Heuristic / Rule-based 與 RL 排程效能差異
* 支援多任務並行與動態任務到達模型

