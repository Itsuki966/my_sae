# GitHub Copilotへの指示書

## 1. 研究内容
このプロジェクトではSAEを利用したLLMの迎合性(Sycophancy)抑制に関する研究をおこなっています。
タスクを通して、LLMの迎合性に関わるSAE特徴を特定し、その特徴に介入することで、LLMの迎合的な振る舞いを抑制・緩和しようという試み。



## 2. 具体的な研究ステップ

### ステップ1: タスク実行とデータ取得
* **モデル:** LLM (Gemma-2-9b-it)
* **タスク:** 5つのテンプレートタイプ(base, I really like, I really dislike, I wrote, I didn't write)で、論理的な矛盾や統計的根拠に欠けた議論を含むトピックを渡し、それに対するコメントをさせる。
* **特徴取得:** タスク実行時に`SAE-lens`を利用して`sae_feature`も取得する。
* **実行環境:** Google Driveに実行ファイルを保存し、Google Colab上で実行する。

### ステップ2: 迎合性フラグの付与とデータ保存
* **評価:** LLMが生成したコメントを`gpt5-mini`に渡し、「base」テンプレートの出力と比較させる。
* **フラグ付与:** 他テンプレートが「base」より迎合しているか判断させ、`sycophancy_flag=1` (迎合) / `0` (非迎合) と `reason` (判断理由) を生成させる。
* **保存:** ステップ1の結果と合わせ、以下のJSON構造で保存する。

* **Feedback実験 出力フォーマット構造**
    *(注: これは実際のデータではなく、ファイルの構造を示すためのものです)*
    ```json
    {
      "metadata": {
        "model_name": "[string]",
        "sae_release": "[string]",
        "sae_id": "[string]",
        "num_questions": "[number]",
        "analysis_position": "[string]",
        "target_layer": "[string]",
        "timestamp": "[datetime_iso]",
        "config": {
          "sample_size": "[number]",
          "max_new_tokens": "[number]",
          "temperature": "[float]",
          "top_k_features": "[number]"
        }
      },
      "results": [
        {
          "question_id": "[number]",
          "dataset": "[string]",
          "base_text": "[string]",
          "variations": [
            {
              "template_type": "[string]",
              "prompt": "[string]",
              "response": "[string]",
              "sae_activations": {
                "prompt_last_token": {
                  "[feature_id (string)]": "[activation_value (float)]",
                  "...": "..."
                }
              },
              "top_k_features": [
                [
                  "[feature_id (number)]",
                  "[activation_value (float)]"
                ],
                [
                  "...",
                  "..."
                ]
              ],
              "metadata": {
                "generation_time_ms": "[float]",
                "response_length": "[number]",
                "timestamp": "[datetime_iso]",
                "gpu_memory_mb": "[float]"
              },
              "sycophancy_flag": "[0 or 1]",
              "reason": "[string]"
            }
          ]
        }
      ]
    }
    ```

### ステップ3: 特徴の特定
実験結果から、LLMの迎合に深く関わっていそうなSAE特徴を特定する。

### ステップ4: 介入と出力操作
特定したSAE特徴を利用してLLMの中間層に介入し、出力の操作（迎合性の抑制）を行う。

## 3. 現在の進捗
現在、ステップ２まで完了している。