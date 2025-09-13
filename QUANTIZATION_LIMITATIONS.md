# 量子化機能の制限と対応方針

## 根本的な問題

### 1. transformer_lens と bitsandbytes の非互換性

**問題**: transformer_lens は bitsandbytes の量子化と根本的に互換性がない

**技術的詳細**:
- bitsandbytes は HuggingFace transformers 専用に設計
- transformer_lens は独自の重み変換ロジックを使用
- 量子化により重みテンソルの形状とメモリレイアウトが変化
- transformer_lens の einops 操作が量子化テンソルを認識できない

**なぜ修正が困難か**:
1. transformer_lens のコア重み変換ロジックの変更が必要
2. SAE-lens が transformer_lens に依存している
3. 根本的なアーキテクチャレベルの非互換性

### 2. HookedTransformer の generate メソッドの制限

**問題**: HookedTransformer.generate() は標準的な transformers.generate() と完全互換ではない

**制限事項**:
- `pad_token_id` パラメータ未サポート
- `return_dict_in_generate` パラメータ未サポート  
- 一部の生成設定オプション未対応

## 現実的な対応方針

### 短期的解決策（実装済み）
1. 量子化失敗時の標準読み込みフォールバック
2. generate メソッドの代替実装
3. エラーハンドリングの強化

### 長期的解決策（推奨）
1. 量子化なしでのメモリ効率化
2. より小さなモデルの使用
3. accelerate ライブラリによるモデル分散

### 非推奨（技術的に困難）
1. transformer_lens の根本的な修正
2. SAE-lens の置き換え
3. 独自の量子化ロジックの実装

## 推奨される実用的なアプローチ

量子化にこだわらず、以下の方法でメモリ効率化を図る：

1. **accelerate による分散読み込み**
2. **float16 精度の使用**
3. **CPU-GPU ハイブリッド実行**
4. **より小さなモデルの採用**
