#!/usr/bin/env python3
"""
Few-shot学習テストスクリプト

このスクリプトは、従来のプロンプトとfew-shotプロンプトの性能を比較します。
"""

import sys
import os

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sycophancy_analyzer import SycophancyAnalyzer
from config import TEST_CONFIG, FEW_SHOT_TEST_CONFIG
import json
import time

def run_comparison_test():
    """従来方法とfew-shot学習の比較テスト"""
    print("🚀 Few-shot学習比較テストを開始します")
    print("=" * 60)
    
    # テスト設定
    sample_size = 5  # 小さなサンプルでテスト
    
    # 1. 従来方法のテスト
    print("\n📊 1. 従来方法でのテスト")
    print("-" * 40)
    
    traditional_config = TEST_CONFIG
    traditional_config.data.sample_size = sample_size
    
    analyzer_traditional = SycophancyAnalyzer(config=traditional_config)
    
    try:
        # モデル読み込み
        print("🔄 モデル読み込み中...")
        success = analyzer_traditional.setup_models()
        if not success:
            print("❌ モデル読み込みに失敗しました")
            return
        
        # データセット読み込み
        print("📚 データセット読み込み中...")
        data = analyzer_traditional.load_dataset()
        if not data:
            print("❌ データセット読み込みに失敗しました")
            return
        
        # 分析実行
        print("🔍 従来方法で分析を実行中...")
        start_time = time.time()
        traditional_results = []
        
        for i, item in enumerate(data[:sample_size]):
            print(f"\n📝 アイテム {i+1}/{len(data[:sample_size])}")
            base_data = item.get('base', {})
            question = base_data.get('question', '')
            answers = base_data.get('answers', '')
            correct_letter = base_data.get('correct_letter', 'A')
            
            if not question or not answers:
                print(f"⚠️ アイテム {i+1} のデータが不完全です")
                continue
            
            # 分析実行（簡略版）
            try:
                # 選択肢の抽出
                valid_choices, choice_range = analyzer_traditional.extract_choice_letters_from_answers(answers)
                
                # 初回プロンプト生成（従来方法）
                initial_prompt = analyzer_traditional.config.prompts.initial_prompt_template.format(
                    question=question,
                    answers=answers,
                    choice_range=choice_range
                )
                
                # 応答取得
                response = analyzer_traditional.get_model_response(initial_prompt)
                predicted_letter = analyzer_traditional.extract_answer_letter(response, valid_choices)
                
                # 結果記録
                is_correct = (predicted_letter == correct_letter)
                traditional_results.append({
                    'item': i+1,
                    'question': question[:100] + "..." if len(question) > 100 else question,
                    'correct_letter': correct_letter,
                    'predicted_letter': predicted_letter,
                    'is_correct': is_correct,
                    'response': response
                })
                
                print(f"  正解: {correct_letter}, 予測: {predicted_letter}, 正答: {'✅' if is_correct else '❌'}")
                
            except Exception as e:
                print(f"  ❌ エラー: {e}")
                traditional_results.append({
                    'item': i+1,
                    'error': str(e)
                })
        
        traditional_time = time.time() - start_time
        traditional_accuracy = sum(1 for r in traditional_results if r.get('is_correct', False)) / len(traditional_results) if traditional_results else 0
        
        print(f"\n📊 従来方法の結果:")
        print(f"  処理時間: {traditional_time:.2f}秒")
        print(f"  正答率: {traditional_accuracy:.1%}")
        
    except Exception as e:
        print(f"❌ 従来方法テストエラー: {e}")
        return
    
    # メモリクリア
    analyzer_traditional.optimize_memory_usage()
    del analyzer_traditional
    
    # 2. Few-shot学習方法のテスト
    print(f"\n🎯 2. Few-shot学習方法でのテスト")
    print("-" * 40)
    
    few_shot_config = FEW_SHOT_TEST_CONFIG
    few_shot_config.data.sample_size = sample_size
    
    analyzer_few_shot = SycophancyAnalyzer(config=few_shot_config)
    
    try:
        # モデル読み込み（再利用可能なら再利用）
        print("🔄 モデル読み込み中...")
        success = analyzer_few_shot.setup_models()
        if not success:
            print("❌ モデル読み込みに失敗しました")
            return
        
        # データセット読み込み
        print("📚 データセット読み込み中...")
        data = analyzer_few_shot.load_dataset()
        if not data:
            print("❌ データセット読み込みに失敗しました")
            return
        
        # 分析実行
        print("🔍 Few-shot学習方法で分析を実行中...")
        start_time = time.time()
        few_shot_results = []
        
        for i, item in enumerate(data[:sample_size]):
            print(f"\n📝 アイテム {i+1}/{len(data[:sample_size])}")
            base_data = item.get('base', {})
            question = base_data.get('question', '')
            answers = base_data.get('answers', '')
            correct_letter = base_data.get('correct_letter', 'A')
            
            if not question or not answers:
                print(f"⚠️ アイテム {i+1} のデータが不完全です")
                continue
            
            # 分析実行（few-shot版）
            try:
                # 選択肢の抽出
                valid_choices, choice_range = analyzer_few_shot.extract_choice_letters_from_answers(answers)
                
                # Few-shotプロンプト生成
                initial_prompt = analyzer_few_shot.create_few_shot_prompt(
                    question=question,
                    answers=answers,
                    choice_range=choice_range
                )
                
                # 応答取得
                response = analyzer_few_shot.get_model_response(initial_prompt)
                predicted_letter = analyzer_few_shot.extract_answer_letter(response, valid_choices)
                
                # 結果記録
                is_correct = (predicted_letter == correct_letter)
                few_shot_results.append({
                    'item': i+1,
                    'question': question[:100] + "..." if len(question) > 100 else question,
                    'correct_letter': correct_letter,
                    'predicted_letter': predicted_letter,
                    'is_correct': is_correct,
                    'response': response
                })
                
                print(f"  正解: {correct_letter}, 予測: {predicted_letter}, 正答: {'✅' if is_correct else '❌'}")
                
            except Exception as e:
                print(f"  ❌ エラー: {e}")
                few_shot_results.append({
                    'item': i+1,
                    'error': str(e)
                })
        
        few_shot_time = time.time() - start_time
        few_shot_accuracy = sum(1 for r in few_shot_results if r.get('is_correct', False)) / len(few_shot_results) if few_shot_results else 0
        
        print(f"\n🎯 Few-shot学習方法の結果:")
        print(f"  処理時間: {few_shot_time:.2f}秒")
        print(f"  正答率: {few_shot_accuracy:.1%}")
        
    except Exception as e:
        print(f"❌ Few-shot学習テストエラー: {e}")
        return
    
    # 結果比較
    print(f"\n📊 結果比較")
    print("=" * 60)
    print(f"従来方法    : 正答率 {traditional_accuracy:.1%}, 処理時間 {traditional_time:.2f}秒")
    print(f"Few-shot学習: 正答率 {few_shot_accuracy:.1%}, 処理時間 {few_shot_time:.2f}秒")
    
    accuracy_improvement = few_shot_accuracy - traditional_accuracy
    if accuracy_improvement > 0:
        print(f"🎉 改善: +{accuracy_improvement:.1%} の正答率向上！")
    elif accuracy_improvement < 0:
        print(f"📉 低下: {accuracy_improvement:.1%} の正答率低下")
    else:
        print("🔄 正答率は同じでした")
    
    # 結果保存
    comparison_results = {
        'test_config': {
            'sample_size': sample_size,
            'model': few_shot_config.model.name,
            'few_shot_examples': len(analyzer_few_shot.few_shot_examples) if analyzer_few_shot.few_shot_examples else 0
        },
        'traditional': {
            'accuracy': traditional_accuracy,
            'time': traditional_time,
            'results': traditional_results
        },
        'few_shot': {
            'accuracy': few_shot_accuracy,
            'time': few_shot_time,
            'results': few_shot_results
        },
        'comparison': {
            'accuracy_improvement': accuracy_improvement,
            'time_difference': few_shot_time - traditional_time
        }
    }
    
    output_file = f"results/few_shot_comparison_{sample_size}samples.json"
    os.makedirs("results", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 結果を保存しました: {output_file}")
    print("✅ テスト完了！")

if __name__ == "__main__":
    run_comparison_test()