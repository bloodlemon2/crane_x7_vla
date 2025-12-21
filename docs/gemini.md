# crane_x7_gemini

Google Gemini Robotics-ER API integration for CRANE-X7 robot manipulation.

## 概要

このパッケージは、Google Gemini Robotics-ER 1.5 モデルをCRANE-X7ロボットアームと統合し、ビジョンベースの物体認識とマニピュレーションタスクを実行します。

## 機能

- **物体検出**: Gemini APIを使用したリアルタイム物体検出
- **座標変換**: 2D画像座標から3Dロボット座標への変換
- **軌道プランニング**: Geminiの推論を使用した自然言語ベースの軌道生成
- **MoveIt統合**: 生成された軌道をMoveItで実行

## 前提条件

### APIキーの設定

Google Gemini APIキーが必要です：

```bash
export GEMINI_API_KEY="your-api-key-here"
```

または、起動時にパラメータとして渡すこともできます：

```bash
ros2 launch crane_x7_gemini trajectory_planner.launch.py api_key:=your-api-key-here
```

### 依存関係

- ROS 2 Humble
- `google-genai` Python パッケージ
- OpenCV
- MoveIt2

## 使用方法

### 軌道プランナー起動

自然言語指示から軌道を生成し実行：

```bash
ros2 launch crane_x7_gemini trajectory_planner.launch.py
```

#### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `api_key` | 環境変数`GEMINI_API_KEY` | Gemini APIキー |
| `model_id` | `gemini-robotics-er-1.5-preview` | Geminiモデル |
| `image_topic` | `/camera/color/image_raw` | 入力カメラトピック |
| `depth_topic` | `/camera/aligned_depth_to_color/image_raw` | 深度画像トピック |
| `prompt_topic` | `/gemini/task_prompt` | タスクプロンプトトピック |
| `move_group` | `arm` | MoveItグループ名 |
| `end_effector_link` | `crane_x7_gripper_base_link` | エンドエフェクタリンク |
| `planning_time` | `5.0` | プランニング時間（秒） |
| `execute_trajectory` | `true` | 軌道を実行するか |
| `temperature` | `0.5` | モデル温度（0.0-1.0） |
| `thinking_budget` | `0` | 推論バジェット |

### タスク指示の送信

別のターミナルから指示を送信：

```bash
ros2 topic pub /gemini/task_prompt std_msgs/msg/String \
  "data: 'Pick up the red cube and place it on the table'"
```

### crane_x7_bringupからの起動

実機またはシミュレーションと統合して起動：

```bash
# 実機 + Gemini
ros2 launch crane_x7_bringup gemini_real.launch.py

# シミュレーション + Gemini
ros2 launch crane_x7_bringup gemini_sim.launch.py
```

## パッケージ構成

```
crane_x7_gemini/
├── crane_x7_gemini/
│   ├── __init__.py
│   ├── gemini_node.py          # Gemini APIクライアント
│   ├── object_detector.py      # 物体検出ノード
│   ├── coordinate_transformer.py # 座標変換
│   ├── trajectory_planner.py   # 軌道プランナー
│   └── prompt_publisher.py     # プロンプト送信ユーティリティ
├── launch/
│   └── trajectory_planner.launch.py
└── setup.py
```

## トピック

### Subscribe

| トピック | 型 | 説明 |
|---------|-----|------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | カメラ画像入力 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 深度画像入力 |
| `/gemini/task_prompt` | `std_msgs/String` | タスク指示 |

### Publish

| トピック | 型 | 説明 |
|---------|-----|------|
| `/gemini/detections` | `std_msgs/String` | JSON形式の検出結果 |

## 検出結果の例

```bash
ros2 topic echo /gemini/detections
```

出力例：
```json
[
  {
    "point": [500, 300],
    "label": "red cube"
  },
  {
    "point": [600, 400],
    "label": "blue cylinder"
  }
]
```

## ライセンス

- **オリジナルコード**: MIT License（Copyright 2025 nop）

## 参考資料

- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [CRANE-X7 Documentation](https://github.com/rt-net/crane_x7_ros)
