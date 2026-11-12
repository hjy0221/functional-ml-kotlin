# Functional ML Kotlin

[日本語](README.ja.md) · [English](README.md) · [한국어](README.ko.md)

関数型設計を通じてMLフレームワークの内部動作を学習する小さなKotlinプロジェクト。

目標はPyTorchを複製することではありません。 JAXのように関数を中心APIとして使用し、パラメータを明示的に扱い、微分を高次関数として表現する設計を段階的に実装します。

## ステップ1：Dual NumberベースのForward-Mode AD

現在実装されている機能：

- スカラーforward-modeの自動微分のための`Dual(value, derivative)`
- `Dual`と`Double`の`+`、`*`演算子のオーバーロード
- 基本関数`sin`、`exp`
- 関数値と微分値を一緒に計算する`valueAndGrad(f, x)`
- `(Dual) -> Dual`関数を`(Double) -> Double`微分関数に変換する`grad(f)`
- `f(x) = x^2 + 3x`と連鎖法則テスト

使用例：

```kotlin
val f: (Dual) -> Dual = { x -> x * x + 3.0 * x }
val df = grad(f)

println(df(2.0)) // 7.0
```

## 対話型環境の構築

このプロジェクトのノートブック環境は、macOS、JDK 21、JupyterLab、Kotlin Jupyter Kernelの組み合わせで検証しました。

### 1. JDK 21の確認

```bash
/usr/libexec/java_home -v 21
```

JDK 21がない場合は、まずインストールします。

```bash
brew install --cask corretto@21
```

### 2. JupyterLabのインストール

```bash
brew install jupyterlab
```

### 3. Kotlinカーネルのインストール

Homebrewが管理するJupyterLabの孤立したPython環境にKotlinカーネルをインストールします。

```bash
JUPYTER_PYTHON="$(brew --prefix jupyterlab)/libexec/bin/python"
"$JUPYTER_PYTHON" -m pip install --upgrade kotlin-jupyter-kernel
"$JUPYTER_PYTHON" -m kotlin_kernel fix-kernelspec-location
```

プロジェクトと同じJDK 21を使用するカーネルを登録します。

```bash
JDK_21_HOME=$(/usr/libexec/java_home -v 21)
"$JUPYTER_PYTHON" -m kotlin_kernel add-kernel --name "JDK 21" --jdk "$JDK_21_HOME" --force
```

インストール結果を確認してください。

```bash
jupyter kernelspec list
```

リストに`kotlin_jdk_21`が表示されたら、準備は完了です。

## Kotlin Notebookの実行

次のコマンドはローカルJARを作成し、JupyterLabでサンプルノートブックを開きます。

```bash
./gradlew notebook
```

ブラウザが開いたら、[`notebooks/forward-mode-ad.ipynb`](notebooks/forward-mode-ad.ipynb)のセルを上から実行します。ノートブックは準備されたJARを呼び出し、関数定義、`valueAndGrad`、`grad`、`sin`、`exp`を対話的に実行します。

ブラウザを自動的に開かずにJARのみを準備するには、次のコマンドを使用します。

```bash
./gradlew prepareNotebook
```

## 例の実行

```bash
./gradlew run
```

予想出力：

```text
f(x) = x^2 + 3x
x = 2.0
f(x) = 10.0
f'(x) = 7.0
```

## テストの実行

```bash
./gradlew test
```

## 拡張ロードマップ

1. Scalar MLP
   - `relu`、減算、除算、単項負数、`pow`を追加します。
   - スカラー階層パラメータを明示的なデータクラスで表します。
   - `linear`、活性化関数、小さなMLPを純粋関数として合成します。

2. Tensor
   - shapeと平坦化された記憶空間を持つ不変`Tensor`を導入します。
   - `map`、`zipWith`、`sum`、`dot`、`matmul`とブロードキャストルールを追加します。
   - Tensor 操作を変更可能な学習状態から切り離します。

3. Loss
   - MSEから実装します。
   - その後、binary cross entropyとsoftmax cross entropyを追加します。
   - 損失関数APIは`loss(params, batch): Scalar`の形式を維持します。

4. Optimizer
   - 純粋関数型SGDである`update(params, grads, learningRate): Params`から始まります。
   - パラメータツリー構造が安定したら、 momentum または Adam を追加します。

5. Computation Graph/IR
   - `sealed class` ベースの小さな式ツリーを作成します。
   - グラフ構成と評価を分離します。
   - IRを介してスカラープログラムを調べ、変換またはコンパイルします。

6. Optimization
   - constant foldingのような単純なグラフ書き換えから実装します。
   - 式同等性規則を定義した後、common subexpression eliminationを追加します。
   - Tensor操作をより小さな実行コアにローワーする方法を探る。
