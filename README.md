# Functional ML Kotlin

[日本語](https://github.com/hjy0221/functional-ml-kotlin#%E6%97%A5%E6%9C%AC%E8%AA%9E) · [English](https://github.com/hjy0221/functional-ml-kotlin#english) · [한국어](https://github.com/hjy0221/functional-ml-kotlin#%ED%95%9C%EA%B5%AD%EC%96%B4)

## 日本語

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

## English

A small Kotlin project to learn the inner workings of ML frameworks through functional design.

The goal is not to clone PyTorch. Like JAX, we implement a step-by-step design that uses functions as the central API, handles parameters explicitly, and expresses derivatives as higher-order functions.

## Step 1: Dual Number based Forward-Mode AD

Currently implemented features:

- `Dual(value, derivative)` for scalar forward-mode automatic differentiation
- `+`, `*` operator overloads of `Dual` and `Double`
- Basic functions `sin`, `exp`
- `valueAndGrad(f, x)` calculates function values and differential values together
- `grad(f)`, which converts the `(Dual) -> Dual` function to the `(Double) -> Double` differential function.
- `f(x) = x^2 + 3x` and chain rule testing

Example usage:

```kotlin
val f: (Dual) -> Dual = { x -> x * x + 3.0 * x }
val df = grad(f)

println(df(2.0)) // 7.0
```

## Building an interactive environment

The notebook environment for this project was verified using a combination of macOS, JDK 21, JupyterLab, and Kotlin Jupyter Kernel.

### 1. Check JDK 21

```bash
/usr/libexec/java_home -v 21
```

If you don't have JDK 21, install it first.

```bash
brew install --cask corretto@21
```

### 2. Install JupyterLab

```bash
brew install jupyterlab
```

### 3. Install Kotlin kernel

Install the Kotlin kernel into an isolated Python environment in JupyterLab managed by Homebrew.

```bash
JUPYTER_PYTHON="$(brew --prefix jupyterlab)/libexec/bin/python"
"$JUPYTER_PYTHON" -m pip install --upgrade kotlin-jupyter-kernel
"$JUPYTER_PYTHON" -m kotlin_kernel fix-kernelspec-location
```

Register a kernel using the same JDK 21 as your project.

```bash
JDK_21_HOME=$(/usr/libexec/java_home -v 21)
"$JUPYTER_PYTHON" -m kotlin_kernel add-kernel --name "JDK 21" --jdk "$JDK_21_HOME" --force
```

Check the installation results.

```bash
jupyter kernelspec list
```

If you see `kotlin_jdk_21` in the list, you're ready.

## Run Kotlin Notebook

The following command creates a local JAR and then opens the example notebook in JupyterLab.

```bash
./gradlew notebook
```

When the browser opens, run the cell [`notebooks/forward-mode-ad.ipynb`](notebooks/forward-mode-ad.ipynb) from above. The notebook loads the prepared JAR and interactively runs the function definitions, `valueAndGrad`, `grad`, `sin`, and `exp`.

To just prepare the JAR without automatically opening the browser, use the following command:

```bash
./gradlew prepareNotebook
```

## Run example

```bash
./gradlew run
```

Expected output:

```text
f(x) = x^2 + 3x
x = 2.0
f(x) = 10.0
f'(x) = 7.0
```

## Run tests

```bash
./gradlew test
```

## Expansion Roadmap

1. Scalar MLPs
   - Add `relu`, subtraction, division, unary negative, `pow`.
   - Express scalar layer parameters as explicit data classes.
   - `linear`, activation function, synthesizes small MLPs into pure functions.

2. Tensor
   - Introduce an immutable `Tensor` with shape and flattened storage.
   - Add broadcasting rules with `map`, `zipWith`, `sum`, `dot`, `matmul`.
   - Separate tensor operations from mutable learning states.

3.Loss
   - Implemented starting from MSE.
   - Afterwards, binary cross entropy and softmax cross entropy are added.
   - The loss function API maintains the form `loss(params, batch): Scalar`.

4. Optimizer
   - We start with `update(params, grads, learningRate): Params`, a purely functional SGD.
   - Once the parameter tree structure is stable, add momentum or Adam.

5. Computation Graph / IR
   - Create a small expression tree based on `sealed class`.
   - Separate graph construction and evaluation.
   - Inspect and convert or compile Scala programs via IR.

6.Optimization
   - Implement simple graph rewriting such as constant folding.
   - After determining expression equality rules, add common subexpression elimination.
   - Explore lowering tensor operations to smaller execution cores.

## 한국어

함수형 설계를 통해 ML 프레임워크의 내부 동작을 학습하는 작은 Kotlin 프로젝트입니다.

목표는 PyTorch를 복제하는 것이 아닙니다. JAX처럼 함수를 중심 API로 사용하고, 매개변수를 명시적으로 다루며, 미분을 고차 함수로 표현하는 설계를 단계적으로 구현합니다.

## 1단계: Dual Number 기반 Forward-Mode AD

현재 구현된 기능:

- 스칼라 forward-mode 자동 미분을 위한 `Dual(value, derivative)`
- `Dual`과 `Double`의 `+`, `*` 연산자 오버로드
- 기본 함수 `sin`, `exp`
- 함수값과 미분값을 함께 계산하는 `valueAndGrad(f, x)`
- `(Dual) -> Dual` 함수를 `(Double) -> Double` 미분 함수로 변환하는 `grad(f)`
- `f(x) = x^2 + 3x` 및 연쇄 법칙 테스트

사용 예시:

```kotlin
val f: (Dual) -> Dual = { x -> x * x + 3.0 * x }
val df = grad(f)

println(df(2.0)) // 7.0
```

## 대화형 환경 구축

이 프로젝트의 Notebook 환경은 macOS, JDK 21, JupyterLab, Kotlin Jupyter Kernel 조합으로 검증했습니다.

### 1. JDK 21 확인

```bash
/usr/libexec/java_home -v 21
```

JDK 21이 없다면 먼저 설치합니다.

```bash
brew install --cask corretto@21
```

### 2. JupyterLab 설치

```bash
brew install jupyterlab
```

### 3. Kotlin 커널 설치

Homebrew가 관리하는 JupyterLab의 격리된 Python 환경에 Kotlin 커널을 설치합니다.

```bash
JUPYTER_PYTHON="$(brew --prefix jupyterlab)/libexec/bin/python"
"$JUPYTER_PYTHON" -m pip install --upgrade kotlin-jupyter-kernel
"$JUPYTER_PYTHON" -m kotlin_kernel fix-kernelspec-location
```

프로젝트와 동일한 JDK 21을 사용하는 커널을 등록합니다.

```bash
JDK_21_HOME=$(/usr/libexec/java_home -v 21)
"$JUPYTER_PYTHON" -m kotlin_kernel add-kernel --name "JDK 21" --jdk "$JDK_21_HOME" --force
```

설치 결과를 확인합니다.

```bash
jupyter kernelspec list
```

목록에 `kotlin_jdk_21`이 표시되면 준비가 끝난 것입니다.

## Kotlin Notebook 실행

다음 명령은 로컬 JAR를 만든 뒤 JupyterLab에서 예제 Notebook을 엽니다.

```bash
./gradlew notebook
```

브라우저가 열리면 [`notebooks/forward-mode-ad.ipynb`](notebooks/forward-mode-ad.ipynb)의 셀을 위에서부터 실행합니다. Notebook은 준비된 JAR를 불러와 함수 정의, `valueAndGrad`, `grad`, `sin`, `exp`를 대화형으로 실행합니다.

브라우저를 자동으로 열지 않고 JAR만 준비하려면 다음 명령을 사용합니다.

```bash
./gradlew prepareNotebook
```

## 예제 실행

```bash
./gradlew run
```

예상 출력:

```text
f(x) = x^2 + 3x
x = 2.0
f(x) = 10.0
f'(x) = 7.0
```

## 테스트 실행

```bash
./gradlew test
```

## 확장 로드맵

1. Scalar MLP
   - `relu`, 뺄셈, 나눗셈, 단항 음수, `pow`를 추가합니다.
   - 스칼라 계층 매개변수를 명시적인 data class로 표현합니다.
   - `linear`, 활성화 함수, 작은 MLP를 순수 함수로 합성합니다.

2. Tensor
   - shape와 평탄화된 저장 공간을 갖는 불변 `Tensor`를 도입합니다.
   - `map`, `zipWith`, `sum`, `dot`, `matmul`과 broadcasting 규칙을 추가합니다.
   - Tensor 연산을 변경 가능한 학습 상태와 분리합니다.

3. Loss
   - MSE부터 구현합니다.
   - 이후 binary cross entropy와 softmax cross entropy를 추가합니다.
   - 손실 함수 API는 `loss(params, batch): Scalar` 형태를 유지합니다.

4. Optimizer
   - 순수 함수형 SGD인 `update(params, grads, learningRate): Params`부터 시작합니다.
   - 매개변수 트리 구조가 안정되면 momentum 또는 Adam을 추가합니다.

5. Computation Graph / IR
   - `sealed class` 기반의 작은 표현식 트리를 만듭니다.
   - 그래프 구성과 평가를 분리합니다.
   - IR을 통해 스칼라 프로그램을 검사하고 변환하거나 컴파일합니다.

6. Optimization
   - constant folding 같은 간단한 그래프 재작성부터 구현합니다.
   - 표현식 동등성 규칙을 정한 뒤 common subexpression elimination을 추가합니다.
   - Tensor 연산을 더 작은 실행 코어로 lowering하는 방법을 탐구합니다.
