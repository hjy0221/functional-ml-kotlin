# Functional ML Kotlin

[日本語](https://github.com/hjy0221/functional-ml-kotlin#%E6%97%A5%E6%9C%AC%E8%AA%9E) · [English](https://github.com/hjy0221/functional-ml-kotlin#english) · [한국어](https://github.com/hjy0221/functional-ml-kotlin#%ED%95%9C%EA%B5%AD%EC%96%B4)

## 日本語

関数型設計から ML フレームワークの内部を学ぶ、小さな Kotlin プロジェクトです。PyTorch の複製ではなく、JAX のように関数を主 API とし、パラメーターを明示し、高階関数として微分を扱う設計を段階的に実装します。

### 第1段階: Dual Number による forward-mode AD

- スカラー自動微分の `Dual(value, derivative)`
- `Dual` と `Double` の `+`、`*`
- `sin`、`exp`
- 値と微分を同時に計算する `valueAndGrad(f, x)`
- `(Dual) -> Dual` を `(Double) -> Double` に変換する `grad(f)`
- `f(x) = x^2 + 3x` と連鎖律のテスト

### 実行

macOS、JDK 21、JupyterLab、Kotlin Jupyter Kernel でノートブック環境を確認しています。JDK と JupyterLab、Kotlin カーネルの設定手順は英語・韓国語セクションにあります。

```bash
./gradlew notebook
./gradlew run
./gradlew test
```

`notebooks/forward-mode-ad.ipynb` は関数定義、`valueAndGrad`、`grad`、`sin`、`exp` を対話的に示します。次の段階では Scalar MLP、Tensor、損失関数、純粋 SGD、計算グラフ、最適化を追加する予定です。

## English

A tiny Kotlin project for learning ML framework internals through a functional design.

The goal is not to clone PyTorch. The first milestone follows the spirit of JAX: functions are the main API, parameters stay explicit, and differentiation is exposed as a higher-order function.

## Stage 1: Forward-Mode AD With Dual Numbers

Implemented:

- `Dual(value, derivative)` for scalar forward-mode automatic differentiation
- `+` and `*` operator overloads for `Dual` and `Double`
- elementary functions: `sin`, `exp`
- `valueAndGrad(f, x)` to evaluate a function and its derivative together
- `grad(f)` to transform `(Dual) -> Dual` into `(Double) -> Double`
- tests for `f(x) = x^2 + 3x`

Example:

```kotlin
val f: (Dual) -> Dual = { x -> x * x + 3.0 * x }
val df = grad(f)

println(df(2.0)) // 7.0
```

## Interactive Environment Setup

The notebook environment has been verified on macOS with JDK 21, JupyterLab, and the Kotlin Jupyter Kernel.

### 1. Check JDK 21

```bash
/usr/libexec/java_home -v 21
```

Install JDK 21 if it is not available:

```bash
brew install --cask corretto@21
```

### 2. Install JupyterLab

```bash
brew install jupyterlab
```

### 3. Install the Kotlin kernel

Install the Kotlin kernel into the isolated Python environment managed by Homebrew for JupyterLab:

```bash
JUPYTER_PYTHON="$(brew --prefix jupyterlab)/libexec/bin/python"
"$JUPYTER_PYTHON" -m pip install --upgrade kotlin-jupyter-kernel
"$JUPYTER_PYTHON" -m kotlin_kernel fix-kernelspec-location
```

Register a kernel that uses the same JDK 21 as the project:

```bash
JDK_21_HOME=$(/usr/libexec/java_home -v 21)
"$JUPYTER_PYTHON" -m kotlin_kernel add-kernel --name "JDK 21" --jdk "$JDK_21_HOME" --force
```

Verify the installation:

```bash
jupyter kernelspec list
```

The environment is ready when `kotlin_jdk_21` appears in the list.

## Run the Kotlin Notebook

The following command builds the local JAR and opens the example in JupyterLab:

```bash
./gradlew notebook
```

When the browser opens, run [`notebooks/forward-mode-ad.ipynb`](notebooks/forward-mode-ad.ipynb) from top to bottom. The notebook loads the prepared JAR and interactively demonstrates function definition, `valueAndGrad`, `grad`, `sin`, and `exp`.

To prepare only the JAR without opening a browser, run:

```bash
./gradlew prepareNotebook
```

## Run Example

```bash
./gradlew run
```

Expected result:

```text
f(x) = x^2 + 3x
x = 2.0
f(x) = 10.0
f'(x) = 7.0
```

## Run Tests

```bash
./gradlew test
```

## Expansion Roadmap

1. Scalar MLP
   - Add `relu`, subtraction, division, unary minus, and `pow`.
   - Represent scalar layer parameters explicitly with data classes.
   - Build pure functions for `linear`, activation, and small MLP composition.

2. Tensor
   - Introduce immutable `Tensor` with shape and flat storage.
   - Add `map`, `zipWith`, `sum`, `dot`, `matmul`, and broadcasting rules.
   - Keep tensor operations independent from mutable training state.

3. Loss Functions
   - Add MSE first.
   - Later add binary cross entropy and softmax cross entropy.
   - Keep the loss API as `loss(params, batch): Scalar`.

4. Optimizer
   - Start with pure SGD: `update(params, grads, learningRate): Params`.
   - Add momentum or Adam after the parameter tree structure is stable.

5. Computation Graph / IR
   - Add a small expression tree with a `sealed class`.
   - Separate graph construction from evaluation.
   - Use the IR to inspect, transform, or compile scalar programs.

6. Optimization
   - Add simple graph rewrites such as constant folding.
   - Add common subexpression elimination once expression identity is defined.
   - Explore lowering tensor operations into a smaller execution core.

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
