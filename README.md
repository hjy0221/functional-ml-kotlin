# Functional ML Kotlin

[한국어](README.md) | [English](README.en.md)

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
