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
