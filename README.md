# Functional ML Kotlin

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
