# Functional ML Kotlin

[한국어](README.md) | [English](README.en.md)

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
