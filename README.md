# Functional ML Kotlin

[日本語](README.ja.md) · [English](README.md) · [한국어](README.ko.md)

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
