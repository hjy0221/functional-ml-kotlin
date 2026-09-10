package dev.functionalml

import dev.functionalml.ad.Dual
import dev.functionalml.ad.times
import dev.functionalml.ad.valueAndGrad

fun main() {
    val f: (Dual) -> Dual = { x -> x * x + 3.0 * x }
    val x = 2.0
    val result = valueAndGrad(f, x)

    println("f(x) = x^2 + 3x")
    println("x = $x")
    println("f(x) = ${result.value}")
    println("f'(x) = ${result.derivative}")
}
