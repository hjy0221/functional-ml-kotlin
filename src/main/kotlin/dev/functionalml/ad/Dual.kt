package dev.functionalml.ad

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin

data class Dual(
    val value: Double,
    val derivative: Double,
) {
    operator fun plus(other: Dual): Dual =
        Dual(value + other.value, derivative + other.derivative)

    operator fun plus(other: Double): Dual =
        Dual(value + other, derivative)

    operator fun times(other: Dual): Dual =
        Dual(
            value * other.value,
            derivative * other.value + value * other.derivative,
        )

    operator fun times(other: Double): Dual =
        Dual(value * other, derivative * other)
}

operator fun Double.plus(other: Dual): Dual =
    other + this

operator fun Double.times(other: Dual): Dual =
    other * this

fun variable(value: Double): Dual =
    Dual(value = value, derivative = 1.0)

fun constant(value: Double): Dual =
    Dual(value = value, derivative = 0.0)

fun sin(x: Dual): Dual =
    Dual(value = sin(x.value), derivative = cos(x.value) * x.derivative)

fun exp(x: Dual): Dual {
    val expValue = exp(x.value)
    return Dual(value = expValue, derivative = expValue * x.derivative)
}

fun valueAndGrad(f: (Dual) -> Dual, x: Double): Dual =
    f(variable(x))

fun grad(f: (Dual) -> Dual): (Double) -> Double =
    { x -> valueAndGrad(f, x).derivative }
