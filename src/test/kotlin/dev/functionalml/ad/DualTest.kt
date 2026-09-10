package dev.functionalml.ad

import kotlin.math.E
import kotlin.math.PI
import kotlin.test.Test
import kotlin.test.assertEquals

class DualTest {
    @Test
    fun `grad computes derivative of x squared plus three x`() {
        val f: (Dual) -> Dual = { x -> x * x + 3.0 * x }

        val df = grad(f)

        assertEquals(7.0, df(2.0), absoluteTolerance = 1e-12)
    }

    @Test
    fun `valueAndGrad returns value and derivative together`() {
        val f: (Dual) -> Dual = { x -> x * x + 3.0 * x }

        val result = valueAndGrad(f, 2.0)

        assertEquals(10.0, result.value, absoluteTolerance = 1e-12)
        assertEquals(7.0, result.derivative, absoluteTolerance = 1e-12)
    }

    @Test
    fun `sin follows the chain rule`() {
        val df = grad { x -> sin(x * x) }

        assertEquals(2.0 * PI * kotlin.math.cos(PI * PI), df(PI), absoluteTolerance = 1e-12)
    }

    @Test
    fun `exp follows the chain rule`() {
        val df = grad { x -> exp(2.0 * x) }

        assertEquals(2.0 * E * E, df(1.0), absoluteTolerance = 1e-12)
    }
}
