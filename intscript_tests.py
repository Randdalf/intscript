#!/usr/bin/env python

"""intscript unit tests"""

import unittest

from intscript import intscript
from intcode import intcode


def run(program, *inputs):
    return intcode(program, *inputs).outputs


def compile_and_run(file, *inputs):
    return run(intscript(file, *inputs), *inputs)


class IntscriptTests(unittest.TestCase):
    def test_booleans(slf):
        slf.assertEqual(
            compile_and_run('tests/booleans.is'),
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
        )

    def test_comparisons(slf):
        slf.assertEqual(
            compile_and_run('tests/comparisons.is'),
            [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        )

    def test_arrays(slf):
        slf.assertEqual(
            compile_and_run('tests/arrays.is'),
            [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 24]
        )


class FibonacciTests(unittest.TestCase):
    def setUp(slf):
        slf.fibonacci = intscript('samples/fibonacci.is')

    def test_fibonacci_0(slf):
        slf.assertEqual(run(slf.fibonacci, 0), [])

    def test_fibonacci_1(slf):
        slf.assertEqual(run(slf.fibonacci, 1), [1])

    def test_fibonacci_5(slf):
        slf.assertEqual(run(slf.fibonacci, 5), [1, 1, 2, 3, 5])

    def test_fibonacci_9(slf):
        slf.assertEqual(run(slf.fibonacci, 9), [1, 1, 2, 3, 5, 8, 13, 21, 34])

    def test_fibonacci_20(slf):
        slf.assertEqual(run(slf.fibonacci, 20)[19], 6765)


if __name__ == '__main__':
    unittest.main()
