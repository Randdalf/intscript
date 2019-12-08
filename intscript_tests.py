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


class IntcodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intcode = intscript('samples/intcode.is')

        # Find the index where memory starts in the program.
        for i in reversed(range(len(cls.intcode))):
            if cls.intcode[i] == 99:
                cls.memstart = i + 1
                break

    def execution_test(slf, program, *inputs):
        """Check that a program runs identically on a real intcode computer."""
        actual = intcode(slf.intcode, len(program), *program, *inputs)
        expected = intcode(program, *inputs)
        slf.assertEqual(actual.outputs, expected.outputs)
        memory = actual.memory[slf.memstart:slf.memstart+len(program)]
        slf.assertEqual(memory, expected.memory)

    def test_fibonacci(slf):
        fibonacci = intscript('samples/fibonacci.is')
        slf.execution_test(fibonacci, 0)
        slf.execution_test(fibonacci, 1)
        slf.execution_test(fibonacci, 5)
        slf.execution_test(fibonacci, 10)

    def test_d02_example1(slf):
        slf.execution_test([1, 0, 0, 0, 99])

    def test_d02_example2(slf):
        slf.execution_test([2, 3, 0, 3, 99])

    def test_d02_example3(slf):
        slf.execution_test([2, 4, 4, 5, 99, 0])

    def test_d02_example4(slf):
        slf.execution_test([1, 1, 1, 4, 99, 5, 6, 0, 99])

    def test_d02_example5(slf):
        slf.execution_test([1, 9, 10, 3, 2, 3, 11, 0, 99, 30, 40, 50])

    def test_d02_puzzle(slf):
        puzzle = "1,0,0,3,1,1,2,3,1,3,4,3,1,5,0,3,2,1,6,19,2,19,6,23,1,23,5,27,1,9,27,31,1,31,10,35,2,35,9,39,1,5,39,43,2,43,9,47,1,5,47,51,2,51,13,55,1,55,10,59,1,59,10,63,2,9,63,67,1,67,5,71,2,13,71,75,1,75,10,79,1,79,6,83,2,13,83,87,1,87,6,91,1,6,91,95,1,10,95,99,2,99,6,103,1,103,5,107,2,6,107,111,1,10,111,115,1,115,5,119,2,6,119,123,1,123,5,127,2,127,6,131,1,131,5,135,1,2,135,139,1,139,13,0,99,2,0,14,0"
        program = list(map(int, puzzle.split(',')))
        program[1] = 89
        program[2] = 76
        cpu = intcode(slf.intcode, len(program), *program)
        slf.assertEqual(cpu.memory[slf.memstart], 19690720)

    def test_d05_example1(slf):
        example1 = [3, 9, 8, 9, 10, 9, 4, 9, 99, -1, 8]
        slf.execution_test(example1, 8)
        slf.execution_test(example1, 13)

    def test_d05_example2(slf):
        example2 = [3, 9, 7, 9, 10, 9, 4, 9, 99, -1, 8]
        slf.execution_test(example2, 7)
        slf.execution_test(example2, 13)

    def test_d05_example3(slf):
        example3 = [3, 3, 1108, -1, 8, 3, 4, 3, 99]
        slf.execution_test(example3, 8)
        slf.execution_test(example3, 13)

    def test_d05_example4(slf):
        example4 = [3, 3, 1107, -1, 8, 3, 4, 3, 99]
        slf.execution_test(example4, 7)
        slf.execution_test(example4, 13)

    def test_d05_example5(slf):
        example5 = [3, 12, 6, 12, 15, 1, 13, 14, 13, 4, 13, 99, -1, 0, 1, 9]
        slf.execution_test(example5, -1)
        slf.execution_test(example5, 0)
        slf.execution_test(example5, 1)

    def test_d05_example6(slf):
        example6 = [3, 3, 1105, -1, 9, 1101, 0, 0, 12, 4, 12, 99, 1]
        slf.execution_test(example6, -1)
        slf.execution_test(example6, 0)
        slf.execution_test(example6, 1)

    def test_d05_example7(slf):
        example7 = [3, 21, 1008, 21, 8, 20, 1005, 20, 22, 107, 8, 21, 20, 1006, 20, 31, 1106, 0, 36, 98, 0, 0, 1002, 21, 125, 20, 4, 20, 1105, 1, 46, 104, 999, 1105, 1, 46, 1101, 1000, 1, 20, 4, 20, 1105, 1, 46, 98, 99]
        slf.execution_test(example7, 7)
        slf.execution_test(example7, 8)
        slf.execution_test(example7, 9)

    def test_d05_puzzle(slf):
        puzzle = "3,225,1,225,6,6,1100,1,238,225,104,0,1102,59,58,224,1001,224,-3422,224,4,224,102,8,223,223,101,3,224,224,1,224,223,223,1101,59,30,225,1101,53,84,224,101,-137,224,224,4,224,1002,223,8,223,101,3,224,224,1,223,224,223,1102,42,83,225,2,140,88,224,1001,224,-4891,224,4,224,1002,223,8,223,1001,224,5,224,1,223,224,223,1101,61,67,225,101,46,62,224,1001,224,-129,224,4,224,1002,223,8,223,101,5,224,224,1,223,224,223,1102,53,40,225,1001,35,35,224,1001,224,-94,224,4,224,102,8,223,223,101,6,224,224,1,223,224,223,1101,5,73,225,1002,191,52,224,1001,224,-1872,224,4,224,1002,223,8,223,1001,224,5,224,1,223,224,223,102,82,195,224,101,-738,224,224,4,224,1002,223,8,223,1001,224,2,224,1,224,223,223,1101,83,52,225,1101,36,77,225,1101,9,10,225,1,113,187,224,1001,224,-136,224,4,224,1002,223,8,223,101,2,224,224,1,224,223,223,4,223,99,0,0,0,677,0,0,0,0,0,0,0,0,0,0,0,1105,0,99999,1105,227,247,1105,1,99999,1005,227,99999,1005,0,256,1105,1,99999,1106,227,99999,1106,0,265,1105,1,99999,1006,0,99999,1006,227,274,1105,1,99999,1105,1,280,1105,1,99999,1,225,225,225,1101,294,0,0,105,1,0,1105,1,99999,1106,0,300,1105,1,99999,1,225,225,225,1101,314,0,0,106,0,0,1105,1,99999,1007,226,226,224,1002,223,2,223,1006,224,329,1001,223,1,223,1108,226,226,224,102,2,223,223,1006,224,344,101,1,223,223,1007,677,677,224,102,2,223,223,1006,224,359,101,1,223,223,1108,677,226,224,1002,223,2,223,1005,224,374,1001,223,1,223,7,677,226,224,102,2,223,223,1005,224,389,1001,223,1,223,1008,677,677,224,1002,223,2,223,1005,224,404,101,1,223,223,108,226,226,224,1002,223,2,223,1006,224,419,101,1,223,223,1008,226,677,224,1002,223,2,223,1006,224,434,1001,223,1,223,1107,677,226,224,1002,223,2,223,1005,224,449,101,1,223,223,1008,226,226,224,102,2,223,223,1005,224,464,1001,223,1,223,8,226,226,224,1002,223,2,223,1006,224,479,1001,223,1,223,107,226,677,224,102,2,223,223,1005,224,494,1001,223,1,223,7,226,226,224,102,2,223,223,1005,224,509,1001,223,1,223,107,226,226,224,102,2,223,223,1005,224,524,101,1,223,223,107,677,677,224,1002,223,2,223,1006,224,539,101,1,223,223,8,677,226,224,1002,223,2,223,1006,224,554,101,1,223,223,1107,677,677,224,1002,223,2,223,1005,224,569,101,1,223,223,108,226,677,224,1002,223,2,223,1006,224,584,101,1,223,223,7,226,677,224,1002,223,2,223,1005,224,599,1001,223,1,223,8,226,677,224,102,2,223,223,1006,224,614,1001,223,1,223,108,677,677,224,1002,223,2,223,1006,224,629,1001,223,1,223,1007,226,677,224,1002,223,2,223,1006,224,644,101,1,223,223,1108,226,677,224,102,2,223,223,1005,224,659,1001,223,1,223,1107,226,677,224,102,2,223,223,1006,224,674,1001,223,1,223,4,223,99,226"
        program = list(map(int, puzzle.split(',')))
        slf.execution_test(program, 1)
        slf.execution_test(program, 5)

if __name__ == '__main__':
    unittest.main()
