# intscript
Compiles programs which run on the [intcode computer](https://adventofcode.com/2019/day/5) used by Advent of Code.

Install requirements using:
```
pip install -r requirements.txt
```

To compile a program:
```
python intscript.py samples/fibonacci.is > samples/fibonacci.ic
```

To run a program pass the intcode program followed by the inputs:
```
python intcode.py samples/fibonacci.ic 10
```
