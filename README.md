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

Basic syntax:
```
# Read an input into variable `n`
input n;

# Output the value of variable `n`
output n;

# Basic expressions
a = 1;
b = 2;
c = a + b;

# `if` statements
if(6 < 7)
{
    output c:
}

# `while` loops
i = 0;
n = 10;
while(i < n)
{
    i += 1;
}
```
