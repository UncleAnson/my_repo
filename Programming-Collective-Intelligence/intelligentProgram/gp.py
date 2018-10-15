from random import random, randint, choice
from copy import deepcopy
from math import log


class fwrapper:
    def __init__(self, function, childcount, name): # childcount 参数个数
        self.function = function
        self.childcount = childcount
        self.name = name


class node:
    def __init__(self, fw, children):
        self.function = fw.function
        self.name = fw.name
        self.children = children

    def evaluate(self, inp):
        results = [n.evaluate(inp) for n in self.children]
        return self.function(results)

    # def display(self, indent=0):
    #     print((' ' * indent) + self.name)
    #     for c in self.children:
    #         c.display(indent + 1)
    #
    # def drawnode(self, draw, x, y):
    #     draw.text((x - 5, y), str(self.name), (0, 0, 0))
    #     width_sum = sum([getwidth(child) for child in self.children]) * 100
    #     left = x - width_sum // 2
    #     right = x + width_sum // 2
    #
    #     for child in self.children:
    #         width = getwidth(child) * 100
    #         left_tmp = left + width // 2
    #         draw.line((x, y + 10, left_tmp, y + 100), fill=(255, 0, 0))
    #         child.drawnode(draw, left_tmp, y + 100)
    #         left += width


class paramnode:
    def __init__(self, idx):
        self.idx = idx

    def evaluate(self, inp):
        return inp[self.idx]

    # def display(self, indent=0):
    #     print('%sp%d' % (' ' * indent, self.idx))
    #
    # def drawnode(self, draw, x, y):
    #     draw.text((x - 5, y), 'p%d' % self.idx, (0, 0, 0))


class constnode:
    def __init__(self, v):
        self.v = v

    def evaluate(self, inp):
        return self.v

    # def display(self, indent=0):
    #     print('%s%d' % (' ' * indent, self.v))
    #
    # def drawnode(self, draw, x, y):
    #     draw.text((x, y), str(self.v), (0, 0, 0))


addw = fwrapper(lambda l: l[0] + l[1], 2, 'add')
subw = fwrapper(lambda l: l[0] - l[1], 2, 'subtract')
mulw = fwrapper(lambda l: l[0] * l[1], 2, 'multiply')


def iffunc(l):
    if l[0] > 0:
        return l[1]
    else:
        return l[2]

ifw = fwrapper(iffunc, 3, 'if')


def isgreater(l):
    if l[0] > l[1]:
        return 1
    else:
        return 0

gtw = fwrapper(isgreater, 2, 'isgreater')

flist = [addw, mulw, ifw, gtw, subw]


def exampletree():
    return node(ifw, [
        node(gtw, [paramnode(0), constnode(3)]),
        node(addw, [paramnode(1), constnode(5)]),
        node(subw, [paramnode(1), constnode(2)]),
    ])

if __name__ == '__main__':
    exampletree = exampletree()
    # def func(x, y):
    #     if x>3:
    #         return y+5
    #     else:
    #         return y-2
    print(exampletree.evaluate([2,3])) # 将参数统一传给待评估的三个children（等待调用）
    print(exampletree.evaluate([5,3]))