import numpy as np


class Priority_Queue():
    def __init__(self, comparator, getFeature):
        self.last_index = 0
        self.dic = {}
        self.queue = [0]
        self.comparator = comparator
        self.feature = getFeature

    def clear(self):
        self.last_index = 0
        self.dic = {}
        self.queue = [0]

    def swap(self, id1, id2):
        self.queue[id1], self.queue[id2] = self.queue[id2], self.queue[id1]
        self.dic[self.feature(self.queue[id1])] = id1
        self.dic[self.feature(self.queue[id2])] = id2

    def update(self, ID):
        while ID > 1:
            fa = ID // 2
            if self.comparator(self.queue[fa], self.queue[ID]):
                break
            self.swap(fa, ID)
            ID = fa

        while ID * 2 <= self.last_index:
            ls = ID * 2
            rs = ls + 1
            if rs > self.last_index:
                rs = ls
            if self.comparator(self.queue[rs], self.queue[ls]):
                ls = rs
            if self.comparator(self.queue[ID], self.queue[ls]):
                break
            self.swap(ls, ID)
            ID = ls

    def push(self, x):
        self.queue.append(x)
        self.last_index += 1
        self.update(self.last_index)

    def delID(self, ID):
        self.swap(self.last_index, ID)
        self.last_index = self.last_index - 1
        self.queue.pop()
        self.update(ID)

    def delFeature(self, x):
        self.delID(self.dic[self.feature(x)])
        self.dic[self.feature(x)] = -1

    def pop(self):
        feature = self.getTopFeature()
        self.delID(1)
        self.dic[feature] = -1

    def updateValue(self, x):
        ID = self.dic[self.feature(x)]
        self.queue[ID] = x
        self.update(ID)

    def getTop(self):
        if self.last_index <= 0:
            return [-1, -1]
        return self.queue[1]

    def getTopFeature(self):
        if self.last_index <= 0:
            return [-1, -1]
        return self.feature(self.queue[1])


def comparator(x, y):
    return x[1] > y[1]


def getFeature(x):
    return x[0]


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if __name__ == '__main__':
    queue = Priority_Queue(comparator, getFeature)
    queue.push([1, 10])
    queue.push([3, 3])
    queue.push([9, 12])
    queue.push([4, 13])
    print(queue.queue)
    print(queue.dic)
    queue.delFeature([9])
    print(queue.queue)
    print(queue.dic)
    queue.updateValue([1, 22])
    print(queue.queue)
    print(queue.dic)
    queue.updateValue([1, 1])
    print(queue.queue)
    print(queue.dic)
    for i in range(3):
        print(queue.getTop())
        queue.pop()
        # print(queue.queue)
        # print(queue.dic)
