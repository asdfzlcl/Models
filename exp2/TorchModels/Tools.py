class Priority_Queue():
    def __init__(self, comparator, getFeature):
        self.last_index = 0
        self.dic = {}
        self.queue = [0]
        self.comparator = comparator
        self.feature = getFeature

    def swap(self, id1, id2):
        self.queue[id1], self.queue[id2] = self.queue[id2], self.queue[id1]
        self.dic[self.feature(self.queue[id1])] = id1
        self.dic[self.feature(self.queue[id2])] = id2

    def update(self, id):
        while id > 1:
            fa = id // 2
            if self.comparator(self.queue[fa], self.queue[id]):
                break
            self.swap(fa, id)
            id = fa

        while id * 2 <= self.last_index:
            ls = id * 2
            rs = ls + 1
            if rs > self.last_index:
                rs = ls
            if self.comparator(self.queue[rs], self.queue[ls]):
                ls = rs
            if self.comparator(self.queue[id], self.queue[ls]):
                break
            self.swap(ls, id)
            id = ls

    def push(self, x):
        self.queue.append(x)
        self.last_index += 1
        self.update(self.last_index)

    def delID(self, id):
        self.swap(self.last_index, id)
        self.last_index = self.last_index - 1
        self.queue.pop()
        self.update(id)

    def delFeature(self, x):
        self.delID(self.dic[self.feature(x)])
        self.dic[self.feature(x)] = -1

    def pop(self):
        feature = self.getTopFeature()
        self.delID(1)
        self.dic[feature] = -1

    def updateValue(self, x):
        id = self.dic[self.feature(x)]
        self.queue[id] = x
        self.update(id)

    def getTop(self):
        if self.last_index <= 0:
            return [-1,-1]
        return self.queue[1]

    def getTopFeature(self):
        if self.last_index <= 0:
            return [-1,-1]
        return self.feature(self.queue[1])


def comparator(x, y):
    return x[1] > y[1]


def getFeature(x):
    return x[0]


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
