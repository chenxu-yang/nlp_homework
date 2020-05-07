import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    print(n)
    houses = []
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        print(line)
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        print(values)
        houses.append(values[0])
    houses=sorted(houses)
    left=0
    right=len(houses)-1
    distance=0
    while left<right:
        distance+=houses[right]-houses[left]
        left+=1
        right-=1
    print(distance)