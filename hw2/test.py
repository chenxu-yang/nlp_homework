def count(n):
    dp=[1]+[0]*n
    for i in range(n+1):
        for j in range(i):
            if dp[j]==1 and (i-j-1)//4%2==1:
                dp[i]+=1
    return sum(dp)
if __name__=='__main__':
    print(count(4))
    print(count(5))
    print(count(6))
    print(count(10))