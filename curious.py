# def har_sum(n, sum=0, step=1):
#     if step > n:
#         return

#     sum += 1 / step
#     print(f"{sum:.4f}")
#     har_sum(n, sum, step + 1)


# # def har_sum_until(n):
# #     for i in range(1, n):
# #         print(f"{har_sum(i):.4f}")

# har_sum(10)

print("--------------------")


def har_sum_iterative(n):
    sum = 0
    for i in range(2, n + 1):
        sum += 1 / i
        print(f"{sum:.4f}")


har_sum_iterative(31)
