def calculate_v1(v2, v3):
    return (0.3 * (4 + 0.9 * v3)) + (0.8 * (2 + 0.9 * v2))

def calculate_v2(v2, v3):
    return (0.3 * (7 + 0.9 * v3)) + (0.7 * (7 + 0.9 * v2))


v2 = 1
v3 = 70

for i in range(10000):
    v2 = calculate_v2(v2, v3)

for i in range(10000):
    v1 = calculate_v1(v2, v3)

print(v2)
print(v1)