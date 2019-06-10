file = open("/home/skutukov/work/FactorMachine/full_norm_set.txt", "r")
file1 = open("/home/skutukov/work/FactorMachine/cut_norm_set.txt", "w")
k = 0
for line in file:
    file1.write(line)
    k += 1
    if(k >= 1000):
        break

file.close()
file1.close()

print(i, j, k)
