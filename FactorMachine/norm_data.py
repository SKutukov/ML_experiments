file = open("/home/skutukov/work/FactorMachine/full_set.txt", "r")
customers = {}
movies = {}
i = 0
j = 0
k = 0
for line in file:
    if not (line.split(' ')[1] in customers):
        customers[line.split(' ')[1]] = i
        #print(len(customers), customers)
        i += 1
    if not(line.split(' ')[0] in movies):
        movies[line.split(' ')[0]] = j
        j += 1
        #print(len(movies))
#    print(k/len(file))
    k += 1
file.close()
print(i, j, k)
file = open("/home/skutukov/work/FactorMachine/full_set.txt", "r")
file1 = open("/home/skutukov/work/FactorMachine/full_norm_set.txt", "w")
for line in file:
    file1.write(str(movies[line.split(' ')[0]]) + ' ' + str(customers[line.split(' ')[1]]) + " " +  line.split(' ')[2])
file1.close()
