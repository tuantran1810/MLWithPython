# list of 4 elements
list1 = ['Alpha', 'Beta', 'Gamma', 'Sigma']
# list of 5 elements
list2 = ['one', 'two', 'three', 'six', 'five']
# list of 3 elments
list3 = [1, 2, 3]

test = zip(list1, list2, list3)  # zip the values
cnt = 0

print(list(test))
print(test)

print('\nPrinting the values of zip')
for values in test:
    print(values)  # print each tuples
    cnt+=1

print('Zip file contains ', cnt, 'elements.');