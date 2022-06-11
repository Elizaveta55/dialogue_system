f = open("ru.txt", "r", encoding='utf-8')
lines = f.readlines()
f.close()

print(len(lines))

## save only 5% of data
f = open("ru_5.txt", "w", encoding='utf-8')
for i in range(119137):
    f.write(lines[i])
f.close()