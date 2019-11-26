

f = open("New_essays.txt", "r", encoding="utf-8")
f_out = open("New_essays_out.txt", "w", encoding="utf-8")
T = ""
for line in f:
	T += line
for i in range(1000):
	T = T.replace("\n\n", "\n")
f_out.write(T)