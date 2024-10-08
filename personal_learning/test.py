
nb = 7

def has_solutions(nb: int):
	i = 0
	while True:
		if nb == 1:
			return True
			break
		if nb % 2 == 0:
			nb = nb / 2
			# print(f"{i}: {nb}")
		else:
			nb = nb * 3 + 1
			# print(f"{i}: {nb}")
		i += 1

for i in range(1, 1000000000):
	print(i)
	has_solutions(i)
	# if has_solutions(i) == False:

