# IMPORT
import csv
import matplotlib.pyplot as plt

# DATASETS

with open('data.csv') as csvfile: #Dimensions = 24 * 2
	reader = csv.DictReader(csvfile)
	dataset = [(int(row['km']), int(row['price'])) for row in reader]


# y=3.7x+8
# with open('perfect_data.csv') as csvfile:
# 	reader = csv.DictReader(csvfile)
# 	dataset = [(float(row['x']), float(row['y'])) for row in reader]

theta = [0, 0]


# MODEL TO DATA
def estimatePrice(mileage: float, theta: list[float]) -> float:
	return (theta[1] * mileage) + theta[0]

# COST FUNCTION
def cost_theta_0(theta: list[float]) -> float:
	total_cost = 0
	for i in range(len(dataset)):
		total_cost += (estimatePrice(dataset[i][0], theta) - dataset[i][1])
	return total_cost * (1 / len(dataset))

def cost_theta_1(theta: list[float]) -> float:
	total_cost = 0
	for i in range(len(dataset)):
		total_cost += ((estimatePrice(dataset[i][0], theta) - dataset[i][1]) * dataset[i][0])
	return total_cost * (1 / len(dataset))

def grad_0(theta: list[float]):
	total_cost = 0
	for i in range(len(dataset)):
		total_cost += ((estimatePrice(dataset[i][0], theta) - dataset[i][1]) ** 2)
	return total_cost * (1 / 2 * len(dataset))

def grad_1(theta: list[float]) -> float:
	total_cost = 0
	for i in range(len(dataset)):
		total_cost += (((estimatePrice(dataset[i][0], theta) - dataset[i][1]) * dataset[i][0]) ** 2)
	return total_cost * (1 / 2 * len(dataset))

# GRADIENT DESCENT ALGORITHM
def gradient_descent(theta: list[float], learningRate: float, n_iterations: int):
	# print(theta)
	for _ in range(n_iterations):
		tmp_theta = (theta[0], theta[1])
		print(f"BEFORE GRAD : {tmp_theta}")
		# print(grad_0(theta=tmp_theta))
		# print(f"{cost_theta_0(theta=tmp_theta)} - {cost_theta_1(theta=tmp_theta)}")
		theta[0] = theta[0] - (learningRate * cost_theta_0(theta=tmp_theta))
		theta[1] = theta[1] - (learningRate * cost_theta_1(theta=tmp_theta))

		# theta[0] = theta[0] - (learningRate * cost_theta_0(theta=tmp_theta))
		# theta[1] = theta[1] - (learningRate * cost_theta_1(theta=tmp_theta))
		# theta[0] = (learningRate * cost_theta_0(theta=tmp_theta))
		# theta[1] = (learningRate * cost_theta_1(theta=tmp_theta))
		print(f"AFTER GRAD : {tmp_theta}")
		print()
		# theta[1] = theta[1] - learningRate * grad_1(theta=tmp_theta)
	print(theta)
	return theta

# print(cost_theta_0(theta=[8, 3.7]))
# print(cost_theta_1(theta=[8, 3.7]))
new_theta = gradient_descent(theta, 0.05, 300)

# VISUALIZE DATA
# x, y = dataset

plt.plot(
	[pair[0] for pair in dataset],
	[ estimatePrice(mileage=dataset[i][0], theta=new_theta) for i in range(len(dataset))],
	c='g'
)

plt.scatter(*zip(*dataset))
plt.plot(
	[pair[0] for pair in dataset],
	[ estimatePrice(mileage=dataset[i][0], theta=[0, 0]) for i in range(len(dataset))],
	c='r'
)

plt.show()

# print(*zip(*dataset))
