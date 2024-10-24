#!/bin/python3

from LinearRegression import LinearRegression
from Colors import *
import argparse

def main():
	parser = argparse.ArgumentParser(description="Train a model using the provided configuration.")

	parser.add_argument(
		'-f', '--file',
		type=str,
		required=True,
		help="The configuration file for training the model."
	)
	parser.add_argument(
		'-s', '--steps',
		type=int,
		default=1000,
		help=f"The number of iterations the AI will go through to train itself (default: 1000)."
	)
	parser.add_argument(
		'-sd', '--show-datas',
		type=float,
		default=0.01,
		help="The learning rate that influences how quickly the model converges (default: 0.01)."
	)
	parser.add_argument(
		'-a', '--animate',
		action='store_true',
		help="Enable animation of the training curves."
	)

	args = parser.parse_args()

	try:
		config_file = args.file
		iterations = args.iterations
		learning_rate = args.learning_rate
		animate = args.animate

		# print(f"Config File: {config_file}")
		# print(f"Iterations: {iterations}")
		# print(f"Learning Rate: {learning_rate}")
		# print(f"Animate: {animate}")

		linearRegression = LinearRegression()
		linearRegression.train_model(
			config_file=config_file,
			iterations=iterations,
			learningRate=learning_rate,
			animate=animate
		)

	except Exception as e:
		error_msg = f"{URED}{BHRED}Error{RESET}\n"
		error_msg += f"{BHRED}Name: {RED}{type(e).__name__}\n"
		error_msg += f"{BHRED}Message: {RED}{e}\n"
		print(error_msg)

if __name__ == "__main__":
	main()
