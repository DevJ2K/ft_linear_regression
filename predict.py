#!/bin/python3

from LinearRegression import LinearRegression
from Colors import *
import argparse

def main():
	parser = argparse.ArgumentParser(description="Use a pre-trained model to make predictions based on the provided input value.")

	parser.add_argument(
		'-f', '--file',
		type=str,
		required=True,
		help="The name of the model file that contains the trained model's parameters."
	)
	parser.add_argument(
		'-m', '--mileage',
		type=int,
		required=True,
		help="The mileage for which you want to predict the price."
	)
	parser.add_argument(
		'-s', '--steps',
		action='store_true',
		help="If enabled, displays the detailed steps taken during the prediction process."
	)
	parser.add_argument(
		'-g', '--graph',
		action='store_true',
		help="If enabled, visualizes the prediction results with a graph."
	)


	args = parser.parse_args()

	try:
		model_file = args.file
		mileage = args.mileage
		steps = args.steps
		graph = args.graph

		linearRegression = LinearRegression()
		linearRegression.use_model(
			model_file=model_file,
			mileage=mileage,
			steps=steps,
			graph=graph
		)

	except Exception as e:
		error_msg = f"{URED}{BHRED}Error{RESET}\n"
		error_msg += f"{BHRED}Name: {RED}{type(e).__name__}\n"
		error_msg += f"{BHRED}Message: {RED}{e}\n"
		print(error_msg)

if __name__ == "__main__":
	main()
