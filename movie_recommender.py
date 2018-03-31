import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

def recommend(model, data, uid):

	#number of users and movies in training data
	users, items = data['train'].shape

	#generate recommendations for each user we input
	for x in uid:

		#movies user already like
		positive_values = data['item_labels'][data['train'].tocsr()[x].indices]

		#movies prediction by model users will like
		score = model.predict(x, np.arange(items))

		#ranking them in order of most liked first
		top_list = data['item_labels'][np.argsort(-score)]

		#printing the results
		print("User: "+str(x))
		print("		Positive Values:")

		for y in positive_values[:3]:
			print("			"+str(y))

		print("		Recommended Values:")

		for y in top_list[:3]:
			print("			"+str(y))

if __name__ == "__main__":

	# fetch data with minimum rating 4
	data = fetch_movielens(min_rating = 4.0)

	#print(type(repr(data['train'])))
	print(repr(data['train']))
	print(repr(data['test']))

	#create model
	model = LightFM(loss="warp")

	#train model
	model.fit(data['train'], epochs=30, num_threads=2)

	recommend(model, data, [3, 25, 450])