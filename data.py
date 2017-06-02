# def get_primes(number):
#     while True:
#         if is_prime(number):
#             yield number
#         number += 1

import numpy as np 

def generate_batch(train, target, batch_size):
	assert train.shape[0] == target.shape[0]
	l_index = 0
	r_index = batch_size
	while r_index <= train.shape[0]:
		yield (train[l_index:r_index], target[l_index:r_index])
		l_index = r_index
		r_index += batch_size
	yield None