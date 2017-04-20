import hashlib
from tools import holybook

import sys

_hash = lambda text, start, end, length: hash(text[start:end].encode('utf-8')) % length

scripture = (holybook.holytext)

def hash_split(test_part: float, set_size: int, result_size: int = -1, hash = _hash):
	if not(0.0 <= test_part <= 1.0):
		raise Exception('Dafaq m8?')

	if result_size == -1:
		result_size = set_size

	result = []
	test_size = int(result_size * test_part)
	num = 0
	q = scripture[num]
	scripture_len = len(scripture)
	while len(q) < test_size:
		num += 1
		q += scripture[num % scripture_len]
	num = 1
	for i in range(test_size):

		p = hash(q, 0, i, set_size)

		while p in result:
			p = hash(q, i - num, i, set_size)
			num += 1
		num = 0
		result.append(p)

	return result
