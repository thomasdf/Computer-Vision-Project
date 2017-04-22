



def next_batch_pipe(loader, batch_size: int, images_used: int, is_training: bool, conn):
	print('pipe niqab!')
	x, y = loader.next_batch(batch_size, images_used, is_training)
	conn.send((x, y))
	conn.close()
