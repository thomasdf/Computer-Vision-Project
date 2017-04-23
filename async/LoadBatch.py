import multiprocessing


def next_batch_pipe(loader, batch_size: int, images_used: int, is_training: bool, conn):
	print('pipe niqab!')
	x, y = loader.next_batch(batch_size, images_used, is_training)
	conn.send((x, y))
	conn.close()


def next_batch_mp(loader, batch_size: int, images_used: int, is_training: bool, batch_x, batch_y):
	x, y = loader.next_batch(batch_size, images_used, is_training)


def next_batch_queue(loader, batch_size: int, images_used: int, is_training: bool, batch_queue_x, batch_queue_y):
	x, y = loader.next_batch(batch_size, images_used, is_training)

	batch_queue_x.put(x)
	batch_queue_y.put(y)

	return


def next_batch_queue_2(loader, batch_size: int, images_used: int, is_training: bool, batch_queue_x, batch_queue_y, lock):
	loader.next_batch_async(batch_size, images_used, is_training, batch_queue_x, batch_queue_y, lock)


def next_batch_queue_3(args):
	loader, batch_size, images_used, is_training, batch_queue_x, batch_queue_y, lock = args
	loader.next_batch_async(batch_size, images_used, is_training, batch_queue_x, batch_queue_y, lock)

def next_batch_arr(loader, batch_size: int, images_used: int, is_training: bool, batch_arr_x, batch_arr_y):
	loader.next_batch_async_arr(batch_size, images_used, is_training, batch_arr_x, batch_arr_y)
