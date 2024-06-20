from easydict import EasyDict as eDict

def getArg():
	arg = eDict()

	arg.batch = 16
	arg.batch1 = 1
	arg.epoch = 200
	arg.lr = 0.0002
	arg.seed = 21
	arg.save_capacity = 5
	arg.train_img_image = "C:/Users/dahae/pix2pix/dataset/image"
	arg.train_img_label = "C:/Users/dahae/pix2pix/dataset/label"
	arg.test_img_image = "C:/Users/dahae/pix2pix/dataset/Test"
	arg.output_path = "C:/Users/dahae/pix2pix/output/" # 결과 폴더
	arg.path2weights_gen = "C:/Users/dahae/pix2pix/output/200epoch/weights_gen_200.pt"

	# arg.train_worker = 4
	# arg.valid_worker = 4
	# arg.test_worker = 4

	arg.custom_name = "200epoch"
	arg.test_batch = 4
	arg.csv_size = 256

	return arg
