import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from create_dataset import *
from utils import *

from single_task_model import Net

# def hook_y(grad):
# 	print("yes")
# 	print(grad)

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--task', default='semantic', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--mode', default='vae', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='/home/uesr/Downloads/mtan-master/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--sample_num', default=1, type=int, help='No of samples')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
opt = parser.parse_args()

if(opt.task =="depth"):
	output_nc = 1
elif(opt.task=="semantic"):
	output_nc = 13
elif(opt.task=="normal"):
	output_nc = 3

# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtan_vae = Net(3, output_nc, opt.mode).to(device)
optimizer = optim.Adam(mtan_vae.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(mtan_vae),
                                                         count_parameters(mtan_vae) / 24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

# define dataset
dataset_path = opt.dataroot

nyuv2_train_set = NYUv2(root=dataset_path, train=True,augmentation=True)
    

nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 2
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False)

beta = 0.00001

def single_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, beta, total_epoch=200):
	train_batch = len(train_loader)
	test_batch = len(test_loader)

	for index in range(total_epoch):
	# iteration for all batches
		multi_task_model.train()
		train_dataset = iter(train_loader)
		avg_train_loss = np.zeros([3], dtype=np.float32)
		avg_test_loss = np.zeros([3], dtype=np.float32)
		del_beta = 0

		running_mean, running_log_var = 0, 0
		for k in range(train_batch):
			multi_task_model.zero_grad()
			optimizer.zero_grad()
			train_data, train_label, train_depth, train_normal = train_dataset.next()
			train_data, train_label = train_data.to(device), train_label.long().to(device)
			train_depth, train_normal = train_depth.to(device), train_normal.to(device)

			if(opt.task=="semantic"):
				y_true = train_label
			elif(opt.task=="depth"):
				y_true = train_depth
			elif(opt.task=="normal"):
				y_true = train_normal

			if(opt.mode == "normal"):
				pred = multi_task_model(train_data)

				loss = model_fit(pred, y_true, opt.task)
				loss.backward()
				optimizer.step()
			elif(opt.mode=="vae"):
				pred, mu, log_var, sig = multi_task_model(train_data)
				loss = model_fit(pred, y_true, opt.task)
				loss.backward(retain_graph=True)
				# optimizer.step()
				# print(sig.shape)
				sig = multi_task_model.bottleneck[3].weight.grad.mean(1).mean(1).mean(1)
				# print(sig.shape)
				kl_loss =  -0.5 *(1 + log_var - mu**2 - log_var.exp())
				kl_loss = kl_loss.mean(2).mean(2)
				kl_loss = beta * torch.mean(-sig*kl_loss)
				# print(kl_loss)
				# multi_task_model.zero_grad()
				# optimizer.zero_grad()
				kl_loss.backward()
				# del_beta += kl_loss.detach().item()
				# g = torch.autograd.grad(loss, sig)
				optimizer.step()

			avg_train_loss[0] += loss.item()
			score = get_scores(pred, y_true, opt.task)
			avg_train_loss[1] += score[0]
			avg_train_loss[2] += score[1]

		
		print("Beta", beta, del_beta)
		# if(beta<=0.5):
		# 	beta += 0.00001


		test_dataset = iter(test_loader)
		for k in range(test_batch):
			test_data, test_label, test_depth, test_normal = test_dataset.next()
			test_data, test_label = test_data.to(device), test_label.long().to(device)
			test_depth, test_normal = test_depth.to(device), test_normal.to(device)

			if(opt.task=="semantic"):
				y_true = test_label
			elif(opt.task=="depth"):
				y_true = test_depth
			elif(opt.task=="normal"):
				y_true = test_normal

			if(opt.mode == "normal"):
				pred = multi_task_model(test_data)
				
			elif(opt.mode=="vae"):
				pred, mu, log_var, _ = multi_task_model(test_data)

		
			# avg_test_loss[0] += loss.item()
			score = get_scores(pred, y_true, opt.task)
			avg_test_loss[1] += score[0]
			avg_test_loss[2] += score[1]


		avg_train_loss = avg_train_loss/(train_batch)
		avg_test_loss = avg_test_loss/(test_batch)

		scheduler.step()

		print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}  ||'
		'TEST: {:.4f} {:.4f} {:.4f} |'
		.format(index, avg_train_loss[0], avg_train_loss[1], avg_train_loss[2],
			avg_test_loss[0], avg_test_loss[1], avg_test_loss[2]))

		# if(index>10):
		


single_task_trainer(nyuv2_train_loader, nyuv2_test_loader, mtan_vae, device, optimizer, scheduler, opt, beta)