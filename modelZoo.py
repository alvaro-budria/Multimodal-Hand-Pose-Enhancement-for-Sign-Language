import numpy as np

rng = np.random.RandomState(23456)

import torch
from torch import nn


class regressor_fcn_bn_32(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_32, self).__init__()

	def build_net(self, feature_in_dim, feature_out_dim, require_text=None, default_size=256):
		self.require_text = require_text
		self.default_size = default_size
		self.use_embeds = True

		embed_size = default_size
		if self.require_text:
			embed_size += default_size
			if self.use_embeds:
				self.text_embeds_postprocess = nn.Sequential(
					nn.Dropout(0.5),
					nn.Linear(512, default_size),  # 512 is the size of CLIP's embeddings
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(default_size, momentum=0.01),
				)

		# if self.require_text == "v2":
		# 	embed_size += default_size
		# 	if self.use_embeds:
		# 		self.text_embeds_postprocess = nn.Sequential(
		# 			nn.Dropout(0.5),
		# 			nn.Linear(768, default_size),  # 768 is the size of BERT's embeddings
		# 			nn.LeakyReLU(0.2, True),
		# 			nn.BatchNorm1d(default_size, momentum=0.01),
		# 		)
		# 		self.text_reduce = nn.Sequential(
		# 			nn.MaxPool1d(kernel_size=2, stride=2),
		# 		)

		self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)

		self.conv5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv6 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv7 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		# self.conv8 = nn.Sequential(
		# 	nn.Dropout(0.5),
		# 	nn.Conv1d(embed_size,embed_size,3,padding=1),
		# 	nn.LeakyReLU(0.2, True),
		# 	nn.BatchNorm1d(embed_size),
		# )

		# self.conv9 = nn.Sequential(
		# 	nn.Dropout(0.5),
		# 	nn.Conv1d(embed_size,embed_size,3,padding=1),
		# 	nn.LeakyReLU(0.2, True),
		# 	nn.BatchNorm1d(embed_size),
		# )

		# self.conv10 = nn.Sequential(
		# 	nn.Dropout(0.5),
		# 	nn.Conv1d(embed_size,embed_size,3,padding=1),
		# 	nn.LeakyReLU(0.2, True),
		# 	nn.BatchNorm1d(embed_size),
		# )

		# self.skip1 = nn.Sequential(
		# 	nn.Dropout(0.5),
		# 	nn.Conv1d(embed_size,embed_size,3,padding=1),
		# 	nn.LeakyReLU(0.2, True),
		# 	nn.BatchNorm1d(embed_size),
		# )
		# self.skip2 = nn.Sequential(
		# 	nn.Dropout(0.5),
		# 	nn.Conv1d(embed_size,embed_size,3,padding=1),
		# 	nn.LeakyReLU(0.2, True),
		# 	nn.BatchNorm1d(embed_size),
		# )
		self.skip4 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
		self.skip5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.decoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),

			nn.Dropout(0.5),
			nn.ConvTranspose1d(embed_size, feature_out_dim, 7, stride=2, padding=3, output_padding=1),
			nn.ReLU(True),
			nn.BatchNorm1d(feature_out_dim),

			nn.Dropout(0.5),
			nn.Conv1d(feature_out_dim, feature_out_dim, 7, padding=3),
		)


	## create text embedding
	def process_text(self, text_, T):
		feat = text_
		
		feat = text_.repeat(1, T//2, 1)  # For a sequence of  T frames, we only have one text embedding.
										 # So we replicate this embedding T/2 times so that we can concatenate it to the body enconder's output. 
		B, T, _ = feat.shape 
		text_ = text_.view(-1, 768)
		feat = self.text_embeds_postprocess(text_)
		feat = feat.view(B, T, self.default_size)
		feat = feat.permute(0, 2, 1).contiguous()
		return feat


	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 


	## forward pass through generator
	def forward(self, input_, audio_=None, percent_rand_=0.7, text_=None):
		B, T = input_.shape[0], input_.shape[2]
		fourth_block = self.encoder(input_)
		if self.require_text == "v1":
			feat = self.process_text(text_, T)
			fourth_block = torch.cat((fourth_block, feat), dim=1)

		fifth_block = self.conv5(fourth_block)
		sixth_block = self.conv6(fifth_block)
		seventh_block = self.conv7(sixth_block)
		# eighth_block = self.conv8(seventh_block)
		# ninth_block = self.conv9(eighth_block)
		# tenth_block = self.conv10(ninth_block)

		# ninth_block = tenth_block + ninth_block
		# ninth_block = self.skip1(ninth_block)

		# eighth_block = ninth_block + eighth_block
		# eighth_block = self.skip2(eighth_block)
		if self.require_text == "v2":
			seventh_block = torch.cat((seventh_block, feat), dim=2)
		# d = torch.randn([8, 256, 1])
		# # bring d into the same format, and then concatenate tensors
		# seventh_block = torch.cat((seventh_block, d), dim=-1)


		sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
		sixth_block = self.skip4(sixth_block)

		fifth_block = sixth_block + fifth_block
		fifth_block = self.skip5(fifth_block)

		output = self.decoder(fifth_block)
		return output 


class regressor_fcn_bn_discriminator(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_discriminator, self).__init__()

	def build_net(self, feature_in_dim):
		self.convs = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,64,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(64),
			## 64

			nn.Dropout(0.5),
			nn.Conv1d(64,64,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(64),
			## 32

			nn.Dropout(0.5),
			nn.Conv1d(64,32,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(32),
			## 16

			nn.Dropout(0.5),
			nn.Conv1d(32,32,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(32),
			## 8

			nn.Dropout(0.5),
			nn.Conv1d(32,16,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(16),
			## 4

			nn.Dropout(0.5),
			nn.Conv1d(16,16,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(16),
			## 2

			nn.Dropout(0.5),
			nn.Conv1d(16,8,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(8),
			## 1

			nn.Dropout(0.5),
			nn.Conv1d(8,1,3,padding=1),
		)

	def forward(self, input_):
		outputs = self.convs(input_)
		return outputs
