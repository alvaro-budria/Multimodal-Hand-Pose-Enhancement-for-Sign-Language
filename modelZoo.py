import numpy as np
import torch
from torch import nn


class regressor_fcn_bn_32_b2h(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_32_b2h, self).__init__()

	def build_net(self, feature_in_dim, feature_out_dim, require_image=False, default_size=256):
		self.require_image = require_image
		self.default_size = default_size
		self.use_resnet = True

		embed_size = default_size
		if self.require_image:
			embed_size += default_size
			if self.use_resnet:
				self.image_resnet_postprocess = nn.Sequential(
					nn.Dropout(0.5),
					nn.Linear(1000*2, default_size),  # 1000 is the size of ResNet50's embeddings (2 hands)
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(default_size, momentum=0.01),
				)
				self.image_reduce = nn.Sequential(
					nn.MaxPool1d(kernel_size=2, stride=2),
				)

		self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,256,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(256),
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


	## create image embedding
	def process_image(self, image_):
		B, T, _ = image_.shape 
		image_ = image_.view(-1, 1000*2)
		feat = self.image_resnet_postprocess(image_)
		feat = feat.view(B, T, self.default_size)
		feat = feat.permute(0, 2, 1).contiguous()
		feat = self.image_reduce(feat)
		return feat


	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 


	## forward pass through generator
	def forward(self, input_, audio_=None, percent_rand_=0.7, feats_=None):
		B, T = input_.shape[0], input_.shape[2]

		fourth_block = self.encoder(input_)
		if self.require_image:
			feat = self.process_image(feats_)
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

		sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
		sixth_block = self.skip4(sixth_block)

		fifth_block = sixth_block + fifth_block
		fifth_block = self.skip5(fifth_block)

		output = self.decoder(fifth_block)
		return output


class regressor_fcn_bn_32(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_32, self).__init__()

	def build_net(self, feature_in_dim, feature_out_dim, require_text=None, default_size=256):
		self.require_text = require_text
		self.default_size = default_size

		embed_size_encoder = default_size
		embed_size = default_size
		if self.require_text:
			embed_size += default_size

			self.text_embeds_postprocess = nn.Sequential(
				nn.Dropout(0.5),
				nn.Linear(512, default_size),  # 512 is the size of CLIP's text embeddings
				nn.LeakyReLU(0.2, True),
				nn.BatchNorm1d(default_size, momentum=0.01),
			)
			self.text_reduce = nn.Sequential(
				nn.MaxPool1d(kernel_size=2, stride=2),
			)

		self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,embed_size_encoder,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size_encoder),
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
	def process_text(self, text_, T):  # "v1"
		text_ = text_.unsqueeze(1).repeat(1, T, 1)
		B, _, E = text_.shape
		text_ = text_.view(-1, E)
		feat = self.text_embeds_postprocess(text_)
		feat = feat.view(B, T, self.default_size)
		feat = feat.permute(0, 2, 1).contiguous()
		feat = self.text_reduce(feat)
		return feat

	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 

	## forward pass through generator
	def forward(self, input_, audio_=None, percent_rand_=0.7, feats_=None):
		B, T = input_.shape[0], input_.shape[2]
		# print(f"input_.shape: {input_.shape}")
		fourth_block = self.encoder(input_)
		if self.require_text:  # "v1"
			# print(text_.shape)
			feat = self.process_text(feats_, T)
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

		sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
		sixth_block = self.skip4(sixth_block)

		fifth_block = sixth_block + fifth_block
		fifth_block = self.skip5(fifth_block)

		output = self.decoder(fifth_block)
		return output


class regressor_fcn_bn_32_v2(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_32_v2, self).__init__()

	def build_net(self, feature_in_dim, feature_out_dim, require_text=None, default_size=256):
		self.require_text = require_text
		self.default_size = default_size

		self.embed_size = default_size
		if self.require_text:
			self.embed_size += default_size
			self.text_embeds_postprocess = nn.Sequential(
				nn.Dropout(0.5),
				nn.Linear(512, self.embed_size),  # 512 is the size of CLIP's text embeddings
				nn.LeakyReLU(0.2, True),
				nn.BatchNorm1d(self.embed_size, momentum=0.01),
			)

		self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)

		self.conv5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.conv6 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.conv7 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.skip4 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)
		self.skip5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.decoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),

			nn.Dropout(0.5),
			nn.ConvTranspose1d(self.embed_size, feature_out_dim, 7, stride=2, padding=3, output_padding=1),
			nn.ReLU(True),
			nn.BatchNorm1d(feature_out_dim),

			nn.Dropout(0.5),
			nn.Conv1d(feature_out_dim, feature_out_dim, 7, padding=3),
		)

	## create text embedding
	def process_text(self, feats_):
		feats_ = feats_.unsqueeze(1)
		B, TT, E = feats_.shape
		feats_ = feats_.view(-1, E)
		feat = self.text_embeds_postprocess(feats_)
		feat = feat.view(B, TT, self.embed_size)  # TT should == 1
		feat = feat.permute(0, 2, 1).contiguous()
		return feat

	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 

	## forward pass through generator
	def forward(self, input_, audio_=None, percent_rand_=0.7, feats_=None):
		B, T = input_.shape[0], input_.shape[2]
		fourth_block = self.encoder(input_)

		fifth_block = self.conv5(fourth_block)
		sixth_block = self.conv6(fifth_block)
		seventh_block = self.conv7(sixth_block)

		if self.require_text:
			feat = self.process_text(feats_)
			seventh_block = torch.cat((seventh_block, feat), dim=2)

		sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
		sixth_block = self.skip4(sixth_block)

		fifth_block = sixth_block + fifth_block
		fifth_block = self.skip5(fifth_block)

		output = self.decoder(fifth_block)
		return output


class regressor_fcn_bn_32_v4(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_32_v4, self).__init__()

	def build_net(self, feature_in_dim, feature_out_dim, require_text=None, default_size=256):
		self.require_text = require_text
		self.default_size = default_size

		self.embed_size = default_size
		if self.require_text:
			self.embed_size += default_size
			self.text_embeds_postprocess = nn.Sequential(
				nn.Dropout(0.5),
				nn.Linear(512, self.embed_size//2),  # 512 is the size of CLIP's text embeddings
				nn.LeakyReLU(0.2, True),
				nn.BatchNorm1d(self.embed_size//2, momentum=0.01),
			)

		self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)

		self.conv5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.conv6 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.conv7 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size//(1+self.require_text),5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size//(1+self.require_text)),
		)

		self.skip4 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.skip5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.decoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),

			nn.Dropout(0.5),
			nn.ConvTranspose1d(self.embed_size, feature_out_dim, 7, stride=2, padding=3, output_padding=1),
			nn.ReLU(True),
			nn.BatchNorm1d(feature_out_dim),

			nn.Dropout(0.5),
			nn.Conv1d(feature_out_dim, feature_out_dim, 7, padding=3),
		)

	## create text embedding
	def process_text(self, feats_, T):
		feats_ = feats_.unsqueeze(1).repeat(1, T, 1)
		B, _, E = feats_.shape
		feats_ = feats_.view(-1, E)
		feat = self.text_embeds_postprocess(feats_)
		feat = feat.view(B, T, -1)
		feat = feat.permute(0, 2, 1).contiguous()
		return feat

	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 

	## forward pass through generator
	def forward(self, input_, audio_=None, percent_rand_=0.7, feats_=None):
		B, T = input_.shape[0], input_.shape[2]
		fourth_block = self.encoder(input_)

		fifth_block = self.conv5(fourth_block)
		sixth_block = self.conv6(fifth_block)
		seventh_block = self.conv7(sixth_block)

		if self.require_text:
			T = seventh_block.shape[2]
			feat = self.process_text(feats_, T)
			seventh_block = torch.cat((seventh_block, feat), dim=1)

		sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
		sixth_block = self.skip4(sixth_block)

		fifth_block = sixth_block + fifth_block
		fifth_block = self.skip5(fifth_block)

		output = self.decoder(fifth_block)
		return output


class regressor_fcn_bn_32_v4_deeper(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_32_v4_deeper, self).__init__()

	def build_net(self, feature_in_dim, feature_out_dim, require_text=None, default_size=256):
		self.require_text = require_text
		self.default_size = default_size

		self.embed_size = default_size
		if self.require_text:
			self.embed_size += default_size
			self.text_embeds_postprocess = nn.Sequential(
				nn.Dropout(0.5),
				nn.Linear(512, self.embed_size//2),  # 512 is the size of CLIP's text embeddings
				nn.LeakyReLU(0.2, True),
				nn.BatchNorm1d(self.embed_size//2, momentum=0.01),
			)

		self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)

		self.conv5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.conv6 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.conv7 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.conv8 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.conv9 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size//(1+self.require_text),3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size//(1+self.require_text)),
		)

		self.conv10 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size//(1+self.require_text),self.embed_size//(1+self.require_text),3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size//(1+self.require_text)),
		)

		self.skip1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.skip2 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.skip3 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)
		self.skip4 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.decoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),

			nn.Dropout(0.5),
			nn.ConvTranspose1d(self.embed_size, feature_out_dim, 7, stride=2, padding=3, output_padding=1),
			nn.ReLU(True),
			nn.BatchNorm1d(feature_out_dim),

			nn.Dropout(0.5),
			nn.Conv1d(feature_out_dim, feature_out_dim, 7, padding=3),
		)

	## create text embedding
	def process_text(self, text_, T):
		text_ = text_.unsqueeze(1).repeat(1, T, 1)
		B, _, E = text_.shape
		text_ = text_.view(-1, E)
		feat = self.text_embeds_postprocess(text_)
		feat = feat.view(B, T, -1)
		feat = feat.permute(0, 2, 1).contiguous()
		return feat

	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 

	## forward pass through generator
	def forward(self, input_, audio_=None, percent_rand_=0.7, feats_=None):
		fourth_block = self.encoder(input_)

		fifth_block = self.conv5(fourth_block)
		sixth_block = self.conv6(fifth_block)
		seventh_block = self.conv7(sixth_block)

		eighth_block = self.conv8(seventh_block)
		ninth_block = self.conv9(eighth_block)
		tenth_block = self.conv10(ninth_block)

		ninth_block = tenth_block + ninth_block
		if self.require_text:
			T = ninth_block.shape[2]
			feat = self.process_text(feats_, T)
			ninth_block = torch.cat((ninth_block, feat), dim=1)
		ninth_block = self.skip1(ninth_block)

		eighth_block = ninth_block + eighth_block
		eighth_block = self.skip2(eighth_block)

		sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
		sixth_block = self.skip3(sixth_block)

		fifth_block = sixth_block + fifth_block
		fifth_block = self.skip4(fifth_block)

		output = self.decoder(fifth_block)
		return output


class decoder_embed2pose(nn.Module):
	def __init__(self):
		super(decoder_embed2pose, self).__init__()

	def build_net(self, feature_in_dim, feature_out_dim, feature_out_len,require_text=None, default_size=256):
		self.require_text = require_text
		self.default_size = default_size
		self.use_embeds = True

		self.conv1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)
		self.conv2 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),
		)

		self.decoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(self.embed_size,self.embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(self.embed_size),

			nn.Dropout(0.5),
			nn.ConvTranspose1d(self.embed_size, feature_out_dim, 7, stride=2, padding=3, output_padding=1),
			nn.ReLU(True),
			nn.BatchNorm1d(feature_out_dim),

			nn.Dropout(0.5),
			nn.Conv1d(feature_out_dim, feature_out_dim, 7, padding=3),
		)

	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 


	## forward pass through generator
	def forward(self, input_, audio_=None, percent_rand_=0.7, feats_=None):
		B, T = input_.shape[0], input_.shape[2]
		# print(f"input_.shape: {input_.shape}")
		output = None
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
