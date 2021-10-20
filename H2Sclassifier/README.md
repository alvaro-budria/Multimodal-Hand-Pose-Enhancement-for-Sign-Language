# 2D_to_3D_conversion

The goal of this repo is to predict the depth coordinate of 2D videos. 

- **Data**<br />
	The videos in the dataset are recordings of people speaking in sign language. For each video, 26 body keypoints of the person have been extracted to obtain a 		stick-figure representation. The depth coordinate for each keypoint has been estimated using a multiple-camera set-up. The data is stored in the following two 		numpy arrays:<br />
	- *body_data.npy* stores for each video in the dataset and for each frame in the video the 2D-coordinates of each keypoint ([x_1, y_1, x_2, y_2, ..., x_26, y_26]). It has shape [NUM_VIDEOS, NUM_FRAMES, 52] <br />
	- *body_ground.npy* stores for each video in the dataset and for each frame in the video the depth coordiante of each keypoint ([z_1, ..., z_26]). It has shape 	[NUM_VIDEOS, NUM_FRAMES, 26] <br />
	
	This folder is not available in the repo.	
	
- **Preprocessing**<br />
	In the preprocessing folder the data is normalized and prepared to be fed into the neural network. There are the following files:<br />
	- *skeleton_parts.py* contains a dictionary to convert from keypoint number to bodypart and viceversa. 
	- *plot_3D_skeleton.py* contains the function *plot_3D_skeleton* that allows to visualize animated 3D plots of the keypoint figures.
	- *rotate_skeleton.py* contains the function *rotate_skeleton* that centers the skeleton in the origin of coordinates and rotates it so that its column (Mid-Hip to Neck vector) is in the Y axis and it is facing forwards (which means that the Nose to Neck vector is in the XY plane). 
	- *scale_axes.py* contains the function *scale_axes* that computes the 2D-length of the skeleton's column and scales the 3 axes with this length. This normalization is inspired in the paper *Can 3D Pose be Learned from 2D Projections Alone, Dylan Drover, Rohith MV, Ching-Hang Chen, ECCVW 2018*
	- *main.py* performs all of the mentioned normalization steps to finally obtain the final xyz_data has shape [TOTAL_NUM_FRAMES, 26, 3] (important remark: with the rotation, the depth coordinate is now dim0 or x)
	
- **Model**<br />
	This folder contains the neural network class DepthLSTM. It consists of a LSTM layer with input size = 52 (two coordinates for each joint), arbitrary hidden_size and num_layers followed by a Linear layer with input size = hidden_size and output size = 26 (one depth coordinate for each joint). LSTM neural networks are explained here: http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
	
	
- **Test_Train**<br />
		This folder contains the scripts for training and testing the model:
	- *train_epoch.py*: trains the model for one epoch. It implements *stateful training*: the dataset is divided in batches, and each batch is divided in consecutive windows of fixed length SEQ_LEN. Then each window is forwarded through the model, the state from the last frame is saved and it used to start training of the following window.
	- *test_epoch.py*: given a model and some test data without the depth coordinate, it returns the average MSE loss and the predicted depth coordinate.
	- *main.py*: this script performs the training and testing of the model for NUM_EPOCHS, plots the MSE loss for training and testing and visualizes the ground truth and predicted skeletons 
		
		
	
