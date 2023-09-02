import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.io as pio
pio.renderers.default = 'colab'
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F




#Load data set
dataset = np.load(r'z3-z4dataset.npy')

batch_size=64 #batch_size

class Data(Dataset):
    def __init__(self, data):
        super(Data, self).__init__()
        self.data = data

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index,:,:]
        data = np.expand_dims(data, axis=0)
        return data

#Training set and test set
train_dataset,test_dataset = train_test_split(dataset, random_state=10)

m = len(train_dataset)
train_data, val_data = random_split(train_dataset, [m-int(m*0.2), int(m*0.2)])
train_data,val_data = np.array(train_data),np.array(val_data)

train_data = Data(train_data)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = Data(val_data)
valid_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

test_dataset = Data(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#Define the convolution autoencoder
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        ###encoder
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.cnn3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.cnn4 = nn.Conv2d(128, 1, kernel_size=(2, 2))

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, return_indices=True)

        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn4 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ###decoder
        self.decnn1 = nn.ConvTranspose2d(1, 128, kernel_size=(2, 2))
        self.decnn2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3))
        self.decnn3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3))
        self.decnn4 = nn.ConvTranspose2d(32, 1, kernel_size=(3, 3))

        self.depool1 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=1)
        self.depool2 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)
        self.depool3 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=1)
        self.depool4 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=1)

        self.bn5 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn6 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn7 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn8 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def encode(self, x):
        x = self.cnn1(x)
        up3out_shape = x.shape
        x = self.bn1(x)
        x = F.relu(x)
        x, i1 = self.pool1(x)


        x = self.cnn2(x)
        up2out_shape = x.shape
        x = self.bn2(x)
        x = F.relu(x)
        x, i2 = self.pool2(x)

        x = self.cnn3(x)
        up1out_shape = x.shape
        x = self.bn3(x)
        x = F.relu(x)
        x, i3 = self.pool3(x)

        x = self.cnn4(x)
        up0out_shape = x.shape
        x, i4 = self.pool4(x)
        x = self.bn4(x)
        z = F.relu(x)
        print(z.shape)

        return z, up3out_shape, i1, up2out_shape, i2, up1out_shape, i3, up0out_shape, i4


    def decode(self, z, up3out_shape, i1, up2out_shape, i2, up1out_shape, i3, up0out_shape, i4):

        x = self.depool1(z, output_size=up0out_shape, indices=i4)
        x = self.decnn1(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.depool2(x, output_size=up1out_shape, indices=i3)
        x = self.decnn2(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.depool3(x, output_size=up2out_shape, indices=i2)
        x = self.decnn3(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.depool4(x, output_size=up3out_shape, indices=i1)
        x = self.decnn4(x)
        x = self.bn8(x)
        y = F.relu(x)

        return y

    def forward(self, x):
        z, up3out_shape, i1, up2out_shape, i2, up1out_shape, i3, up0out_shape, i4 = self.encode(x)
        x_reconst = self.decode(z, up3out_shape, i1, up2out_shape, i2, up1out_shape, i3, up0out_shape, i4)
        return z, x_reconst



### Set the random seed for reproducible results
torch.manual_seed(0)

### Define an optimizer (both for the encoder and the decoder!)
lr = 0.0001 # Learning rate

######mse+L1
criterion1 = torch.nn.SmoothL1Loss()
criterion2 = nn.MSELoss()

def loss_fn(recon_x, x):
    """
    recon_x: generating images
    x: origin images
    """
    L1 = criterion1(recon_x, x)
    MSE = criterion2(recon_x, x)
    return 0.2*L1+MSE



# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')


model = CAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=6e-05)
# Move both the encoder and the decoder to the selected device
model.to(device)

### Training function
z_true = None
def train_epoch(model,epoch,device, train_loader, loss_fn,z_true,num_epochs):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for batch_idx, image_batch in enumerate(train_loader): # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.float().to(device)
        # Encode data  # Decode data
        encoded_data, decoded_data = model(image_batch)
        encoded_data = encoded_data.reshape(encoded_data.shape[0], -1)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
        if (epoch + 1) == num_epochs:
            z_true_one = encoded_data.detach().cpu().numpy()

            if z_true is None:
                z_true = z_true_one

            else:
                z_true = np.vstack([z_true, z_true_one])

    return np.mean(train_loss),z_true


### Testing function
def test_epoch(model,epoch, device, test_loader, loss_fn,zval_true,num_epochs):
    # Set evaluation mode for encoder and decoder
    model.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for batch_idx, image_batch in enumerate(test_loader):
            # Move tensor to the proper device
            image_batch = image_batch.float().to(device)

            # Encode data # Decode data
            encoded_data, decoded_data = model(image_batch)
            encoded_data = encoded_data.reshape(encoded_data.shape[0], -1)

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)

        if (epoch + 1) == num_epochs:
            z_true_one = encoded_data.detach().cpu().numpy()

            if zval_true is None:
                zval_true = z_true_one

            else:
                zval_true = np.vstack([zval_true, z_true_one])
    return val_loss.data,zval_true


def plot_ae_outputs(model,n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[i+5]
      img = torch.from_numpy(img)
      img = img.type(torch.FloatTensor)
      img = torch.unsqueeze(img, dim=0).to(device)
      model.eval()
      with torch.no_grad():
         z, rec_img = model(img)
      plt.imshow(img.cpu().squeeze().numpy(), cmap='seismic')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='seismic')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()

def train():
    num_epochs = 1
    history = {'train_loss': [], 'val_loss': []}
    zval_true = None
    z_true = None
    for epoch in range(num_epochs):

        train_loss, z_true = train_epoch(model, epoch, device, train_loader, loss_fn, z_true,num_epochs)
        val_loss, zval_true = test_epoch(model, epoch, device, valid_loader, loss_fn, zval_true,num_epochs)
        print(
            '\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss,
                                                                            val_loss))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if (epoch + 1) == num_epochs:
            plot_ae_outputs(model, n=5)

    #Save network
    torch.save(model, r'niuzhuang(L1mse0.2).pth')

    val_loss, zval_true = test_epoch(model, epoch, device, valid_loader, loss_fn, zval_true,num_epochs)
    print('val_loss:', val_loss.item())  # Plot losses
    plt.figure(figsize=(10, 8))
    plt.semilogy(history['train_loss'], label='Train')
    plt.semilogy(history['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid()
    plt.legend()
    plt.title('loss')
    plt.show()

if __name__ == '__main__':
  train()
































