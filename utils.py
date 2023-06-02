# Check if CUDA available
def check_cuda():
	# CUDA?
	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

# Get Model Summary
def get_summary(nn):
	!pip install torchsummary
	from torchsummary import summary
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	model = nn.to(device)
	summary(model, input_size=(1, 28, 28))