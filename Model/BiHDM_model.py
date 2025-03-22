import torch
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from importlib import import_module

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LProjector(nn.Module):
    """A module for the local projector layer in BiHDM.
    Leaky ReLU is used as the activation function to add non-linearity.
    
    Parameters
    ----------
    n : int
        Number of electrodes in a stream (input RNN sequence length).
    d : int
        Number of global high level features in each node.
    k : int
        Number of nodes in the projector layer.
    a : float, optional (default=0.01)
        Slope for LeakyReLU.
    """

    def __init__(self, n, d, k, a=0.01):
        super(LProjector, self).__init__()
        self.n = n
        self.d = d
        self.k = k
        self.a = a

        self.act_func_ = nn.LeakyReLU(a)

        # Weights and bias
        self.weight = nn.Parameter(torch.randn((n, k)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros((d, k)), requires_grad=True)

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')
        b_bound = 1 / math.sqrt(n)
        nn.init.uniform_(self.bias, -b_bound, b_bound) # U(-sqrt(k), sqrt(k)) where k = 1/n

    def forward(self, x):
        '''
        Forward pass of LProjector.

        Parameters
        ----------
        x : torch.Tensor
            Torch tensor of shape s x n x d, where s is the size of sample x.

        Returns
        -------
        g : torch.Tensor
            Torch tensor of shape s x d x k, where s is the size of sample x.
        '''
        ws = torch.einsum('nk,snd->sdk', self.weight, x)
        return torch.add(self.act_func_(ws), self.bias)

class GradReverse(Function):
    """The gradient reversal function.
    Forward propogation does not change input, while backward propogation 
    reverses the gradients.

    Implementation followed this forum answer: 
    https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()
    

class GRL(nn.Module):
    def forward(self, x):
        return GradReverse.apply(x)


class BiHDM(nn.Module):
    '''An implementation of the BiHDM model proposed in [1].

    Parameters
    ----------
    lh_stream : list of int
        Indices for selecting the left hemisphere horizontal stream.
    rh_stream : list of int
        Indices for selecting the right hemisphere horizontal stream.
    lv_stream : list of int
        Indices for selecting the left hemisphere vertical stream.
    rv_stream : list of int
        Indices for selecting the right hemisphere vertical stream.
    n_classes : int
        Number of classes for classification.
    d_input: int
        Number of features for each electrode's raw representation.
    d_stream : int, optional (default=32)
        Number of features for each electrode's deep representation (d_l in paper [1]).
    d_pair: int , optional (default=32)
        Number of features for each electrode's deep representation after the pairwise 
        operation (d_p1, d_p2, or d_p3 in paper [1]).
    d_global: int, optional (default=32)
        Number of global high level features (d_g in paper [1]).
    d_out: int, optional (default=16)
        Number of output features (d_o in paper [1]).
    k: int, optional (default=6)
        Number of nodes for global high level features (K in paper [1]).
    a : float, optional (default=0.01)
        Slope for LeakyReLU in high level feature projector.
    pairwise_operation: str or custom function (default='subtraction')
        Pairwise operation for the two hemispheres' electrode deep representation streams. 
        If string, acceptable operations are 'subtraction', 'addition', 'division', and 
        'inner_product'. If custom function, it should take two torch.Tensor objects (s_left 
        and s_right of sizes L x N x d_stream) and return one torch.Tensor of size L x N x d_pair.
        Where L is the number of electrodes in a stream, N is the batch size.
    output_domain: bool, optional (default=True)
        If the domain adversarial strategy should be performed. If set to true, an extra domain
        discrimination layer will be added in parallel to the class discrimination layer. The domain
        predictions will be returned as the second element of a tuple together with the class predictions
        in the forward method.
    rnn_stream_kwargs: dict, optional (default={})
        kwargs to feed into RNNs extracting electrodes' deep features.
    rnn_global_kwargs: dict, optional (default={})
        kwargs to feed into RNNs extracting global high level features.

    Notes
    -----
    [1] Y. Li et al., “A Novel Bi-Hemispheric Discrepancy Model for EEG Emotion Recognition,” 
        IEEE Trans. Cogn. Dev. Syst., vol. 13, no. 2, pp. 354–367, Jun. 2021.
    '''

    def __init__(self, lh_stream, rh_stream, lv_stream, rv_stream, n_classes,
                 d_input, d_stream=32, d_pair=32, d_global=32, d_out=16, k=6, a=0.01, 
                 pairwise_operation='subtraction', output_domain=True,
                 rnn_stream_kwargs={}, rnn_global_kwargs={}):
        super(BiHDM, self).__init__()

        # Store the inputs as instance variables.
        self.lh_stream = lh_stream
        self.rh_stream = rh_stream
        self.lv_stream = lv_stream
        self.rv_stream = rv_stream
        self.d_input = d_input
        self.d_stream = d_stream
        self.d_pair = d_pair
        self.d_global = d_global
        self.d_out = d_out
        self.k = k
        self.a = a
        self.n_classes = n_classes
        self.pairwise_operation = pairwise_operation
        self.output_domain = output_domain
        self.rnn_stream_kwargs = rnn_stream_kwargs
        self.rnn_global_kwargs = rnn_global_kwargs

        # Define the RNNs for each stream.
        self.rnn_lh_ = nn.RNN(d_input, d_stream, batch_first=False, **rnn_stream_kwargs)
        self.rnn_rh_ = nn.RNN(d_input, d_stream, batch_first=False, **rnn_stream_kwargs)
        self.rnn_lv_ = nn.RNN(d_input, d_stream, batch_first=False, **rnn_stream_kwargs)
        self.rnn_rv_ = nn.RNN(d_input, d_stream, batch_first=False, **rnn_stream_kwargs)

        # Define the pairwise operation to use based on the input argument.
        if pairwise_operation == 'subtraction':
            self.pair_ = self.pairwise_subtraction
        elif pairwise_operation == 'addition':
            self.pair_ = self.pairwise_addition
        elif pairwise_operation == 'division':
            self.pair_ = self.pairwise_division
        elif pairwise_operation == 'inner_product':
            self.pair_ = self.pairwise_inner
        else:
            # Use a custom pairwise operation if one is provided.
            self.pair_ = self.pairwise_operation
        
        # Define the RNNs for the global representations of the two paired streams.
        self.rnn_hg_ = nn.RNN(d_pair, d_global, batch_first=False, **rnn_global_kwargs)
        self.rnn_vg_ = nn.RNN(d_pair, d_global, batch_first=False, **rnn_global_kwargs)

        # Define the LProjector instances for the two streams.
        self.proj_h_ = LProjector(len(lh_stream), d_global, k, a)
        self.proj_v_ = LProjector(len(lh_stream), d_global, k, a)

        # Define the learnable weight matrices for the final linear layers.
        self.map_h_ = nn.Parameter(torch.randn((d_out, d_global)), requires_grad=True)
        self.map_v_ = nn.Parameter(torch.randn((d_out, d_global)), requires_grad=True)

        # Define the output layers.
        self.out_class_ = nn.Sequential(
            nn.Linear(d_out * k, n_classes, bias=True),
            nn.LogSoftmax(dim=-1)
        )
        if self.output_domain:
            self.out_domain_ = nn.Sequential(
                GRL(),
                nn.Linear(d_out * k, 2, bias=True),
                nn.LogSoftmax(dim=-1)
            )

        # Initialize the weights.
        with torch.no_grad():
            self.init_weights()

    def init_weights(self):
        """Initialize weights of the model.

        This method initializes the RNN weights with Xavier uniform distribution,
        and initializes the map weights and output linear weights with Xavier uniform 
        distribution with gain=1.
        """
        # init RNN weights with xavier uniform
        def rnn_init_weights(m):
            if type(m) == nn.RNN:
                for ws in m._all_weights:
                    for w in ws:
                        if 'weight' in w:
                            nn.init.xavier_uniform_(getattr(m, w))
        self.apply(rnn_init_weights)

        # LProjectors were initialised on construction

        # init maps with xavier uniform and gain=1
        nn.init.xavier_uniform_(self.map_h_)
        nn.init.xavier_uniform_(self.map_v_)

        # init output linear weights with xavier uniform and gain=1
        nn.init.xavier_uniform_(self.out_class_[0].weight)
        if self.output_domain:
            nn.init.xavier_uniform_(self.out_domain_[1].weight)

    def pairwise_subtraction(self, sl, sr):
        return sl - sr
    
    def pairwise_addition(self, sl, sr):
        return sl + sr

    def pairwise_division(self, sl, sr):
        return sl / sr

    def pairwise_inner(self, sl, sr):
        '''Column-wise inner product'''
        return torch.einsum('lnd,lnd->ln', sl, sr)[:,:,None]

    def forward(self, x):
        '''
        Compute the forward pass of the BiHDM model.

        Parameters
        ----------
        x : torch.Tensor of shape n_sample x n_channels x d_input
            The input tensor.

        Returns
        -------
        torch.Tensor or tuple
            If self.output_domain is set to False, a tensor of shape (n_sample, n_classes) 
            representing the class probabilities will be returned. Otherwise a tuple of size 2
            will be returned where the first element contains the class probabilities, and the 
            second element contains another tensor of shape (n_sample, 2) representing the domain
            probabilities.
        '''
        # electrode deep representation (len(stream) x n_sample x d_stream)
        lhs, _ = self.rnn_lh_(x[:,self.lh_stream].permute(1,0,2))
        rhs, _ = self.rnn_rh_(x[:,self.rh_stream].permute(1,0,2))
        lvs, _ = self.rnn_lv_(x[:,self.lv_stream].permute(1,0,2))
        rvs, _ = self.rnn_rv_(x[:,self.rv_stream].permute(1,0,2))

        # pairwise operation (len(stream) x n_sample x d_pair)
        ph = self.pair_(lhs, rhs)
        pv = self.pair_(lhs, rhs)

        # high level features (len(stream) x n_sample x d_global)
        gh, _ = self.rnn_hg_(ph)
        gv, _ = self.rnn_vg_(pv)

        # project high level features (n_sample x d_global x k)
        gh = self.proj_h_(gh.permute(1,0,2))
        gv = self.proj_v_(gv.permute(1,0,2))

        # map and summarise (n_sample x d_out x k)
        gh = torch.einsum('og,sgk->sok', self.map_h_, gh)
        gv = torch.einsum('og,sgk->sok', self.map_v_, gv)
        hv = gh + gv
        hv = hv.flatten(start_dim=1)

        if self.output_domain:
            return self.out_class_(hv), self.out_domain_(hv)
        else:
            return self.out_class_(hv)


class BiHDMClassifier(BaseEstimator, ClassifierMixin):
    """
    BiHDMClassifier is a classification algorithm that uses BiHDM (Bivariate Hierarchical
    Dirichlet Models) to extract relevant information from multivariate time-series EEG data.

    Parameters
    ----------
    ch_names : list of str
        List of channel names in the EEG data.
    lh_chs : list of str
        List of channel names in the left hemisphere horizontal stream.
    rh_chs : list of str
        List of channel names in the right hemisphere horizontal stream.
    lv_chs : list of str
        List of channel names in the left hemisphere vertical stream.
    rv_chs : list of str
        List of channel names in the right hemisphere vertical stream.
    d_stream : int, optional (default=32)
        Number of features for each electrode's deep representation (d_l in paper [1]).
    d_pair: int , optional (default=32)
        Number of features for each electrode's deep representation after the pairwise 
        operation (d_p1, d_p2, or d_p3 in paper [1]).
    d_global: int, optional (default=32)
        Number of global high level features (d_g in paper [1]).
    d_out: int, optional (default=16)
        Number of output features (d_o in paper [1]).
    k: int, optional (default=6)
        Number of nodes for global high level features (K in paper [1]).
    a : float, optional (default=0.01)
        Slope for LeakyReLU in high level feature projector.
    pairwise_operation : str, optional (default='subtraction')
        Operation used to compute pairwise interactions (see BiHDM for details).
    rnn_stream_kwargs : dict, optional (default={})
        Keyword arguments for the stream RNN.
    rnn_global_kwargs : dict, optional (default={})
        Keyword arguments for the global RNN.
    loss : str or nn.Module class - custom loss, optional (default='NLLLoss')
        Type of loss function for the classifier. If is a string, it must be a string exactly 
        equal to the name of a loss function in torch.nn module (e.g., 'MSELoss', 
        'CrossEntropyLoss', etc.), as you are importing the loss function. See `torch.nn` for 
        available loss functions. If is a custom loss, it must be a nn.Module class like NLLLoss.
    domain_loss : str or nn.Module class - custom loss, optional (default='NLLLoss')
        Type of loss function for the domain discriminator. If is a string, it must be a string 
        exactly equal to the name of a loss function in torch.nn module (e.g., 'MSELoss', 
        'CrossEntropyLoss', etc.), as you are importing the loss function. See `torch.nn` for 
        available loss functions. If is a custom loss, it must be a nn.Module class like NLLLoss.
    optimizer : str or optim.Optimizer class - custom optimizer, optional (default='SGD')
        Type of optimizer to use. If is a string, it must be a string exactly equal to the name 
        of an optimizer in torch.optim module (e.g., 'SGD', 'Adam', etc.), as you are importing 
        the optimizer function. See `torch.optim` for available optimizers. If is a custom 
        optimizer, it must be an optim.Optimizer class like SGD.
    lr : float, optional (default=0.003)
        Learning rate.
    epochs : int, optional (default=10)
        Number of epochs.
    batch_size : int, optional (default=200)
        Batch size for training.
    loss_kwargs : dict, optional (default={})
        Keyword arguments for the loss function.
    domain_loss_kwargs : dict, optional (default={})
        Keyword arguments for the domain loss function.
    optimizer_kwargs : dict, optional (default={'momentum': 0.9, 'weight_decay': 0.95})
        Keyword arguments for the optimizer.
    random_state : int, optional (default=42)
        Seed to ensure reproducibility.
    use_gpu : bool, optional (default=True)
        Whether to use GPU acceleration.
    verbose : bool, optional (default=True)
        Whether to print progress messages.

    Attributes
    ----------
    n_channels_ : int
        Number of channels in the EEG data.
    n_features_in_ : int
        Number of input features.
    n_features_per_ch_ : int
        Number of features per channel.
    n_classes_ : int
        Number of classes in the target.
    le_ : LabelEncoder
        LabelEncoder object.
    device_ : torch.device
        PyTorch device.
    lh_stream_ : list of int
        Indices of the channels in the left hemisphere horizontal stream.
    rh_stream_ : list of int
        Indices of the channels in the right hemisphere horizontal stream.
    lv_stream_ : list of int
        Indices of the channels in the left hemisphere vertical stream.
    rv_stream_ : list of int
        Indices of the channels in the right hemisphere vertical stream.
    optimizer_ : torch.optim.Optimizer
        Optimizer used for training.
    loss_fn_ : nn.Module
        Loss function used for training.
    classes_ : ndarray of shape (n_classes,)
        Unique classes in the target variable.
    domain_discrimination_ : bool
        Either to perform domain adversarial strategy or not, inferred from the inputs to
        the fit method.
    """

    def __init__(self, ch_names, lh_chs, rh_chs, lv_chs, rv_chs, 
                 d_stream=32, d_pair=32, d_global=32, d_out=16, 
                 k=6, a=0.01, pairwise_operation='subtraction', 
                 rnn_stream_kwargs={}, rnn_global_kwargs={}, 
                 loss='NLLLoss', domain_loss='NLLLoss', optimizer='SGD', lr=0.003,
                 epochs=10, batch_size=200, loss_kwargs={}, domain_loss_kwargs={}, 
                 optimizer_kwargs=dict(momentum=0.9, weight_decay=0.95),
                 random_state=42, use_gpu=True, verbose=True):

        self.ch_names = ch_names
        self.lh_chs = lh_chs
        self.rh_chs = rh_chs
        self.lv_chs = lv_chs
        self.rv_chs = rv_chs

        self.d_stream = d_stream
        self.d_pair = d_pair
        self.d_global = d_global
        self.d_out = d_out
        self.k = k
        self.a = a
        self.pairwise_operation = pairwise_operation

        self.rnn_stream_kwargs = rnn_stream_kwargs
        self.rnn_global_kwargs = rnn_global_kwargs

        self.loss = loss
        self.domain_loss = domain_loss
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_kwargs = loss_kwargs
        self.domain_loss_kwargs = domain_loss_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.random_state = random_state
        self.use_gpu = use_gpu
        self.verbose = verbose

        # selecting the indices for corresponding streams
        self.n_channels_ = len(ch_names)
        self.lh_stream_ = [list(ch_names).index(ch) for ch in lh_chs]
        self.rh_stream_ = [list(ch_names).index(ch) for ch in rh_chs]
        self.lv_stream_ = [list(ch_names).index(ch) for ch in lv_chs]
        self.rv_stream_ = [list(ch_names).index(ch) for ch in rv_chs]

        if torch.cuda.is_available() and use_gpu==True:
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device_ = torch.device(dev)

    def fit(self, X, y, X_test=None):
        '''
        Fit the BiHDMClassifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            X will be internally reshaped to (n_samples, n_channels, n_features_per_channel)
            by calling numpy.reshape using C-like index order.
        y : array-like of shape (n_samples,)
            Target variable.
        X_test : array-like of shape (n_test_samples, n_features), optional (default=None)
            The test data for performing the domain adversarial strategy. If not provided, the 
            classifier will be trained only with the class discriminator but not the domain 
            discriminator. X_test will be internally reshaped to (n_test_samples, n_channels, 
            n_features_per_channel) by calling numpy.reshape using C-like index order.

        Returns
        -------
        self : BiHDMClassifier
            The trained classifier.
        '''
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        if X_test is not None:
            self.domain_discrimination_ = True
            X_test = check_array(X_test)
        else:
            self.domain_discrimination_ = False

        # Get dimensions of the input data
        self.n_features_in_ = X.shape[1]
        self.n_features_per_ch_ = int(self.n_features_in_/self.n_channels_)
        assert self.n_features_per_ch_ * self.n_channels_ == self.n_features_in_, \
            f"Number of features ({self.n_features_in_}) could not be equally divided by the channels ({self.n_channels_})."
        self.n_classes_ = np.unique(y).shape[0]

        if not self.random_state is None:
            torch.manual_seed(self.random_state)

        # Encode the target variable
        self.le_ = LabelEncoder()
        self.le_.fit(y)

        # Reshape X and y and cast into torch tensors for training the BiHDM module
        X_ = np.reshape(X, [X.shape[0], self.n_channels_, self.n_features_per_ch_], order='C')
        X_ = torch.as_tensor(X_, dtype=torch.float).to(self.device_)
        y_ = torch.as_tensor(self.le_.transform(y), dtype=torch.int64).to(self.device_)
        if self.domain_discrimination_:
            X_test_ = np.reshape(X_test, [X_test.shape[0], self.n_channels_, self.n_features_per_ch_], order='C')
            X_test_ = torch.as_tensor(X_test_, dtype=torch.float).to(self.device_)
            n_tests = X_test_.size()[0]

        # Construct BiHDM
        self.bihdm_ = BiHDM(self.lh_stream_, self.rh_stream_, self.lv_stream_, self.rv_stream_, 
                            n_classes=self.n_classes_, d_input=self.n_features_per_ch_, 
                            d_stream=self.d_stream, d_pair=self.d_pair, 
                            d_global=self.d_global, d_out=self.d_out, k=self.k, a=self.a, 
                            pairwise_operation=self.pairwise_operation, output_domain=self.domain_discrimination_,
                            rnn_stream_kwargs=self.rnn_stream_kwargs, 
                            rnn_global_kwargs=self.rnn_global_kwargs)
        self.bihdm_.to(self.device_)
        
        # Setup training steps
        dataset = TensorDataset(X_, y_)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        loss_fn = getattr(import_module('torch.nn'), self.loss)(**self.loss_kwargs) \
            if type(self.loss) == str else self.loss(**self.loss_kwargs)
        loss_domain = getattr(import_module('torch.nn'), self.domain_loss)(**self.domain_loss_kwargs) \
            if type(self.domain_loss) == str else self.domain_loss(**self.domain_loss_kwargs)
        optimizer = getattr(import_module('torch.optim'), self.optimizer) if type(self.optimizer) == str else self.optimizer
        optimizer = optimizer(self.bihdm_.parameters(), lr=self.lr, **self.optimizer_kwargs)

        # Iterate through epochs to train BiHDM
        self.bihdm_.train(True)
        for ep in range(self.epochs):
            running_loss = 0.
            
            for i, (batch, labels) in enumerate(loader):
                optimizer.zero_grad()
                if self.domain_discrimination_:
                    outputs, domains_train = self.bihdm_.forward(batch)
                    _, domains_test = self.bihdm_.forward(X_test_)
                    domains = torch.cat([domains_train, domains_test], dim=0)
                    n_trains = labels.size()[0]
                    domain_labels = torch.zeros(n_trains + n_tests, dtype=torch.int64).to(self.device_)
                    domain_labels[n_trains:] = 1
                    loss = loss_fn(outputs, labels) + loss_domain(domains, domain_labels)
                else:
                    outputs = self.bihdm_.forward(batch)
                    loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(loader)

            if self.verbose:
                print(f'Epoch {ep}: loss_train={avg_loss}')
        self.bihdm_.train(False)
        
        return self

    def predict(self, X):
        """Predict the class labels for the given input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        X = np.reshape(X, [X.shape[0], self.n_channels_, self.n_features_per_ch_], order='C')
        X = torch.as_tensor(X, dtype=torch.float).to(self.device_)

        pred = self.bihdm_(X)
        pred = pred[0] if self.domain_discrimination_ else pred
        return self.le_.inverse_transform(torch.argmax(pred, dim=-1).to('cpu').detach().numpy())

    def predict_proba(self, X):
        """Predict class probabilities for the given input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        X = np.reshape(X, [X.shape[0], self.n_channels_, self.n_features_per_ch_], order='C')
        X = torch.as_tensor(X, dtype=torch.float).to(self.device_)

        pred = self.bihdm_(X)
        pred = pred[0] if self.domain_discrimination_ else pred
        return torch.exp(pred).to('cpu').detach().numpy()
    