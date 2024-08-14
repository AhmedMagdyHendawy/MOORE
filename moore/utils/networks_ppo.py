import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import moore.utils.mixture_layers as mixture_layers

class MiniGridPPONetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features, 
                       **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        n_input_channels = self._n_input[-1]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.Tanh(), 
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros((1, 3, 7, 7)).float()).shape[1]

        input_size = n_flatten
        
        self._output_head = nn.Sequential()

        if len(n_features) > 0:
            self._output_head.append(nn.Linear(input_size, n_features[0]))
            self._output_head.append(nn.Tanh())

            input_size = n_features[0]

        self._output_head.append(nn.Linear(input_size, self._n_output))

    def forward(self, state):
        f = self._output_head(self.cnn(state.float()))

        return f

####################################################################################################################################
class MiniGridPPOSHNetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       n_contexts = 1,
                       use_cuda = False, 
                       **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        self._n_contexts = n_contexts

        self._use_cuda = use_cuda

        n_input_channels = self._n_input[-1]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.Tanh(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros((1, 3, 7, 7)).float()).shape[1]

        input_size = n_flatten + n_contexts

        self._output_head = nn.Sequential()

        if len(n_features) > 0:
            self._output_head.append(nn.Linear(input_size, n_features[0]))
            self._output_head.append(nn.Tanh())

            input_size = n_features[0]

        self._output_head.append(nn.Linear(input_size, self._n_output))

    def forward(self, state, c = None):

        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        c = F.one_hot(c, num_classes = self._n_contexts)

        if self._use_cuda:
            c = c.cuda()
        
        features_cnn = self.cnn(state.float())
        f = self._output_head(torch.cat((features_cnn, c.float()), dim=1))
            
        return f
#######################################################################################################################################
class MiniGridPPOMHNetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       n_contexts = 1,
                       use_cuda = False, 
                       **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        self._n_contexts = n_contexts

        self._use_cuda = use_cuda

        n_input_channels = self._n_input[-1]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.Tanh(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros((1, 3, 7, 7)).float()).shape[1]


        self._output_heads = nn.ModuleList([])

        for _ in range(self._n_contexts):

            input_size = n_flatten
            
            head = nn.Sequential()

            if len(n_features) > 0:
                head.append(nn.Linear(input_size, n_features[0]))
                head.append(nn.Tanh())

                input_size = n_features[0]

            head.append(nn.Linear(input_size, self._n_output))

            self._output_heads.append(head)

    def forward(self, state, c = None):

        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        if self._use_cuda:
            c = c.cuda()
        
        features_cnn = self.cnn(state.float())

        f = torch.zeros(size=(state.shape[0], self._n_output))
        
        if self._use_cuda:
            f = f.cuda()

        for ci in torch.unique(c):
            ci_idx = torch.argwhere(c == ci).ravel()
            fi = self._output_heads[ci](features_cnn[ci_idx, :])
            f[ci_idx] = fi

        return f
#####################################################################################
class MiniGridPPOMixtureMHNetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       n_contexts = 1,
                       n_experts = 4,
                       orthogonal = True,
                       use_cuda = False,
                       task_encoder_bias = False,
                       **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        self._n_contexts = n_contexts
        self._orthogonal = orthogonal
        self._use_cuda = use_cuda

        n_input_channels = self._n_input[-1]

        
        self._task_encoder = nn.Linear(n_contexts, n_experts, bias = task_encoder_bias)

        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = cnn(torch.zeros((1, 3, 7, 7)).float()).shape[1]

        if orthogonal:
            self.cnn = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                     mixture_layers.ParallelLayer(cnn),
                                     mixture_layers.OrthogonalLayer1D())
        else:
            self.cnn = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                        mixture_layers.ParallelLayer(cnn))
    
        self._output_heads = nn.ModuleList([])

        for _ in range(self._n_contexts):

            input_size = n_flatten
            
            head = nn.Sequential()

            if len(n_features) > 0:
                head.append(nn.Linear(input_size, n_features[0]))
                head.append(nn.Tanh())

                input_size = n_features[0]

            head.append(nn.Linear(input_size, self._n_output))

            self._output_heads.append(head)

    def forward(self, state, c = None):

        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        if self._use_cuda:
            c = c.cuda()

        c_onehot = F.one_hot(c, num_classes = self._n_contexts)

        # task-weight and task-embeddings
        w = self._task_encoder(c_onehot.float()).unsqueeze(1)

        # image embeddings
        features_cnn = self.cnn(state.float())
        features_cnn = torch.permute(features_cnn, (1,0,2))

        # task-image embeddings
        features_cnn = w@features_cnn
        features_cnn = features_cnn.squeeze(1)

        # only activation after weighting the features 
        features_cnn = torch.tanh(features_cnn)

        f = torch.zeros(size=(state.shape[0], self._n_output))
        
        if self._use_cuda:
            f = f.cuda()

        for ci in torch.unique(c):
            ci_idx = torch.argwhere(c == ci).ravel()
            fi = self._output_heads[ci](features_cnn[ci_idx, :])
            f[ci_idx] = fi

        return f
    
    def compute_features(self, state):
        feat = self.cnn(state.float()).detach()

        return torch.permute(feat, (1,0,2)).cpu().numpy()

    def save_shared_backbone(self, save_dir):
        torch.save(self.cnn.state_dict(), save_dir)
    
    def load_shared_backbone(self, load_dir):
        self.cnn.load_state_dict(torch.load(load_dir))

        for param in self.cnn.parameters():
            param.requires_grad = False

    def save_task_encoder(self, save_dir):
        torch.save(self._task_encoder.state_dict(), save_dir) 
###################################################################################################################################################
class MiniGridPPOMixtureSHNetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       n_contexts = 1,
                       n_experts = 4,
                       orthogonal = True,
                       use_cuda = False,
                       task_encoder_bias = False,
                       **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]
        self._n_contexts = n_contexts
        self._orthogonal = orthogonal
        self._use_cuda = use_cuda

        n_input_channels = self._n_input[-1]

        self._task_encoder = nn.Linear(n_contexts, n_experts, bias = task_encoder_bias)

        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = cnn(torch.zeros((1, 3, 7, 7)).float()).shape[1]

        if orthogonal:
            self.cnn = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                     mixture_layers.ParallelLayer(cnn),
                                     mixture_layers.OrthogonalLayer1D())
        else:
            self.cnn = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                     mixture_layers.ParallelLayer(cnn))
        
        input_size = n_flatten + n_contexts

        self._output_head = nn.Sequential()

        if len(n_features) > 0:
            self._output_head.append(nn.Linear(input_size, n_features[0]))
            self._output_head.append(nn.Tanh())

            input_size = n_features[0]

        self._output_head.append(nn.Linear(input_size, self._n_output))


    def forward(self, state, c = None):

        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        c = F.one_hot(c, num_classes = self._n_contexts)

        if self._use_cuda:
            c = c.cuda()

        # task-weight and task-embeddings
        w = self._task_encoder(c.float()).unsqueeze(1)
        
        # image embeddings
        features_cnn = self.cnn(state.float())
        features_cnn = torch.permute(features_cnn, (1,0,2))

        # task-image embeddings
        features_cnn = w@features_cnn
        features_cnn = features_cnn.squeeze(1)

        # only activation after weigthing the features
        features_cnn = torch.tanh(features_cnn)

        # output with one-hot vectors or with task-embeddings
        f = self._output_head(torch.cat((features_cnn, c.float()), dim=1))

        return f

    def save_shared_backbone(self, save_dir):
        torch.save(self.cnn.state_dict(), save_dir)
    
    def load_shared_backbone(self, load_dir):
        self.cnn.load_state_dict(torch.load(load_dir))

        for param in self.cnn.parameters():
            param.requires_grad = False

    def save_task_encoder(self, save_dir):
        torch.save(self._task_encoder.state_dict(), save_dir)