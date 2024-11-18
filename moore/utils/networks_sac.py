import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from mushroom_rl.utils.torch import get_weights, set_weights

import moore.utils.mixture_layers as mixture_layers

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MetaworldSACMixtureMHCriticNetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       activation = 'ReLU', 
                       n_head_features = [],
                       n_contexts = 1, 
                       subspace = None, 
                       orthogonal = False, 
                       n_experts = 4, 
                       agg_activation = ['ReLU', 'ReLU'], 
                       use_cuda = True, 
                       **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        self._use_cuda = use_cuda
        self._subspace = subspace
        self._n_contexts = n_contexts

        n_layers = len(n_features) #handle if the list is empty
        n_head_layers = len(n_head_features)

        self._task_encoder = nn.Linear(n_contexts, n_experts, bias = False)
        nn.init.xavier_uniform_(self._task_encoder.weight,
                                    gain=nn.init.calculate_gain('linear'))
        
        self._agg_activation = agg_activation


        self._h = nn.Sequential()
        
        input_size = self._n_input[0]

        if n_layers > 1:
            for i in range(0, n_layers):
                if i == n_layers - 1:
                    activation_fn = None
                    if not activation.lower() == "linear":
                        activation_fn = getattr(nn, activation)()

                    _activation = activation.lower()
                else:
                    activation_fn = nn.ReLU()
                    _activation = "relu"

                layer = nn.Linear(input_size, n_features[i])
                nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain(_activation))
                self._h.add_module(f"backbone_layer_{i}", layer)
                if activation_fn is not None:
                    self._h.add_module(f"act_{i}", activation_fn)
                
                input_size = n_features[i]


        if orthogonal:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    mixture_layers.OrthogonalLayer1D(),
                                    )
        else:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    )
    

        self._output_heads = nn.ModuleList()
        for c in range(n_contexts):
            head = nn.Sequential()

            input_size = n_features[-1]

            if n_head_layers > 0:
                for i in range(0, n_head_layers):
                    layer = nn.Linear(input_size, n_head_features[i])
                    nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('relu'))
                    head.add_module(f"head_{c}_layer_{i}",layer)

                    head.add_module(f"head_{c}_act_{i}",nn.ReLU())

                    input_size = n_head_features[i]
            
            layer = nn.Linear(input_size, self._n_output)
            nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain('linear'))
            head.add_module(f"head_{c}_out",layer)
            
            self._output_heads.append(head)

    def get_shared_weights_t(self):
        weights = []

        for l in self._h:
            if isinstance(l, nn.Linear):
                weights.append(l.weight)
                
        return weights
    
    def get_shared_weights(self):
        return [w.detach().cpu().numpy() for w in self.get_shared_weights_t()]

    def forward(self, state, action=None, c = None):
        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        if self._use_cuda:
            c = c.cuda()
        
        # task-weight computation
        c_onehot = F.one_hot(c, num_classes = self._n_contexts)
        w = self._task_encoder(c_onehot.float()).unsqueeze(1)

        state_action = torch.cat((state.float(), action.float()), dim=1)
        
        # shared features
        features = self._h(state_action)
        features  = torch.permute(features, (1,0,2))

        # activation before
        if not self._agg_activation[0].lower() == "linear":
            features = getattr(torch, self._agg_activation[0].lower())(features)

        # task-features
        features = w@features
        features = features.squeeze(1)

        # activation after
        if not self._agg_activation[1].lower() == "linear":
            features = getattr(torch, self._agg_activation[1].lower())(features)

        q = torch.zeros(size=(state.shape[0], self._n_output))
        
        if self._use_cuda:
            q = q.cuda()

        for i, ci in enumerate(torch.unique(c)):
            ci_idx = torch.argwhere(c == ci).ravel()
            qi = self._output_heads[ci](features[ci_idx, :])
            q[ci_idx] = qi
            
        return torch.squeeze(q)

#######
#Actor#
#######      
class MetaworldSACMixtureMHActorNetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       activation = 'ReLU', 
                       n_head_features = [],
                       shared_mu_sigma = False, 
                       n_contexts = 1,
                       subspace = None, 
                       orthogonal = False, 
                       n_experts = 4, 
                       agg_activation = ['ReLU', 'ReLU'], 
                       use_cuda = True, **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        if shared_mu_sigma:
            self._n_output*=2

        self._shared_mu_sigma = shared_mu_sigma

        self._use_cuda = use_cuda
        self._subspace = subspace
        self._n_contexts = n_contexts

        n_layers = len(n_features)
        n_head_layers = len(n_head_features)

        self._task_encoder = nn.Linear(n_contexts, n_experts, bias = False)
        nn.init.xavier_uniform_(self._task_encoder.weight,
                                    gain=nn.init.calculate_gain('linear'))
        
        self._agg_activation = agg_activation

        self._h = nn.Sequential()
        
        input_size = self._n_input[0]

        if n_layers > 1:
            for i in range(0, n_layers):
                if i == n_layers - 1:
                    activation_fn = None
                    if not activation.lower() == "linear":
                        activation_fn = getattr(nn, activation)()

                    _activation = activation.lower()
                else:
                    activation_fn = nn.ReLU()
                    _activation = "relu"

                layer = nn.Linear(input_size, n_features[i])
                nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain(_activation))
                self._h.add_module(f"backbone_layer_{i}", layer)
                if activation_fn is not None:
                    self._h.add_module(f"act_{i}", activation_fn)
                
                input_size = n_features[i]


        if orthogonal:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    mixture_layers.OrthogonalLayer1D(),
                                    )
        else:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    )
        
        self._output_heads = nn.ModuleList()
        for c in range(n_contexts):
            head = nn.Sequential()

            input_size = n_features[-1]

            if n_head_layers > 0:
                for i in range(0, n_head_layers):
                    layer = nn.Linear(input_size, n_head_features[i])
                    nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('relu'))
                    head.add_module(f"head_{c}_layer_{i}",layer)

                    head.add_module(f"head_{c}_act_{i}",nn.ReLU())

                    input_size = n_head_features[i]
            
            layer = nn.Linear(input_size, self._n_output)
            nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain('linear'))
            head.add_module(f"head_{c}_out",layer)
            
            self._output_heads.append(head)

    def get_shared_weights_t(self):
        weights = []

        for l in self._h:
            if isinstance(l, nn.Linear):
                weights.append(l.weight)
                
        return weights
    
    def get_shared_weights(self):
        return [w.detach().cpu().numpy() for w in self.get_shared_weights_t()]
    
    def forward(self, state, c = None):
        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        if self._use_cuda:
            c = c.cuda()
        
        # task-weight computation
        c_onehot = F.one_hot(c, num_classes = self._n_contexts)
        w = self._task_encoder(c_onehot.float()).unsqueeze(1)

        # shared features
        features = self._h(state.float())
        features  = torch.permute(features, (1,0,2))

        # activation before
        if not self._agg_activation[0].lower() == "linear":
            features = getattr(torch, self._agg_activation[0].lower())(features)

        # task-features
        features = w@features
        features = features.squeeze(1)

        # activation after
        if not self._agg_activation[1].lower() == "linear":
            features = getattr(torch, self._agg_activation[1].lower())(features)

        a = torch.zeros(size=(state.shape[0], self._n_output))
        
        if self._use_cuda:
            a = a.cuda()

        for i, ci in enumerate(torch.unique(c)):
            ci_idx = torch.argwhere(c == ci).ravel()
            ai = self._output_heads[ci](features[ci_idx, :])

            a[ci_idx] = ai

        return a