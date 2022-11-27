import math
import numpy as np
import textwrap
import copy
from collections import OrderedDict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.base_nets import Module, Sequential, MLP
from robomimic.models.obs_nets import ObservationGroupEncoder, ObservationDecoder
from robomimic.models.distributions import TanhWrappedDistribution

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:d_model//2]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(Module):

    def __init__(self, d_in: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model # Embedding of size d_in modified to d_model
        self.linearIn = nn.Linear(d_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            activation='relu',
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=True
        )
        self.norm_final = torch.nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=nlayers,
            norm=self.norm_final,
            enable_nested_tensor=True
        )

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size] (batch_first = True)
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            attended Tensor of shape [seq_len, batch_size]
        """
        src = self.linearIn(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output
    
    def output_shape(self, input_shape=None):
        # infers temporal dimension from input shape
        obs_group = list(self.input_obs_group_shapes.keys())[0] # The dictionary under 'obs'
        mod = list(self.input_obs_group_shapes[obs_group].keys())[0] # mod could be 'object'
        T = input_shape[obs_group][mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="TRANSFORMER_MIMO_MLP: input_shape inconsistent in temporal dimension")
        
        return [T, self.d_model] # e.g [T, 7]
        
class TRANSFORMER_MIMO_MLP(Module):
    """
    A wrapper class for a transformer and a per-step MLP (optional) and a decoder.

    Structure: [encoder -> transformer -> mlp -> decoder]

    All temporal inputs are processed by a shared @ObservationGroupEncoder,
    followed by an Transformer, and then a per-step multi-output MLP. 
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        mlp_layer_dims,
        mlp_activation=nn.ReLU,
        mlp_layer_func=nn.Linear,
        encoder_kwargs=None,
        device = None
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
                Example : {
                    'obs' : {
                        'object' : [10],
                        'robot0_eef_pos': [3],
                        'robot0_eef_quat': [4],
                        'robot0_gripper_qpos': [2]
                    },
                    'goal' : None
                }

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs. (usually the action dimension)
                Example : {
                    'action': (7,)
                }

            mlp_layer_dim (list(int)): eg. [1024, 1024]. Dimensions of the intermediate 
                and output layers of the MLP

            dim_feedforward (int): Dimension of the hidden layer of the feedforward network
                that follows the multi-head attention. Default is 2048. 

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(TRANSFORMER_MIMO_MLP, self).__init__()
        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)
        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes
        self.device = device
        self.history = None

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        """
        Example: 
            self.nets["encoder"] = 
                ObservationGroupEncoder(
                    group=obs
                    ObservationEncoder(
                        Key(
                            name=object
                            shape=[10]
                            modality=low_dim
                            randomizer=None
                            net=None
                            sharing_from=None
                        )
                        Key(
                            name=robot0_eef_pos
                            shape=[3]
                            modality=low_dim
                            randomizer=None
                            net=None
                            sharing_from=None
                        )
                        Key(
                            name=robot0_eef_quat
                            shape=[4]
                            modality=low_dim
                            randomizer=None
                            net=None
                            sharing_from=None
                        )
                        Key(
                            name=robot0_gripper_qpos
                            shape=[2]
                            modality=low_dim
                            randomizer=None
                            net=None
                            sharing_from=None
                        )
                        output_shape=[19]
                    )
                )
        """

        # flat encoder output dimension
        encoder_out_dim = self.nets["encoder"].output_shape()[0] # e.g 19

        # transformer output dimension
        config_trans_input_dim = 512
        trans_output_dim = config_trans_input_dim

        assert (len(mlp_layer_dims) > 0)

        self.nets["mlp"] = MLP(
            input_dim=trans_output_dim,
            output_dim=mlp_layer_dims[-1],
            layer_dims=mlp_layer_dims[:-1],
            output_activation=mlp_activation,
            layer_func=mlp_layer_func
        )
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=mlp_layer_dims[-1],
        )
        self.nets["action"] = Sequential(self.nets["mlp"], self.nets["decoder"])
        """
        Example:
            self.nets["decoder"] = 
                ObservationDecoder(
                    Key(
                        name=mean
                        shape=(5, 7)
                        modality=low_dim
                        net=(Linear(in_features=400, out_features=35, bias=True))
                    )
                    Key(
                        name=scale
                        shape=(5, 7)
                        modality=low_dim
                        net=(Linear(in_features=400, out_features=35, bias=True))
                    )
                    Key(
                        name=logits
                        shape=(5,)
                        modality=low_dim
                        net=(Linear(in_features=400, out_features=5, bias=True))
                    )
                )
        """
        # core network
        "Need to fill these into config"
        config_nhead = 8
        config_d_feedforward_hid = 1024
        config_nlayers = 24
        config_dropout = 0.5
        config_max_history_len = 500

        self.max_history_len = config_max_history_len
        self.reset_history()
        self.nets["trans"] = TransformerModel(encoder_out_dim,
            config_trans_input_dim, 
            config_nhead,
            config_d_feedforward_hid,
            config_nlayers,
            config_dropout)

    def output_shape(self, input_shape):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.

        Args:
            input_shape (dict): dictionary of dictionaries, where each top-level key
                corresponds to an observation group, and the low-level dictionaries
                specify the shape for each modality in an observation dictionary
        """

        # infers temporal dimension from input shape
        obs_group = list(self.input_obs_group_shapes.keys())[0] # The dictionary under 'obs'
        mod = list(self.input_obs_group_shapes[obs_group].keys())[0] # mod could be 'object'
        T = input_shape[obs_group][mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="TRANSFORMER_MIMO_MLP: input_shape inconsistent in temporal dimension")
        # returns a dictionary instead of list since outputs are dictionaries
        return { k : [T] + list(self.output_shapes[k]) for k in self.output_shapes } # {'action': [T, 7]}

    def forward(self, **inputs):
        """
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.
            Example : {
                    'obs' : {
                        'object' : [100, 10, 10], ([B, T, ..])
                        'robot0_eef_pos': [100, 10, 3],
                        'robot0_eef_quat': [100, 10, 4],
                        'robot0_gripper_qpos': [100, 10, 2]
                    },
                    'goal' : None
                }

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        # use encoder to extract flat transformer inputs
        trans_inputs = TensorUtils.time_distributed(inputs, self.nets["encoder"], inputs_as_kwargs=True)
        " Inputs to be fed into the Transformer network, Example: [100, 10, 19]"
        assert trans_inputs.ndim == 3  # [B, T, D]

        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        ntoken = (trans_inputs.shape)[1]
        src_mask = torch.triu(torch.ones(ntoken, ntoken) * float('-inf'), diagonal=1).to(self.device)

        outputs = self.nets["trans"].forward(trans_inputs, src_mask)
        outputs = TensorUtils.time_distributed(outputs, self.nets["action"])
        
        return outputs

    def reset_history(self):
        self.history = None

    def forward_step(self, obs_dict, goal_dict=None):
        """
        Unroll network over a single timestep.

        Args:
            inputs (dict): expects same modalities as @self.input_shapes, with
                additional batch dimension (but NOT time), since this is a 
                single time step.
                In this case it happens to include all modalities, even those not 
                used by the policy

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Does not contain time dimension.
        """
        if goal_dict is not None:
            raise NotImplementedError("Cannot handle goal conditioned inputs")
        
        key_list = list(self.input_obs_group_shapes['obs'].keys())
        # ensure that no extra dimension time is present
        assert np.all([obs_dict[k].ndim == 2 for k in key_list])
        # ensure that the batch size is one
        assert np.all([obs_dict[k].shape[0] == 1 for k in key_list])

        # add history, padding and batch size of 1
        if self.history is None:
            self.history = dict(zip(key_list, [None]*len(key_list)))
            for k in key_list:
                self.history[k] = torch.unsqueeze(obs_dict[k], dim=1).clone()
        else:
            history_len = self.history[key_list[0]].shape[1]
            if history_len < self.max_history_len :
                for k in key_list:
                    self.history[k] = torch.cat((self.history[k],
                                                 torch.unsqueeze(obs_dict[k], dim=1)), dim=1)
            elif history_len == self.max_history_len:
                for k in key_list:
                    self.history[k] = self.history[k][:,1:,:]
                    self.history[k] = torch.cat((self.history[k],
                                                 torch.unsqueeze(obs_dict[k], dim=1)), dim=1)
            else:
                raise ValueError("History lenght exceeds number of tokens")     
           
        inputs = copy.deepcopy(self.history)
        outputs = TRANSFORMER_MIMO_MLP.forward(self, obs = inputs, goal = None)
        for k in outputs:
            outputs[k] = outputs[k][:,-1,:]

        return outputs

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\n\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\ntransformer={}".format(self.nets["trans"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg

class TransformerActorNetwork(TRANSFORMER_MIMO_MLP):
    """
    An Transformer policy network that predicts actions from observations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        goal_shapes=None,
        encoder_kwargs=None,
        device = None
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.ac_dim = ac_dim

        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes

        # set up different observation groups for @TRANSFORMER_MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(TransformerActorNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            mlp_layer_dims=mlp_layer_dims,
            mlp_activation=nn.ReLU,
            mlp_layer_func=nn.Linear,
            encoder_kwargs=encoder_kwargs,
            device = device
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @TRANSFORMER_MIMO_MLP, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        return OrderedDict(action=(self.ac_dim,))

    def reset_history(self):
        return super(TransformerActorNetwork, self).reset_history()

    def output_shape(self, input_shape):
        # note: @input_shape should be dictionary (key: mod)
        # infers temporal dimension from input shape
        mod = list(self.obs_shapes.keys())[0]
        T = input_shape[mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="TransformerActorNetwork: input_shape inconsistent in temporal dimension")
        return [T, self.ac_dim]

    def forward(self, obs_dict, goal_dict=None):
        """
        Forward a sequence of inputs through the Transformer.

        Args:
            obs_dict (dict): batch of observations - each tensor in the dictionary
                should have leading dimensions batch and time [B, T, ...]
            goal_dict (dict): if not None, batch of goal observations
            return_state (bool): whether to return hidden state

        Returns:
            actions (torch.Tensor): predicted action sequence
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        outputs = super(TransformerActorNetwork, self).forward(
            obs=obs_dict, goal=goal_dict)

        actions = outputs
        
        # apply tanh squashing to ensure actions are in [-1, 1]
        actions = torch.tanh(actions["action"])
        return actions

    def forward_step(self, obs_dict, goal_dict=None):
        """
        Unroll Transformer over single timestep to get actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            actions (torch.Tensor): batch of actions - does not contain time dimension
        """
        outputs = super(TransformerActorNetwork, self).forward_step(
            obs_dict, goal_dict)

        # apply tanh squashing to ensure actions are in [-1, 1]
        actions = torch.tanh(outputs["action"])       
        return actions

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)

class TransformerGMMActorNetwork(TransformerActorNetwork):
    """
    An Transformer GMM policy network that predicts sequences of action distributions from observation sequences.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
        device=None,
    ):
        """
        Args:

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert std_activation in self.activations, \
            "std_activation must be one of: {}; instead got: {}".format(self.activations.keys(), std_activation)
        self.std_activation = std_activation

        super(TransformerGMMActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            goal_shapes=goal_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
            device = device,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.ac_dim), 
            scale=(self.num_modes, self.ac_dim), 
            logits=(self.num_modes,),
        )

    def forward_train(self, obs_dict, goal_dict=None, step=False):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL 
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        if step == False:
            outputs = TRANSFORMER_MIMO_MLP.forward(
                self, obs=obs_dict, goal=goal_dict)
        else:
            outputs = TRANSFORMER_MIMO_MLP.forward_step(
                self, obs=obs_dict, goal=goal_dict)
        
        means = outputs["mean"]
        scales = outputs["scale"]
        logits = outputs["logits"]

        # apply tanh squashing to mean if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, timesteps, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1) # shift action dim to event shape

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            dists = TanhWrappedDistribution(base_dist=dists, scale=1.)

        return dists

    def forward(self, obs_dict, goal_dict=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(obs_dict=obs_dict, goal_dict=goal_dict)

        return out.sample()

    def forward_train_step(self, obs_dict, goal_dict=None):
        """
        Unroll Transformer over single timestep to get action GMM distribution, which 
        is useful for computing quantities necessary at train-time, like 
        log-likelihood, KL divergence, etc.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            ad (Distribution): GMM action distributions
        """
        ad = self.forward_train(
            obs_dict, goal_dict, step = True)

        # to squeeze time dimension, make another action distribution
        assert ad.component_distribution.base_dist.loc.shape[1] == 1
        assert ad.component_distribution.base_dist.scale.shape[1] == 1
        assert ad.mixture_distribution.logits.shape[1] == 1
        component_distribution = D.Normal(
            loc=ad.component_distribution.base_dist.loc.squeeze(1),
            scale=ad.component_distribution.base_dist.scale.squeeze(1),
        )
        component_distribution = D.Independent(component_distribution, 1)
        mixture_distribution = D.Categorical(logits=ad.mixture_distribution.logits.squeeze(1))
        ad = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        return ad

    def forward_step(self, obs_dict, goal_dict=None):
        """
        Unroll Transformer over single timestep to get sampled actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            acts (torch.Tensor): batch of actions - does not contain time dimension
        """
        out = self.forward_train(obs_dict=obs_dict, goal_dict=goal_dict, step=True)

        return out.sample()

    def _to_string(self):
        """Info to pretty print."""
        msg = "action_dim={}, std_activation={}, low_noise_eval={}, num_nodes={}, min_std={}".format(
            self.ac_dim, self.std_activation, self.low_noise_eval, self.num_modes, self.min_std)
        return msg
