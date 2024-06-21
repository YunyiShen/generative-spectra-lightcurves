import dataclasses

import jax
import jax.numpy as np
import flax.linen as nn

from models.transformer import Transformer
from models.mlp import MLP

from models.diffusion_utils import get_timestep_embedding

from functools import partial


class classcondTransformerScoreNet(nn.Module):
    """Transformer score network."""

    d_t_embedding: int = 32
    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
        }
    )
    num_classes: int = None # number of classes for conditioning, if conditing
    adanorm: bool = False

    @nn.compact
    def __call__(self, flux, t, wavelength , conditioning, mask):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(flux.shape[0])  # Ensure t is a vector

        #t_embedding = get_timestep_embedding(t, self.d_t_embedding)    
        #t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)
        t_embedding = np.sin( nn.Dense(self.score_dict["d_model"])(t[:,None]))
        #breakpoint()
        if wavelength is None:
            wavelength_embd = 0.0
        else:
            wavelength_embd = np.sin(nn.Dense(self.score_dict["d_model"])(wavelength))
            #breakpoint()

        if conditioning is None:
            cond = t_embedding[:, None, :] + wavelength_embd
        elif self.num_classes > 1:
            conditioning = nn.Embed(self.num_classes,self.score_dict["d_model"])(conditioning)
            cond = t_embedding[:, None, :] + wavelength_embd + conditioning[:, None, :]
        else:
            raise ValueError(f"there are {self.num_classes} classes, but num_classes must be > 1")

        # Make copy of score dict since original cannot be in-place modified; remove `score` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("score", None)

        h = Transformer(n_input=flux.shape[-1], **score_dict)(flux, cond, mask)

        return flux + h




class photometrycondTransformerScoreNet(nn.Module):
    """Transformer score network."""

    d_t_embedding: int = 32
    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
        }
    )
    num_classes: int = None # number of classes for conditioning, if conditing
    adanorm: bool = False

    @nn.compact
    def __call__(self, flux, t, wavelength , mask, 
                 green_flux, green_time, green_mask, 
                 red_flux, red_time, red_mask):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(flux.shape[0])  # Ensure t is a vector

        t_embedding = get_timestep_embedding(t, self.d_t_embedding)
        t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)
        wavelength_embd = nn.Dense(self.score_dict["d_model"])(wavelength) # embedding wavelengthuency

        # Make copy of score dict since original cannot be in-place modified; remove `score` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("score", None)

        if green_time is None or red_time is None:
            cond = t_embedding[:, None, :] + wavelength_embd
        else:
            # transformer for green and red channels
            green_embd = Transformer(n_input=green_flux.shape[-1], **score_dict)(green_flux, green_time, green_mask)
            green_embd = np.reshape(green_embd, (green_embd.shape[0], -1)) # flatten
            red_embd = Transformer(n_input=red_flux.shape[-1], **score_dict)(red_flux, red_time, red_mask)
            red_embd = np.reshape(red_embd, (red_embd.shape[0], -1))
            conditioning = np.concatenate((green_embd, red_embd), axis=1)
            conditioning = nn.Dense(self.score_dict["d_model"])(conditioning)
            cond = t_embedding[:, None, :] + wavelength_embd + conditioning[:, None, :]

        

        h = Transformer(n_input=flux.shape[-1], **score_dict)(flux, cond, mask)

        return flux + h

