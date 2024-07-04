import dataclasses

import jax
import jax.numpy as np
import flax.linen as nn

from models.transformer import Transformer
from models.mlp import MLP

from models.diffusion_utils import get_timestep_embedding,get_sinusoidal_embedding

from functools import partial


class classcondTransformerScoreNet(nn.Module):
    """Transformer score network."""

    d_t_embedding: int = 32
    d_wave_embedding: int = 64
    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
            "concat_conditioning": False,
        }
    )
    num_classes: int = None # number of classes for conditioning, if conditing
    adanorm: bool = False

    @nn.compact
    def __call__(self, flux, t, wavelength , conditioning, mask):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(flux.shape[0])  # Ensure t is a vector

        t_embedding = get_timestep_embedding(t, self.d_t_embedding)    
        t_embedding = nn.gelu(nn.Dense(self.score_dict["d_model"])(t_embedding))
        t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)
        #t_embedding = np.sin( nn.Dense(self.score_dict["d_model"])(t[:,None]))
        #breakpoint()
        if wavelength is None:
            wavelength_embd = 0.0
        else:
            #wavelength_embd = np.sin(nn.Dense(self.score_dict["d_model"])(wavelength))
            
            # sinusoidal -- MLP, follow the time embedding from DiT paper
            wavelength_embd = get_sinusoidal_embedding(wavelength, self.d_wave_embedding)
            wavelength_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(wavelength_embd))
            wavelength_embd = nn.Dense(self.score_dict["d_model"])(wavelength_embd)

            

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
        #breakpoint()
        h = Transformer(n_input=flux.shape[-1], **score_dict)(flux, cond, mask)

        return flux + h


class classtimecondTransformerScoreNet(nn.Module):
    """Transformer score network."""

    d_t_embedding: int = 32
    d_wave_embedding: int = 64
    d_spectime_embedding: int = 64

    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
            "concat_conditioning": False,
        }
    )
    num_classes: int = None # number of classes for conditioning, if conditing
    adanorm: bool = False

    @nn.compact
    def __call__(self, flux, t, wavelength, spectime, conditioning, mask):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(flux.shape[0])  # Ensure t is a vector


        t_embedding = get_timestep_embedding(t, self.d_t_embedding)    
        t_embedding = nn.gelu(nn.Dense(self.score_dict["d_model"])(t_embedding))
        t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)
        #t_embedding = np.sin( nn.Dense(self.score_dict["d_model"])(t[:,None]))
        #breakpoint()
        if wavelength is None:
            wavelength_embd = 0.0
        else:
            #wavelength_embd = np.sin(nn.Dense(self.score_dict["d_model"])(wavelength))
            
            # sinusoidal -- MLP, follow the time embedding from DiT paper
            wavelength_embd = get_sinusoidal_embedding(wavelength, self.d_wave_embedding)
            wavelength_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(wavelength_embd))
            wavelength_embd = nn.Dense(self.score_dict["d_model"])(wavelength_embd)

        if spectime is None:
            spectime_embd = 0.0
        else:
            spectime_embd = get_timestep_embedding(spectime, self.d_spectime_embedding)
            spectime_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(spectime_embd))
            spectime_embd = nn.Dense(self.score_dict["d_model"])(spectime_embd)

            

        if conditioning is None:
            cond = t_embedding[:, None, :] + wavelength_embd + spectime_embd[:, None, :]
        elif self.num_classes > 1:
            conditioning = nn.Embed(self.num_classes,self.score_dict["d_model"])(conditioning)
            cond = t_embedding[:, None, :] + wavelength_embd + spectime_embd[:, None, :] + conditioning[:, None, :]
        else:
            raise ValueError(f"there are {self.num_classes} classes, but num_classes must be > 1")

        # Make copy of score dict since original cannot be in-place modified; remove `score` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("score", None)
        #breakpoint()
        h = Transformer(n_input=flux.shape[-1], **score_dict)(flux, cond, mask)

        return flux + h



class photometrycondTransformerScoreNet(nn.Module):
    """Transformer score network."""

    d_t_embedding: int = 32
    d_wave_embedding: int = 64
    d_spectime_embedding: int = 64
    d_photometrictime_embedding: int = 64
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
    def __call__(self, flux, t, wavelength , spectime, mask, 
                 photometric_flux, photometric_time , photometric_wavelength, 
                 photometric_mask): 
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(flux.shape[0])  # Ensure t is a vector

        t_embedding = get_timestep_embedding(t, self.d_t_embedding)    
        t_embedding = nn.gelu(nn.Dense(self.score_dict["d_model"])(t_embedding))
        t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)
        #t_embedding = np.sin( nn.Dense(self.score_dict["d_model"])(t[:,None]))
        #breakpoint()


        # time at spectrum taken, embeded differently from photometry since they might have different length scale
        if spectime is None:
            spectime_embd = 0.0
        else:
            spectime_embd = get_timestep_embedding(spectime, self.d_spectime_embedding)
            spectime_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(spectime_embd))
            spectime_embd = nn.Dense(self.score_dict["d_model"])(spectime_embd)
        
            
        # sinusoidal -- MLP, follow the time embedding from DiT paper, embeded the same for photometry and spectrum
        wave_mlp = MLP([self.score_dict['d_model'], self.score_dict['d_model']], activation=nn.gelu)

        wavelength_embd = get_sinusoidal_embedding(wavelength, self.d_wave_embedding)
        wavelength_embd = wave_mlp(wavelength_embd)

        # Make copy of score dict since original cannot be in-place modified; remove `score` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("score", None)

        if photometric_flux is None:
            cond = t_embedding[:, None, :] + wavelength_embd + spectime_embd[:, None, :]
        else:
            # transformer for green and red channels
            photometric_time_embd = get_sinusoidal_embedding(photometric_time, self.d_photometrictime_embedding)
            photometric_time_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(photometric_time_embd))
            photometric_time_embd = nn.Dense(self.score_dict["d_model"])(photometric_time_embd)
            
            photometric_wavelength_embd = get_sinusoidal_embedding(photometric_wavelength, self.d_wave_embedding)
            photometric_wavelength_embd = wave_mlp(photometric_wavelength_embd)

            photometric_cond = photometric_time_embd + photometric_wavelength_embd
            
            #breakpoint()
            photometric_embd = Transformer(n_input=photometric_flux.shape[-1], **score_dict)(photometric_flux, photometric_cond, photometric_mask)
            photometric_embd = np.reshape(photometric_embd, (photometric_embd.shape[0], -1)) # flatten
            
            conditioning = nn.gelu(nn.Dense(self.score_dict["d_model"])(photometric_embd))
            conditioning = nn.Dense(self.score_dict["d_model"])(conditioning)
            cond = t_embedding[:, None, :] + wavelength_embd + spectime_embd[:, None, :] + conditioning[:, None, :]

        

        h = Transformer(n_input=flux.shape[-1], **score_dict)(flux, cond, mask)

        return flux + h

