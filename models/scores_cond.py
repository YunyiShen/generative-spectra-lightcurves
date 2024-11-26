import dataclasses

import jax
import jax.numpy as np
import flax.linen as nn

from models.transformer import Transformer, TransformerWavelength
from models.mlp import MLP

from models.diffusion_utils import get_timestep_embedding,get_sinusoidal_embedding

from functools import partial

class classtimecondTransformerScoreNet2(nn.Module):
    """Transformer score network using cross attentions."""

    d_t_embedding: int = 64
    d_spectime_embedding: int = 64
    d_wave_embedding: int = 64

    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
            "concat_wavelength": True,
        }
    )
    num_classes: int = None # number of classes for conditioning, if conditing
    adanorm: bool = False

    @nn.compact
    def __call__(self, flux, t, wavelength, spectime, conditioning, mask):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(flux.shape[0])  # Ensure t is a vector


        t_embedding = get_timestep_embedding(t, self.d_t_embedding)
        #skip = t_embedding    
        t_embedding = nn.gelu(nn.Dense(self.score_dict["d_model"])(t_embedding))
        #t_embedding = nn.gelu(nn.Dense(self.score_dict["d_model"])(t_embedding))
        t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)
        t_embedding = t_embedding #+ skip
        #t_embedding = np.sin( nn.Dense(self.score_dict["d_model"])(t[:,None]))
        #breakpoint()
        
            
        # sinusoidal -- MLP, follow the time embedding from DiT paper
        wavelength_embd = get_sinusoidal_embedding(wavelength, self.d_wave_embedding)
        wavelength_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(wavelength_embd))
        wavelength_embd = nn.Dense(self.score_dict["d_model"])(wavelength_embd)
         

        
        #breakpoint()
        spectime_embd = get_timestep_embedding(spectime, self.d_spectime_embedding)
        spectime_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(spectime_embd))
        spectime_embd = nn.Dense(self.score_dict["d_model"])(spectime_embd)
         

        if conditioning is None:
            cond = np.concatenate([t_embedding[:, None, :], spectime_embd[:,None,:]], axis=1)
        elif self.num_classes > 1:
            conditioning = nn.Embed(self.num_classes,self.score_dict["d_model"])(conditioning)
            
            cond = np.concatenate([t_embedding[:,None,:],
                                   spectime_embd[:,None,:], 
                                   conditioning[:,None,:]], axis=1)
            cond = nn.Dense(self.score_dict["d_model"])(cond)
            cond = nn.gelu(cond)
            cond = nn.Dense(self.score_dict["d_model"])(cond)
        else:
            raise ValueError(f"there are {self.num_classes} classes, but num_classes must be > 1")

        # Make copy of score dict since original cannot be in-place modified; remove `score` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("score", None)
        #breakpoint()
        h = TransformerWavelength(n_input=flux.shape[-1], **score_dict)(flux, wavelength_embd, cond, mask)

        return flux + h



############# a cross attention version for photometry conditioning #######
class photometrycondTransformerScoreNet2(nn.Module):
    """Transformer score network."""

    d_t_embedding: int = 64
    d_wave_embedding: int = 64
    d_spectime_embedding: int = 64
    d_photometrictime_embedding: int = 64
    nbands: int = 3
    concat_photo: bool = True
    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 6,
            "n_heads": 4,
            "concat_wavelength": True,
        }
    )
    adanorm: bool = False

    @nn.compact
    def __call__(self, flux, t, wavelength , spectime, mask, 
                 photometric_flux, photometric_time , photometric_wavelength, 
                 photometric_mask): 
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        #assert self.transformer_dict_photometry['d_model'] == self.score_dict['d_model']
        t = t * np.ones(flux.shape[0])  # Ensure t is a vector

        t_embedding = get_timestep_embedding(t, self.d_t_embedding)    
        t_embedding = nn.gelu(nn.Dense(self.score_dict["d_model"])(t_embedding))
        t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)
        #t_embedding = np.sin( nn.Dense(self.score_dict["d_model"])(t[:,None]))
        #breakpoint()


        # time at spectrum taken, embeded differently from photometry since they might have different length scale
        
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
        
        
        photometric_time_embd = get_sinusoidal_embedding(photometric_time, self.d_photometrictime_embedding)
        photometric_time_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(photometric_time_embd))
        photometric_time_embd = nn.Dense(self.score_dict["d_model"])(photometric_time_embd)
        
        photometric_flux_embd = photometric_flux
        photometric_flux_embd = nn.Dense(self.score_dict["d_model"])(photometric_flux_embd)
        
        #photometric_wavelength_embd = get_sinusoidal_embedding(photometric_wavelength, self.d_wave_embedding)
        #photometric_wavelength_embd = wave_mlp(photometric_wavelength_embd)
        photometric_wavelength_embd = nn.Embed(self.nbands,self.score_dict["d_model"])(photometric_wavelength)
        #breakpoint()
        
        # adding together
        #photometric_cond = photometric_wavelength_embd + photometric_flux_embd + photometric_time_embd
        
        #breakpoint()
        
        # concat then MLP
        if self.concat_photo:
            photometric_cond = np.concatenate([photometric_time_embd, 
                                           photometric_wavelength_embd, 
                                           photometric_flux_embd], axis=-1)
            photometric_cond =  nn.Dense(self.score_dict["d_model"])(photometric_cond)
            photometric_cond = nn.gelu(photometric_cond)
            photometric_cond = nn.Dense(self.score_dict["d_model"])(photometric_cond)
        else:
            photometric_cond = photometric_flux_embd + photometric_time_embd + photometric_wavelength_embd
        
            #direct concat
        cond = np.concatenate([t_embedding[:,None,:],
                                spectime_embd[:,None,:], 
                                photometric_cond], axis=1)
        #cond = nn.Dense(self.score_dict["d_model"])(cond)
        #cond = nn.gelu(cond)
        #cond = nn.Dense(self.score_dict["d_model"])(cond)
        '''
        # transformer for photometry
        score_dict2 = dict(self.score_dict)
        score_dict2.pop("score", None)
        score_dict2.pop("concat_wavelength", None)
        photometric_cond = Transformer(n_input=photometric_cond.shape[-1], 
                                       **score_dict2)(photometric_cond, mask = photometric_mask)
        photometric_cond = np.sum(photometric_cond * photometric_mask[:,:,None], axis=1)/np.sum(photometric_mask[:,:,None], axis = 1) # average over the photometric time

        cond = np.concatenate([t_embedding[:,None,:],
                                spectime_embd[:,None,:], 
                                photometric_cond[:,None,:]], axis=1)
           
        '''
        

        score_dict = dict(self.score_dict)
        score_dict.pop("score", None)
        
        condmask = np.pad(photometric_mask, ((0,0),(2,0)), constant_values = True) # pad for (diffusion) time and phase
        h = TransformerWavelength(n_input=flux.shape[-1], **score_dict)(flux, wavelength_embd, cond, mask, condmask)
        #h = TransformerWavelength(n_input=flux.shape[-1], **score_dict)(flux, wavelength_embd, cond, mask)

        return flux + h

######## think about continuous knobs #######

class knobcondTransformerScoreNet(nn.Module):
    """Transformer score network using cross attentions."""

    d_t_embedding: int = 64
    d_spectime_embedding: int = 64
    d_wave_embedding: int = 64
    d_knob_embedding: int = 64

    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
            "concat_wavelength": True,
        }
    )
    adanorm: bool = False

    @nn.compact
    def __call__(self, flux, t, wavelength, spectime, conditioning, mask):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(flux.shape[0])  # Ensure t is a vector
        n_knobs = conditioning.shape[-1]


        t_embedding = get_timestep_embedding(t, self.d_t_embedding)
        #skip = t_embedding    
        t_embedding = nn.gelu(nn.Dense(self.score_dict["d_model"])(t_embedding))
        #t_embedding = nn.gelu(nn.Dense(self.score_dict["d_model"])(t_embedding))
        t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)
        t_embedding = t_embedding #+ skip
        #t_embedding = np.sin( nn.Dense(self.score_dict["d_model"])(t[:,None]))
        #breakpoint()
        
            
        # sinusoidal -- MLP, follow the time embedding from DiT paper
        wavelength_embd = get_sinusoidal_embedding(wavelength, self.d_wave_embedding)
        wavelength_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(wavelength_embd))
        wavelength_embd = nn.Dense(self.score_dict["d_model"])(wavelength_embd)
         

        
        #breakpoint()
        spectime_embd = get_timestep_embedding(spectime, self.d_spectime_embedding)
        spectime_embd = nn.gelu(nn.Dense(self.score_dict["d_model"])(spectime_embd))
        spectime_embd = nn.Dense(self.score_dict["d_model"])(spectime_embd)
         

        if conditioning is None:
            cond = np.concatenate([t_embedding[:, None, :], spectime_embd[:,None,:]], axis=1)
        else:
            conditioning = get_sinusoidal_embedding(conditioning[:,:,None], self.d_knob_embedding)
            conditioning = nn.gelu(nn.Dense(self.score_dict["d_model"])(conditioning))
            conditioning = nn.Dense(self.score_dict["d_model"])(conditioning)
            cond = np.concatenate([t_embedding[:,None,:],
                                   spectime_embd[:,None,:], 
                                   conditioning], axis=1)
            cond = nn.Dense(self.score_dict["d_model"])(cond)
            cond = nn.gelu(cond)
            cond = nn.Dense(self.score_dict["d_model"])(cond)
        

        # Make copy of score dict since original cannot be in-place modified; remove `score` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("score", None)
        #breakpoint()
        h = TransformerWavelength(n_input=flux.shape[-1], **score_dict)(flux, wavelength_embd, cond, mask)

        return flux + h