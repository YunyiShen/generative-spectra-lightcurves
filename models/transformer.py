import jax
import jax.numpy as np
from flax import linen as nn
from models.diffusion_utils import get_timestep_embedding,get_sinusoidal_embedding



class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention. Uses pre-LN configuration (LN within residual stream), which seems to work much better than post-LN."""

    n_heads: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, x, y, mask=None, conditioning=None):
        mask = None if mask is None else mask[..., None, :, :]

        # Multi-head attention
        if x is y:  # Self-attention
            x_sa = nn.LayerNorm()(x)  # pre-LN
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(x_sa, x_sa, mask=mask)
        else:  # Cross-attention
            x_sa, y_sa = nn.LayerNorm()(x), nn.LayerNorm()(y)
            #x_sa, y_sa = nn.LayerNorm()(x), y
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(y_sa, x_sa, mask=mask)
            #x_sa = nn.LayerNorm()(x_sa)
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(x_sa, y_sa, mask=mask)

            #x_sa, y_sa = nn.LayerNorm()(x), nn.LayerNorm()(y)
            #x_sa, y_sa = nn.LayerNorm()(x), y
            

        # Add into residual stream
        x += x_sa

        # MLP
        x_mlp = nn.LayerNorm()(x)  # pre-LN
        x_mlp = nn.gelu(nn.Dense(self.d_mlp)(x_mlp))
        x_mlp = nn.Dense(self.d_model)(x_mlp)

        # Add into residual stream
        x += x_mlp

        return x


class PoolingByMultiHeadAttention(nn.Module):
    """PMA block from the Set Transformer paper."""

    n_seed_vectors: int
    n_heads: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, z, mask=None):
        seed_vectors = self.param(
            "seed_vectors",
            nn.linear.default_embed_init,
            (self.n_seed_vectors, z.shape[-1]),
        )
        seed_vectors = np.broadcast_to(seed_vectors, z.shape[:-2] + seed_vectors.shape)
        mask = None if mask is None else mask[..., None, :]
        return MultiHeadAttentionBlock(
            n_heads=self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp
        )(seed_vectors, z, mask)


class Transformer(nn.Module):
    """Simple decoder-only transformer for set modeling.
    Attributes:
      n_input: The number of input (and output) features.
      d_model: The dimension of the model embedding space.
      d_mlp: The dimension of the multi-layer perceptron (MLP) used in the feed-forward network.
      n_layers: Number of transformer layers.
      n_heads: The number of attention heads.
      induced_attention: Whether to use induced attention.
      n_inducing_points: The number of inducing points for induced attention.
      concat_conditioning: Whether to concatenate conditioning to input.
      ada_norm: Whether to use AdaNorm (LayerNorm with bias parameters learned via conditioning context).
    """

    n_input: int
    d_model: int = 128
    d_mlp: int = 512
    n_layers: int = 4
    n_heads: int = 4
    induced_attention: bool = False
    n_inducing_points: int = 32
    concat_conditioning: bool = False

    @nn.compact
    def __call__(self, x: np.ndarray, conditioning: np.ndarray = None, mask=None):
        # Input embedding
        #x = get_sinusoidal_embedding(x, int(self.d_model))
        #skip = x
        x = nn.Dense(int(self.d_model))(x)  # (batch, seq_len, d_model)
        #x = nn.gelu(x)
        #x = nn.Dense(self.d_model)(x)
        #x = nn.gelu(x)
        #x = nn.Dense(self.d_model)(x)
        #x = x + skip
        if conditioning is not None:   
            if self.concat_conditioning:
                x = np.concatenate([x, conditioning], axis=-1)
        # Transformer layers
        for _ in range(self.n_layers):
            if conditioning is not None:
                if not self.concat_conditioning:
                    x += conditioning  # (batch, seq_len, d_model)
                #breakpoint()

            if not self.induced_attention:  # Vanilla self-attention
                mask_attn = (
                    None if mask is None else mask[..., None] * mask[..., None, :]
                )
                x = MultiHeadAttentionBlock(
                    n_heads=self.n_heads,
                    d_model= (1 + self.concat_conditioning) * self.d_model,
                    d_mlp=self.d_mlp,
                )(x, x, mask_attn, conditioning)
            else:  # Induced attention from set transformer paper
                h = PoolingByMultiHeadAttention(
                    self.n_inducing_points,
                    self.n_heads,
                    d_model=(1 + self.concat_conditioning) * self.d_model,
                    d_mlp=self.d_mlp,
                )(x, mask)
                mask_attn = None if mask is None else mask[..., None]
                x = MultiHeadAttentionBlock(
                    n_heads=self.n_heads, d_model=(1 + self.concat_conditioning) * self.d_model, 
                            d_mlp=self.d_mlp
                )(x, h, mask_attn)
        # Final LN as in pre-LN configuration
        x = nn.LayerNorm()(x)
        # Unembed; zero init kernel to propagate zero residual initially before training
        x = nn.Dense(self.n_input, kernel_init=jax.nn.initializers.zeros)(x)
        return x  #+ conditioning if conditioning is not None else x



class TransformerCrossattn(nn.Module):
    """Simple decoder-only transformer for set modeling.
    Attributes:
      n_input: The number of input (and output) features.
      d_model: The dimension of the model embedding space.
      d_mlp: The dimension of the multi-layer perceptron (MLP) used in the feed-forward network.
      n_layers: Number of transformer layers.
      n_heads: The number of attention heads.
      induced_attention: Whether to use induced attention.
      n_inducing_points: The number of inducing points for induced attention.
      concat_conditioning: Whether to concatenate conditioning to input.
      ada_norm: Whether to use AdaNorm (LayerNorm with bias parameters learned via conditioning context).
    """

    n_input: int
    d_model: int = 128
    d_mlp: int = 512
    n_layers: int = 4
    n_heads: int = 4

    @nn.compact
    def __call__(self, x: np.ndarray, conditioning: np.ndarray = None, mask=None):
        # Input embedding
        x = nn.Dense(int(self.d_model))(x)  # (batch, seq_len, d_model)
        # Transformer layers
        for _ in range(self.n_layers):
            # self attention 
            mask_attn = (
                    None if mask is None else mask[..., None] * mask[..., None, :]
                )
            
            x = MultiHeadAttentionBlock(
                    n_heads=self.n_heads,
                    d_model= self.d_model,
                    d_mlp=self.d_mlp,
                )(x, x, mask_attn, conditioning)# + skip
            if conditioning is not None:
                # cross attention
                
                x = MultiHeadAttentionBlock(
                    n_heads=self.n_heads,
                    d_model= self.d_model,
                    d_mlp=self.d_mlp,
                )(x, conditioning, mask_attn, conditioning) #+ skip
                
        # Final LN as in pre-LN configuration
        x = nn.LayerNorm()(x)
        # Unembed; zero init kernel to propagate zero residual initially before training
        x = nn.Dense(self.n_input)(x)
        return x  



###################### some more adatpted #########################
class MultiHeadAttentionBlock2(nn.Module):
    """Multi-head attention. Uses pre-LN configuration (LN within residual stream), which seems to work much better than post-LN."""

    n_heads: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, x, y, mask=None):
        mask = None if mask is None else mask[..., None, :, :]

        # Multi-head attention
        if x is y:  # Self-attention
            x_sa = nn.LayerNorm()(x)  # pre-LN
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(x_sa, x_sa, mask=mask)
        else:  # Cross-attention
            x_sa, y_sa = nn.LayerNorm()(x), nn.LayerNorm()(y)
            #breakpoint()
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(x_sa, y_sa, y_sa, mask=mask)
            

        # Add into residual stream
        x += x_sa

        # MLP
        x_mlp = nn.LayerNorm()(x)  # pre-LN
        x_mlp = nn.gelu(nn.Dense(self.d_mlp)(x_mlp))
        x_mlp = nn.Dense(self.d_model)(x_mlp)

        # Add into residual stream
        x += x_mlp

        return x

class TransformerWavelength(nn.Module):
    """Simple decoder-only transformer for set modeling.
    Attributes:
      n_input: The number of input (and output) features.
      d_model: The dimension of the model embedding space.
      d_mlp: The dimension of the multi-layer perceptron (MLP) used in the feed-forward network.
      n_layers: Number of transformer layers.
      n_heads: The number of attention heads.
      induced_attention: Whether to use induced attention.
      n_inducing_points: The number of inducing points for induced attention.
      concat_conditioning: Whether to concatenate conditioning to input.
      ada_norm: Whether to use AdaNorm (LayerNorm with bias parameters learned via conditioning context).
    """

    n_input: int
    d_model: int = 256
    d_mlp: int = 512
    n_layers: int = 4
    n_heads: int = 4
    concat_wavelength: bool = True

    @nn.compact
    def __call__(self, x: np.ndarray, 
                 wavelengthembd: np.ndarray, # wavelength to be concatenated
                 conditioning: np.ndarray = None, 
                 mask=None):
        # Input embedding
        x = nn.Dense(int(self.d_model))(x)  # (batch, seq_len, d_model)
        # Transformer layers
        if self.concat_wavelength:
            x = np.concatenate([x, wavelengthembd], axis=-1)
            conditioning = nn.Dense(x.shape[-1])(conditioning)
        else:
            wavelengthembd = nn.Dense(self.d_model)(wavelengthembd)
            x += wavelengthembd
        for _ in range(self.n_layers):
            # self attention 
            mask_attn = (
                    None if mask is None else mask[..., None] * mask[..., None, :]
                )
            x = MultiHeadAttentionBlock2(
                    n_heads=self.n_heads,
                    d_model= x.shape[-1],
                    d_mlp=self.d_mlp,
                )(x, x, mask_attn) # skip connection is in the attention block
            if conditioning is not None:
                # cross attention
                x = MultiHeadAttentionBlock2(
                    n_heads=self.n_heads,
                    d_model= x.shape[-1],
                    d_mlp=self.d_mlp,
                )(x, conditioning, mask[..., None]) # mask not needed from conditioning
                
        # Final LN as in pre-LN configuration
        x = nn.LayerNorm()(x)
        # Unembed; zero init kernel to propagate zero residual initially before training
        x = nn.Dense(self.n_input)(x)
        return x  

