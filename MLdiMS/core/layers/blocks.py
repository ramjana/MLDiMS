#initial version
import functools
from pprint import pprint
from typing import Any, Callable, Dict, Literal, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .baseops import (
        Dense,
        RMSNorm,
        DenseGeneral,
        Dropout,
        relu,
        silu,
        AttnOutput,
        SinusoidalPositionalEncoding,
        LearnedPositionalEncoding,
        dot_product_attention,
        softmax,
        Embed,
        Linear,
        gelu
    )



##MLPBlock building block of the transformer
#linear layer scaling up the hidden dimensionality , non-linearity 
#and a linear layer scaling down the hidden dimensionality

class MLPBlockInput(nn.Module):
    features: int
    data_type : jnp.dtype
    use_norm : bool = False
    use_bias : bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_norm:
            x = RMSNorm(dtype=self.data_type, name="MLPPreNorm")(x)
        x = Dense(
            features=self.features,
            kernel_init = self.kernel_init,
            use_bias = self.use_bias,
            dtype=self.data_type,
            name="MLPDense"
        )(x)
        #x = gelu()(x)
        x = nn.gelu(x)
        return x

class MLPBlockOutput(nn.Module):
    features: int
    data_type : jnp.dtype
    use_bias : bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = Dense(
            features=self.features,
            kernel_init = self.kernel_init,
            use_bias = self.use_bias,
            dtype=self.data_type,
            name="MLPDense"
        )(x)
        return x

class MLPBlock(nn.Module):
     hidden_dim: int
     embedding_dim: int
     data_type: jnp.dtype
     use_norm: bool = False
     use_bias: bool = True

     def setup(self):
         super().setup()

     @nn.compact
     def __call__(self, x: jax.Array) -> jax.Array:
         input_features = x[-1]

         x = RMSNorm(data_type=self.data_type,name="MLPPrenorm")(x)
         x = MLPBLockInput(
             features=self.hidden_dim,
             use_norm=self.use_norm,
             use_bias=self.use_bias,
             name="MLPInput")(x)
         x = MLPBLockOutput(
             features=self.embedding_dim,
             use_bias=self.use_bias,
             name="MLPOutput")(x)
         return x

class QKVProjection(nn.Module):
    
    use_bias: bool
    head_dim: int
    num_heads: int
    data_type: jnp.dtype
    kernel_init : Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros
    normalize_qk : bool = False

    def setup(self):
        super().setup()


    @nn.compact
    def __call__(self,x: jax.Array) -> tuple [jax.Array,jax.Array,jax.Array]:
        q = DenseGeneral(
                (self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.data_type,
                use_bias = self.use_bias,
                name="QProj",
        )(x)
        k = DenseGeneral(
                (self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.data_type,
                use_bias = self.use_bias,
                name="KProj",
        )(x)
        v = DenseGeneral(
                (self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.data_type,
                use_bias = self.use_bias,
                name="VProj",
        )(x)
        if self.normalize_qk:
            q = RMSNorm(
                dtype=self.data_type,
                name="Qnorm",
            )(q)
            k = RMSNorm(
                dtype=self.data_type,
                name="Knorm",
            )(k)
        return q,k,v

class AttnOut(nn.Module):
    features: int
    data_type: jnp.dtype
    use_bias: bool = True
    kernel_init : Callable = nn.initializers.lecun_normal()

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        x = DenseGeneral(
            features=self.features,
            axes=(-2,-1),
            data_type=self.data_type,
            use_bias = self.use_bias,
            kernel_init=self.kernel_init,
        )(x)
        return x

class MultiHeadAttention(nn.Module):
     embedding_dim : int  #model dimension 
     num_heads: int
     use_bias_qkv: bool
     attention: Callable
     data_type: jnp.dtype
     mask : jax.Array | None = None
     normalize_qk: bool = False

     def setup(self):
         super().setup()

     @nn.compact
     def __call__(self,x: jax.Array) -> jax.Array:
         features = x.shape[-1]
         #Normalize lyaer
         x = RMSNorm(dtype=self.data_type,name="PreNorm")(x)
         #QKV projection
         attn_dim = self.embedding_dim//self.num_heads
         q,k,v = QKVProjection(
                 head_dim=attn_dim,
                 num_heads = self.num_heads,
                 use_bias = self.use_bias_qkv,
                 normalize_qk = self.normalize_qk,
                 data_type = self.data_type,
                 name="QKVProjection"
         )(x)
         x = dot_product_attention(
             q,k,v,self.mask,
             )(x)
         #print(x.shape)
         x = AttnOutput(
             features=features,
             data_type = self.data_type,
             name="AttnOut",
         )(x)
         #print(x.shape)
         return x

class OutputLayer(nn.Module):
    num_outputs: int
    data_type: jnp.dtype = jnp.float16


    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:

        x = RMSNorm(
            dtype=self.data_type,
            )(x)
        x = Dense(
            features=self.num_ouputs,
            dtype=self.data_type,
            )(x)
        return x

class TransformerBlock(nn.Module):
    embedding_dim: int = 1024
    num_heads: int = 8
    causal_mask: bool = True
    normalize_qk: bool = False
    use_bias_qkv: bool = True
    hidden_dim: int = 4096
    train : bool = True
    data_type: jnp.dtype = jnp.float16
    dropout_rate: jnp.float32 = 0.1

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        #attention layer block
        attn_out  = MultiHeadAttention(
            embedding_dim = self.embedding_dim,
            num_heads = self.num_heads,
            use_bias_qkv = self.use_bias_qkv,
            mask = self.mask,
            data_type= self.data_type,
            )(x)
        attn_out = Dropout(rate=self.dropout_rate, deterministic=not self.train)(attn_out)
        x = x + attn_out
        #MLP block
        mlp_out = MLPBlock(
            embedding_dim = self.embedding_dim,
            hidden_dim=self.hidden_dim,
            use_norm=False,
            use_bias=True,
            data_type=self.data_type,
        )(x)
        mlp_out = Dropout(
            rate=self.dropout_rate,
            deterministic = not self.train,
        )(mlp_out)
        x = x + mlp_out
        return x

class RotaryPositionalEncoding(nn.Module):
    embedding_dim: int
    shard_axis_name: str
    base_exponent: int = 10000

    def setup(self):
        assert(embedding_dim%2)


    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        exponents = jnp.arange(0,embedding_dim,2,dtype=x.dtype)
        ## calculate frequnecy of each token normalized by embedding dimension
        freq_per_token = (1/ (base_exponent ** (exponents/self.embedding_dim)))

        token = jnp.arange(0,x.shape[1], dtype=x.dtype)
        token_phase = jnp.einsum("i,j -> ij",token,freq_per_token)
        token_phase = jnp.tile(token_phase,reps=(1,2))[None,:,None,:]
        x = x*jnp.cos(token_phase) + jnp.sin(token_phase)*jnp.rotate_half(x)
        return x

    @staticmethod
    def rotate_half(self, x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x,2, axis=-1)
        return jnp.concatenate((-x2,x1),axis=-1)

class PositionalEncoding(nn.Module):
  embedding_dim: int
  shard_axis_name: str
  base_exponent: int = 10000

  def __call__(
      self,
      x: jax.Array,
      position: jax.Array,
  ) -> jax.Array:

    seq_len, num_features = x.shape[-2:]
    tp_size = jax.lax.psum(1, self.shard_axis_name)
    num_timescales = self.embedding_dim // 2 // tp_size
    log_timescale_increment = jnp.log(float(self.base_exponent)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    if position is None:
        position = jnp.arange(0, seq_len, dtype=jnp.float32)
        #expand axis in batch dimension
        position = jnp.expand_dims(position,0)
    inv_timescales = jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
    #expand dimension in embedding dimension
    position = position[:, :, jnp.newaxis]
    inv_timescales = inv_timescales[jnp.newaxis, jnp.newaxis, :]
    scaled_time = position * inv_timescales
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
    # signal = jnp.pad(signal, [[0, jnp.mod(self.embedding_dims, 2)]])
    position_embedding = signal.astype(jnp.float32)
    pos_emb = jnp.expand_dims(position_embedding, axis=range(x.ndim - 2))
    return x + position_embedding

class OldPositionalEncoding(nn.Module):
    embedding_dim:int
    shard_axis_name: str
    dtype: jnp.dtype = jnp.float32
    encoding_type: str = "learned"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        #FIXME use argument
        tp_size = jax.lax.psum(1, self.shard_axis_name)
        tp_index = jax.lax.axis_index(self.shard_axis_name)
        seq_len, num_feats = x.shape[-2:]
        #print(num_feats)
        if self.encoding_type == "learned":
            pos_emb = self.param(
                "pos_emb",
                nn.initializers.normal(stddev=0.02),
                (seq_len, num_feats),
            )
        elif self.encoding_type == "sinusoidal":
            # Adjusted to multi-device setting.
            position = jnp.arange(0, seq_len, dtype=jnp.float32)[:, None]
            div_term = jnp.exp(
                jnp.arange(tp_index * num_feats, (tp_index + 1) * num_feats, 2)
                * (-np.log(10000.0) / (tp_size * num_feats))
            )
            pos_emb = jnp.stack(
                [jnp.sin(position * div_term), jnp.cos(position * div_term)], axis=-1
            )
            pos_emb = jnp.reshape(pos_emb, (seq_len, num_feats))
        else:
            raise ValueError(
                f"Unknown positional encoding type: {self.config.positional_encoding_type}"
            )
        pos_emb = pos_emb.astype(
            x.dtype
        )  # Cast to the same dtype as the input, e.g. support bfloat16.
        pos_emb = jnp.expand_dims(pos_emb, axis=range(x.ndim - 2))
        x = x + pos_emb
        return x

class InputEmbedding(nn.Module):
    seq_len: int
    shard_axis_name: str
    data_type: jnp.dtype
    vocab_size: int
    embedding_dim: int
    encoding_type: Literal["learned","sinusoidal"] = "sinusoidal"

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        num_devices = jax.lax.psum(1,self.shard_axis_name)
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim//num_devices,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=self.data_type,
            name="token_embedding",
        )(x)
        x = OldPositionalEncoding(
            embedding_dim = self.embedding_dim,    
            shard_axis_name=self.shard_axis_name,
            name="PositionalEncoding")(x)
        return x
