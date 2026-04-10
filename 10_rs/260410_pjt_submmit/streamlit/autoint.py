import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Layer,
    LayerNormalization,
)
from tensorflow.keras.regularizers import l2


CAT_COLS = [
    "user_id",
    "movie_id",
    "movie_decade",
    "movie_year",
    "rating_year",
    "rating_month",
    "rating_decade",
    "genre1",
    "genre2",
    "genre3",
    "gender",
    "age",
    "occupation",
    "zip",
]


@dataclass
class ModelConfig:
    embedding_size: int = 16
    att_layer_num: int = 3
    att_head_num: int = 2
    att_res: bool = True
    dnn_hidden_units: Tuple[int, ...] = (64, 64)
    dnn_activation: str = "relu"
    l2_reg_dnn: float = 0.0
    l2_reg_embedding: float = 1e-5
    dnn_use_bn: bool = False
    dnn_dropout: float = 0.4
    init_std: float = 1e-4

    @classmethod
    def from_dict(cls, payload: Dict) -> "ModelConfig":
        config = dict(payload)
        config["dnn_hidden_units"] = tuple(config.get("dnn_hidden_units", (64, 64)))
        return cls(**config)


class FeaturesEmbedding(Layer):
    def __init__(
        self,
        field_dims: Sequence[int],
        embed_dim: int,
        l2_reg_embedding: float = 1e-5,
        init_std: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_fields = len(field_dims)
        self.embedding_layers = [
            Embedding(
                input_dim=field_dim,
                output_dim=embed_dim,
                embeddings_initializer=TruncatedNormal(stddev=init_std),
                embeddings_regularizer=l2(l2_reg_embedding),
                name=f"emb_field_{idx}",
            )
            for idx, field_dim in enumerate(field_dims)
        ]

    def call(self, inputs):
        embed_list = []
        for idx in range(self.num_fields):
            embed = self.embedding_layers[idx](inputs[:, idx])
            embed_list.append(tf.expand_dims(embed, axis=1))
        return tf.concat(embed_list, axis=1)


class MultiLayerPerceptron(Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int] = (32, 32),
        activation: str = "relu",
        l2_reg: float = 0.0,
        dropout_rate: float = 0.0,
        use_bn: bool = False,
        init_std: float = 1e-4,
        output_layer: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_bn = use_bn
        self.dense_layers: List[Dense] = []
        self.bn_layers: List[BatchNormalization | None] = []
        self.act_layers: List[tf.keras.layers.Activation] = []
        self.dropout_layers: List[Dropout] = []

        for units in tuple(hidden_units):
            self.dense_layers.append(
                Dense(
                    units,
                    activation=None,
                    kernel_initializer=TruncatedNormal(stddev=init_std),
                    kernel_regularizer=l2(l2_reg),
                )
            )
            self.bn_layers.append(BatchNormalization() if use_bn else None)
            self.act_layers.append(tf.keras.layers.Activation(activation))
            self.dropout_layers.append(Dropout(dropout_rate))

        self.out_layer = (
            Dense(
                1,
                activation=None,
                kernel_initializer=TruncatedNormal(stddev=init_std),
                kernel_regularizer=l2(l2_reg),
            )
            if output_layer
            else None
        )

    def call(self, inputs, training=False):
        x = inputs
        for dense, bn, act, drop in zip(
            self.dense_layers,
            self.bn_layers,
            self.act_layers,
            self.dropout_layers,
        ):
            x = dense(x)
            if bn is not None:
                x = bn(x, training=training)
            x = act(x)
            x = drop(x, training=training)
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x


class MultiHeadSelfAttention(Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        att_res: bool = True,
        init_std: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.att_res = att_res

        init = TruncatedNormal(stddev=init_std)
        self.W_q = Dense(embed_dim, use_bias=False, kernel_initializer=init)
        self.W_k = Dense(embed_dim, use_bias=False, kernel_initializer=init)
        self.W_v = Dense(embed_dim, use_bias=False, kernel_initializer=init)
        self.W_o = Dense(embed_dim, use_bias=False, kernel_initializer=init)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def _split_heads(self, x):
        batch_size = tf.shape(x)[0]
        num_fields = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, num_fields, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q = self._split_heads(self.W_q(inputs))
        k = self._split_heads(self.W_k(inputs))
        v = self._split_heads(self.W_v(inputs))

        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / scale
        weights = tf.nn.softmax(scores, axis=-1)
        ctx = tf.matmul(weights, v)

        ctx = tf.transpose(ctx, perm=[0, 2, 1, 3])
        batch_size = tf.shape(inputs)[0]
        num_fields = tf.shape(inputs)[1]
        ctx = tf.reshape(ctx, (batch_size, num_fields, self.embed_dim))

        out = self.W_o(ctx)
        if self.att_res:
            out = out + inputs
        return self.layer_norm(out)


class AutoIntMLP(Layer):
    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_size: int,
        att_layer_num: int = 3,
        att_head_num: int = 2,
        att_res: bool = True,
        dnn_hidden_units: Sequence[int] = (32, 32),
        dnn_activation: str = "relu",
        l2_reg_dnn: float = 0.0,
        l2_reg_embedding: float = 1e-5,
        dnn_use_bn: bool = False,
        dnn_dropout: float = 0.4,
        init_std: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_fields = len(field_dims)
        self.embed_output_dim = self.num_fields * embedding_size

        self.embedding = FeaturesEmbedding(
            field_dims,
            embedding_size,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
        )
        self.int_layers = [
            MultiHeadSelfAttention(
                embedding_size,
                att_head_num,
                att_res,
                init_std=init_std,
            )
            for _ in range(att_layer_num)
        ]
        self.flatten = Flatten()
        self.dnn_linear = Dense(
            1,
            use_bias=False,
            kernel_initializer=TruncatedNormal(stddev=init_std),
        )
        self.dnn = MultiLayerPerceptron(
            input_dim=self.embed_output_dim,
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            output_layer=True,
        )

    def call(self, inputs, training=False):
        embed_x = self.embedding(inputs)

        att_input = embed_x
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = self.flatten(att_input)
        att_output = tf.nn.relu(self.dnn_linear(att_output))

        dnn_input = tf.reshape(embed_x, (-1, self.embed_output_dim))
        dnn_output = self.dnn(dnn_input, training=training)

        return tf.sigmoid(att_output + dnn_output)


class AutoIntMLPModel(Model):
    def __init__(self, field_dims: Sequence[int], cfg: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.autoint_mlp = AutoIntMLP(
            field_dims=field_dims,
            embedding_size=cfg.embedding_size,
            att_layer_num=cfg.att_layer_num,
            att_head_num=cfg.att_head_num,
            att_res=cfg.att_res,
            dnn_hidden_units=cfg.dnn_hidden_units,
            dnn_activation=cfg.dnn_activation,
            l2_reg_dnn=cfg.l2_reg_dnn,
            l2_reg_embedding=cfg.l2_reg_embedding,
            dnn_use_bn=cfg.dnn_use_bn,
            dnn_dropout=cfg.dnn_dropout,
            init_std=cfg.init_std,
        )

    def call(self, inputs, training=False):
        return self.autoint_mlp(inputs, training=training)


def load_encoder_maps(path: Path) -> Dict[str, Dict[str, int]]:
    return json.loads(path.read_text(encoding="utf-8"))


def encode_frame(
    frame: pd.DataFrame,
    encoder_maps: Dict[str, Dict[str, int]],
    cat_cols: Sequence[str] = CAT_COLS,
) -> np.ndarray:
    encoded = np.empty((len(frame), len(cat_cols)), dtype=np.int32)

    for idx, col in enumerate(cat_cols):
        values = frame[col].fillna("no").astype(str).map(encoder_maps[col])
        if values.isna().any():
            unseen = sorted(frame.loc[values.isna(), col].astype(str).unique().tolist())[:5]
            raise ValueError(f"unseen labels in {col}: {unseen}")
        encoded[:, idx] = values.astype(np.int32).to_numpy()

    return encoded


def load_artifact_model(artifact_dir: str | Path):
    artifact_path = Path(artifact_dir)
    config = json.loads((artifact_path / "config.json").read_text(encoding="utf-8"))
    field_dims = np.load(artifact_path / "field_dims.npy").tolist()
    model_cfg = ModelConfig.from_dict(config["model_cfg"])
    encoder_maps = load_encoder_maps(artifact_path / "encoder_maps.json")

    model = AutoIntMLPModel(field_dims=field_dims, cfg=model_cfg)
    model(tf.zeros((1, len(field_dims)), dtype=tf.int32))
    model.load_weights(str(artifact_path / "model.weights.h5"))

    return model, encoder_maps, config


def predict_top_k(
    model: AutoIntMLPModel,
    pred_df: pd.DataFrame,
    encoder_maps: Dict[str, Dict[str, int]],
    top_k: int = 10,
    batch_size: int = 8192,
) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()

    features = encode_frame(pred_df, encoder_maps, CAT_COLS)
    scores = model.predict(features, batch_size=batch_size, verbose=0).reshape(-1)
    top_indices = np.argsort(scores)[::-1][:top_k]

    ranked = pred_df.iloc[top_indices].copy().reset_index(drop=True)
    ranked["score"] = scores[top_indices]
    return ranked
