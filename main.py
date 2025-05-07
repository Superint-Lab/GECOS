# ------------------- Clustering ------------------- #
import pandas as pd
import numpy as np

class CorrelationCluster:
    def __init__(self, corr_matrix, n_groups=3):
        self.corr_matrix = corr_matrix
        self.n_groups = n_groups
        self.groups = [set() for _ in range(n_groups)]
        self.num_variables = corr_matrix.shape[0]

        for idx in range(self.num_variables):
            self.groups[idx % n_groups].add(idx)

    def avg_corr(self, group):
        if len(group) < 2:
            return 0
        sub_matrix = self.corr_matrix[np.ix_(list(group), list(group))]
        return np.triu(sub_matrix, k=1).mean()

    def optimize_groups(self):
        changed = True
        while changed:
            changed = False
            for i in range(self.num_variables):
                from_group_idx = next(idx for idx, g in enumerate(self.groups) if i in g)
                best_group_idx = from_group_idx
                best_score = sum(self.avg_corr(g) for g in self.groups) / self.n_groups

                for to_group_idx in range(self.n_groups):
                    if to_group_idx == from_group_idx or len(self.groups[from_group_idx]) <= 1:
                        continue

                    self.groups[from_group_idx].remove(i)
                    self.groups[to_group_idx].add(i)
                    score = sum(self.avg_corr(g) for g in self.groups) / self.n_groups

                    if score > best_score:
                        best_score = score
                        best_group_idx = to_group_idx

                    self.groups[to_group_idx].remove(i)
                    self.groups[from_group_idx].add(i)

                if best_group_idx != from_group_idx:
                    self.groups[from_group_idx].remove(i)
                    self.groups[best_group_idx].add(i)
                    changed = True

        return self.groups


# Usage Example:
df = pd.read_csv('correlation_matrix_0721.csv', header=None).fillna(0)
corr_matrix = df.values
clusterer = CorrelationCluster(corr_matrix, n_groups=3)
final_groups = clusterer.optimize_groups()
for idx, group in enumerate(final_groups):
    print(f"Final group {idx + 1}: {group}")


# ------------------- RCTL Model ------------------- #
import tensorflow as tf
from keras.layers import Input, Conv1D, BatchNormalization, ReLU, Dropout, Add, LSTM, Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

class RCTLModel:
    def __init__(self, steps, f_list=[16, 32, 64, 64, 32, 16]):
        self.steps = steps
        self.f_list = f_list

    def build_model(self):
        inputs = Input(shape=(self.steps, 1), dtype=tf.float32)
        x = inputs
        outputs_store = []

        for i, filters in enumerate(self.f_list):
            dilation = 1 if i == 0 else 2 * i

            fx = Conv1D(filters, 3, padding='causal', dilation_rate=dilation)(x)
            fx = BatchNormalization()(fx)
            fx = ReLU()(fx)
            fx = Dropout(0.05)(fx)

            fx = Conv1D(filters, 3, padding='causal', dilation_rate=dilation)(fx)
            fx = BatchNormalization()(fx)
            fx = ReLU()(fx)
            fx = Dropout(0.05)(fx)

            shortcut = Conv1D(filters, 1, padding='causal')(x)
            block_output = Add()([shortcut, fx])
            block_output = LSTM(return_sequences=True, units=filters)(block_output)

            block_output_shortcut = Conv1D(filters, 1, padding='same')(x)
            block_output = Add()([block_output_shortcut, block_output])

            outputs_store.append(Conv1D(filters, 1, padding='same')(block_output))

            if i == 2:
                x = Add()([outputs_store[-1], block_output])
            elif i == 3:
                x = Add()([outputs_store[2], block_output])
            elif i == 4:
                x = Add()([outputs_store[1], block_output])
            elif i == 5:
                x = Add()([outputs_store[0], block_output])
            else:
                x = block_output

        final_shortcut = Conv1D(16, 1, padding='same')(inputs)
        x = Add()([x, final_shortcut])
        x = Flatten()(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.optimizers.Adam(), loss='MAE')

        return model

    def train(self, x_train, y_train, x_test, y_test, batch_size=512, epochs=1000):
        model = self.build_model()
        checkpoint = ModelCheckpoint('RCTL_model.h5', monitor='val_loss', save_best_only=True)

        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[checkpoint],
            batch_size=batch_size,
            verbose=1
        )

        return model, history
