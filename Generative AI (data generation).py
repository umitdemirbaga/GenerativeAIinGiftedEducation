import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv("moralSensitivity.csv")

# Normalize the data
normalized_df = (df - df.mean()) / df.std()

# Define the dimensionality of the input noise
latent_dim = 100

# Define the generator model
def build_generator(latent_dim):
    inputs = Input(shape=(latent_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(df.shape[1], activation='tanh')(x)
    model = Model(inputs, outputs)
    return model

# Define the discriminator model
def build_discriminator(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

# Compile the discriminator
discriminator = build_discriminator(df.shape[1])
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])

# Compile the combined model (generator + discriminator)
generator = build_generator(latent_dim)
discriminator.trainable = False
z = Input(shape=(latent_dim,))
synthetic_data = generator(z)
validity = discriminator(synthetic_data)
combined_model = Model(z, validity)
combined_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))

# Train the GAN
epochs = 10000
batch_size = 32
for epoch in range(epochs):
    # Select a random batch of real data
    idx = np.random.randint(0, normalized_df.shape[0], batch_size)
    real_data = normalized_df.iloc[idx]

    # Generate a batch of synthetic data
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    synthetic_data_batch = generator.predict(noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(synthetic_data_batch, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined_model.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Generate synthetic data
num_samples = 1000
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_data = generator.predict(noise)

# Convert the synthetic data back to original scale if necessary
# synthetic_data = synthetic_data * df.std() + df.mean()

# Save synthetic data to CSV
synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
synthetic_df.to_csv("synthetic_data.csv", index=False)

# Load the synthetic data
synthetic_data_org = pd.read_csv("synthetic_data.csv")

# Define a custom Min-Max scaling function to preserve integer values
def custom_min_max_scaling(data, min_val, max_val):
    return (data - data.min()) / (data.max() - data.min()) * (max_val - min_val) + min_val

# Scale the synthetic data using the custom Min-Max scaling function
min_val = df.min()
max_val = df.max()
synthetic_data_scaled = custom_min_max_scaling(synthetic_data_org, min_val, max_val)

# Convert scaled data back to integers
synthetic_data_scaled_int = synthetic_data_scaled.round().astype(int)

# Create DataFrame and save to CSV
synthetic_df_scaled = pd.DataFrame(synthetic_data_scaled_int, columns=df.columns)
synthetic_df_scaled.to_csv("synthetic_data_scaled.csv", index=False)
