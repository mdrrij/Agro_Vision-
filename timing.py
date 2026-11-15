import time
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("agrovision_cnn.keras")
x = np.random.randint(0, 255, (1, 128, 128, 3)).astype("float32")
model.predict(x, verbose=0)  # warm-up

t = time.time()
for _ in range(50):
    model.predict(x, verbose=0)
print((time.time() - t) / 50 * 1000, "ms per image")