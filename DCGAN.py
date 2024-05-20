import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"#setting env variable to avoid tensorflow print on standard error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

class DCGAN():
    def __init__(self,learning_rate,datasetpath):
        if not os.path.exists(datasetpath):
            os.mkdir(f"{os.getcwd()}\\{datasetpath}")
            print(f"Dataset folder has been created at {os.getcwd()}\\{datasetpath}\nPut images in the created folder to begin training.")
            exit()
            
        if not os.path.exists(f"{os.getcwd()}\\generated"):
            os.mkdir(f"{os.getcwd()}\\generated")
            print(f"Folder not found for generated images, one was created in the current working directory.")

        self.datasetpath=datasetpath
        self.latent_dim=128

        self.discriminator=keras.Sequential([
            keras.Input(shape=(64,64,3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1,activation="sigmoid"),
        ])

        self.generator=keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(8*8*128),
            layers.Reshape((8,8,128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")
        ])

        self.discriminator_optimizer=keras.optimizers.Adam(learning_rate)
        self.generator_optimizer=keras.optimizers.Adam(learning_rate)

        self.dataset=keras.preprocessing.image_dataset_from_directory(
            directory=self.datasetpath, label_mode=None, image_size=(64,64), batch_size=32, shuffle=True
        ).map(lambda x: x/255.0)

        self.loss_function=keras.losses.BinaryCrossentropy()

    def resize_images(self,dim):#if you use this it should be a tuple like (64,64)
        import PIL.Image as I
        for x in os.listdir(f"{os.getcwd()}\\{self.datasetpath}"):
            x=os.fsdecode(f"{self.datasetpath}\\{x}")
            if x.endswith(".py"):
                continue
            i=I.open(x)
            i=i.resize(dim)
            i.save(x)

    def train(self):
        for epoch in range(100):
            for idx, real in enumerate(tqdm(self.dataset)):
                batch_size=real.shape[0]
                random_latent_vectors = tf.random.normal(shape=(batch_size,self.latent_dim))

                fake=self.generator(random_latent_vectors)

                if idx % 100 == 0:
                    img = keras.preprocessing.image.array_to_img(fake[0])
                    img.save(f"generated/ImageAtEpoch_{epoch}_{idx}.png")

                with tf.GradientTape() as disc_tape:
                    loss_disc_real = self.loss_function(tf.ones((batch_size,1)),self.discriminator(real))
                    loss_disc_fake = self.loss_function(tf.zeros(batch_size,1),self.discriminator(fake))
                    loss_disc = (loss_disc_real + loss_disc_fake)/2

                grads = disc_tape.gradient(loss_disc, self.discriminator.trainable_weights)
                self.discriminator_optimizer.apply_gradients(
                    zip(grads, self.discriminator.trainable_weights)
                )

                with tf.GradientTape() as gen_tape:
                    fake = self.generator(random_latent_vectors)
                    output = self.discriminator(fake)
                    loss_gen = self.loss_function(tf.ones(batch_size,1),output)

                grads = gen_tape.gradient(loss_gen, self.generator.trainable_weights)
                self.generator_optimizer.apply_gradients(
                    zip(grads, self.generator.trainable_weights)
                )


network=DCGAN(1e-4,"blackhole_dataset_orig")
network.resize_images((225,225)) #example resize
network.train()