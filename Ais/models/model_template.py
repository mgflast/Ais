# This file outlines the template to which a custom SegmentationEditor model must conform. Most models that are available
# in the default library are simple tensorflow.keras Model objects, which is the object type that the SEModel implementation expects,
# but any class that implements the following methods can be used:
# .fit(train_x, train_y, epochs, batch_size, shuffle, callbacks=[])
# .predict(images)
# .count_params() -> int
# .save()
# .load()

# finally, every model file should include the fields 'title: str' and 'include: bool' and a function 'create(input_shape)' that returns the model object.

title = "Template_model"
include = False


def create(input_shape):
    return TemplateModel(input_shape)


class TemplateModel:
    def __init__(self, input_shape):
        self.img_shape = input_shape
        self.generator, self.discriminator = self.compile_custom_model()

    def compile_custom_model(self):
        # e.g.: compile generator, compile discriminator, return.
        return 0, 0

    def count_params(self):
        # e.g. return self.generator.count_params()
        # for the default models, the number of parameters that is returned is the amount that are involved in processing, not in training. So for e.g. a GAN, the discriminator params are not included.
        return 0

    def fit(self, train_x, train_y, epochs, batch_size=1, shuffle=True, callbacks=[]):
        for c in callbacks:
            c.set_params({**(c.params or {}), 'epochs': epochs})   # Keras 3 leaves params None until set

        # fit model, e.g.:
        for e in range(epochs):
            for i in range(len(train_x) // batch_size):
                # fit batch
                pass

                logs = {'loss': 0.0}
                for c in callbacks:
                    c.on_batch_end(i, logs)

    def predict(self, images):
        # e.g.: return self.generator.predict(images)
        return None

    def save(self, path):
        pass

    def load(self, path):
        pass


