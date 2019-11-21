import pickle as pkl
import config
import cnn

data = []
for i,pickle_file in enumerate(["X_train","Y_train","X_validation","Y_validation","X_test","Y_test"]):
    with open("data/"+pickle_file+".pkl","rb") as pklf:
            data.append(pkl.load(pklf))

X_train, Y_train, X_validation, Y_validation, X_test, Y_test = data
print(X_train.shape)
print("Creating model...")
model = cnn.create_model(config.n_classes, config.image_size, config.learning_rate)
model.summary()
print("Training the model...")
model.fit(X_train, Y_train, epochs=config.epochs, batch_size=config.batch_size, shuffle=True, validation_data=(X_validation, Y_validation),)
print("Model trained!")
print("Saving the weights...")
model.save_weights('audio_event_classifier.h5')
print("Weights saved!")
