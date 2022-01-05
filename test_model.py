import header as h

m = h.keras.models.load_model('cache/m2')
print(m.predict(list(h.X_test[0].reshape(12, 1, 1000))), h.Y_test[0])